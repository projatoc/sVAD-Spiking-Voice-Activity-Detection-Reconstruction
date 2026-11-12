"""Stage 5: controlled sweep of attention MSE coefficient on the real-speech pilot.

Place alongside model.py, data.py, metrics.py. Paired clean/noisy inputs are
required when lambda_mse > 0. Checkpoints chosen by mean validation HTER over
selected SNRs; the test split is never used for selection.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import Config, SVAD, frame_audio, parameter_breakdown
from data import PairedAudioDataset
from metrics import confusion_counts, rates


def get_class_weights(dataset):
    counts = [0, 0]
    for example in dataset:
        y = example['labels']
        counts[0] += int((y == 0).sum())
        counts[1] += int((y == 1).sum())
    if min(counts) == 0:
        raise ValueError(f'Both classes needed for weighted CE: {counts}')
    n = sum(counts)
    return torch.tensor([n / (2 * x) for x in counts], dtype=torch.float32), counts


def evaluate(model, loader, cfg, device):
    model.eval()
    totals = dict(tp=0, fn=0, fp=0, tn=0)
    with torch.no_grad():
        for batch in loader:
            logits = model(frame_audio(batch['noisy'].to(device), cfg))['logits']
            c = confusion_counts(logits, batch['labels'].to(device))
            for k in totals:
                totals[k] += c[k]
    return rates(totals), totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--segment-s', type=float, default=1.5)
    ap.add_argument('--snrs', type=float, nargs='+', default=[-10, -5, 0, 5, 10, 15])
    ap.add_argument('--val-snrs', type=float, nargs='+', default=[-10, -5, 0, 5, 10, 15])
    ap.add_argument('--lambda-mse', type=float, required=True)
    ap.add_argument('--clean-target', choices=['joint', 'detach'], default='joint',
                    help='joint = previous implementation; detach = stop gradients through clean reference')
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--recurrent-type', choices=['diagonal', 'dense'], default='diagonal')
    ap.add_argument('--attention-width1', type=int, default=31)
    ap.add_argument('--attention-width2', type=int, default=31)
    ap.add_argument('--output', default='runs/stage5_best.pt')
    ap.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    args = ap.parse_args()
    if args.lambda_mse < 0:
        raise ValueError('lambda-mse must be >= 0')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = Config(lambda_mse=args.lambda_mse, recurrent_type=args.recurrent_type,
                 attention_width1=args.attention_width1, attention_width2=args.attention_width2)
    device = args.device
    if device == 'auto':
        device = ('cuda' if torch.cuda.is_available() else
                  'mps' if torch.backends.mps.is_available() else 'cpu')
    train_data = PairedAudioDataset(args.manifest, 'train', cfg, args.segment_s,
                                    args.snrs, random_crop=True, seed=args.seed)
    count_data = PairedAudioDataset(args.manifest, 'train', cfg, args.segment_s,
                                    [args.val_snrs[0]], random_crop=False, seed=args.seed)
    w, class_counts = get_class_weights(count_data)
    print('Train class counts:', class_counts, 'class weights:', w.tolist(), flush=True)
    weights = w.to(device)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loaders = {}
    for snr in args.val_snrs:
        ds = PairedAudioDataset(args.manifest, 'val', cfg, args.segment_s,
                                [snr], random_crop=False, seed=args.seed)
        val_loaders[str(snr)] = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = SVAD(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=40, gamma=0.1)
    print('Device:', device, 'Parameters:', parameter_breakdown(model), flush=True)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = total_ce = total_mse = 0.0
        n = 0
        for batch in train_loader:
            noisy = frame_audio(batch['noisy'].to(device), cfg)
            clean = (frame_audio(batch['clean'].to(device), cfg)
                     if args.lambda_mse != 0 else None)
            labels = batch['labels'].to(device)
            out = model(noisy, clean)
            ce = F.cross_entropy(out['logits'].reshape(-1, 2), labels.reshape(-1), weight=weights)
            if clean is None:
                mse = ce.new_zeros(())
            else:
                reference = out['clean_features']
                if args.clean_target == 'detach':
                    reference = reference.detach()
                mse = F.mse_loss(out['attended'], reference)
            loss = ce + args.lambda_mse * mse
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = labels.shape[0]
            total += float(loss.item()) * bs
            total_ce += float(ce.item()) * bs
            total_mse += float(mse.item()) * bs
            n += bs
        scheduler.step()
        by_snr = {}
        counts_by_snr = {}
        for key, loader in val_loaders.items():
            by_snr[key], counts_by_snr[key] = evaluate(model, loader, cfg, device)
        mean_hter = float(np.mean([m['hter'] for m in by_snr.values()]))
        row = dict(epoch=epoch, train_loss=total/n, train_ce=total_ce/n,
                   train_mse=total_mse/n, val_mean_hter=mean_hter, val_by_snr=by_snr)
        history.append(row)
        print(f'epoch {epoch}: loss={total/n:.4f} CE={total_ce/n:.4f} '
              f'MSE={total_mse/n:.4f} val_mean_HTER={mean_hter:.4f}', flush=True)
        if mean_hter < best:
            best = mean_hter
            torch.save(dict(state_dict=model.state_dict(), config=vars(cfg),
                            train_args=vars(args), val_metrics={'mean_hter': mean_hter, 'by_snr': by_snr},
                            val_counts=counts_by_snr, epoch=epoch,
                            class_weights=w.tolist(),
                            label_warning='Energy-estimated speech intervals, not verified ground truth.'), path)
            print('  saved new best:', path, flush=True)
    history_file = path.with_suffix('.json')
    history_file.write_text(json.dumps(history, indent=2))
    print('Best mean validation HTER:', best, 'History:', history_file)


if __name__ == '__main__':
    main()
