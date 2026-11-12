"""Stage 6: train full attention vs all-ones pass-through attention ablation.

Place beside model.py, data.py, metrics.py. Requires same manifest as Stage 5.
Stage 6 trains with lambda_mse=0 to isolate the architectural attention block.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import Config, SVAD, frame_audio, parameter_breakdown
from data import PairedAudioDataset
from metrics import confusion_counts, rates


class PassThroughAttention(nn.Module):
    """No parameters; returns all ones, so x * mask == x."""
    def forward(self, features):
        return torch.ones_like(features)


def build_model(cfg, attention_mode):
    model = SVAD(cfg)
    if attention_mode == 'bypass':
        model.attention = PassThroughAttention()
    elif attention_mode != 'full':
        raise ValueError(f'Unknown attention mode: {attention_mode}')
    return model


def get_class_weights(dataset):
    counts = [0, 0]
    for example in dataset:
        labels = example['labels']
        counts[0] += int((labels == 0).sum())
        counts[1] += int((labels == 1).sum())
    if min(counts) == 0:
        raise ValueError(f'Both classes needed for weighted CE: {counts}')
    n = sum(counts)
    return torch.tensor([n / (2 * x) for x in counts], dtype=torch.float32), counts


def evaluate(model, loader, cfg, device):
    model.eval()
    total = dict(tp=0, fn=0, fp=0, tn=0)
    with torch.no_grad():
        for batch in loader:
            frames = frame_audio(batch['noisy'].to(device), cfg)
            logits = model(frames)['logits']
            cc = confusion_counts(logits, batch['labels'].to(device))
            for key in total:
                total[key] += cc[key]
    return rates(total), total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--attention-mode', choices=['full', 'bypass'], required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--segment-s', type=float, default=1.5)
    parser.add_argument('--snrs', type=float, nargs='+', default=[-10, -5, 0, 5, 10, 15])
    parser.add_argument('--val-snrs', type=float, nargs='+', default=[-10, -5, 0, 5, 10, 15])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument("--recurrent-type", choices=["diagonal", "dense"], default="diagonal")
    parser.add_argument("--recurrent-hidden", type=int, default=32)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be >= 1')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    if device == 'auto':
        device = ('cuda' if torch.cuda.is_available() else
                  'mps' if torch.backends.mps.is_available() else 'cpu')
    cfg = Config(lambda_mse=0.0, recurrent_type=args.recurrent_type, recurrent_hidden=args.recurrent_hidden, attention_width1=31, attention_width2=31)
    train_data = PairedAudioDataset(args.manifest, 'train', cfg, args.segment_s,
                                    args.snrs, random_crop=True, seed=args.seed)
    count_data = PairedAudioDataset(args.manifest, 'train', cfg, args.segment_s,
                                    [args.val_snrs[0]], random_crop=False, seed=args.seed)
    weights, counts = get_class_weights(count_data)
    print('Training frame class counts:', counts, 'weights:', weights.tolist(), flush=True)
    weights = weights.to(device)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)

    val_loaders = {}
    for snr in args.val_snrs:
        dataset = PairedAudioDataset(args.manifest, 'val', cfg, args.segment_s,
                                     [snr], random_crop=False, seed=args.seed)
        val_loaders[str(snr)] = DataLoader(dataset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=0)

    model = build_model(cfg, args.attention_mode).to(device)
    params = parameter_breakdown(model)
    total_params = sum(p.numel() for p in model.parameters())
    print('Device:', device, 'attention:', args.attention_mode,
          'parameters:', params, 'total:', total_params, flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    best = float('inf')
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        weighted_loss, items = 0.0, 0
        for batch in train_loader:
            frames = frame_audio(batch['noisy'].to(device), cfg)
            labels = batch['labels'].to(device)
            logits = model(frames)['logits']
            loss = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1), weight=weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bs = labels.size(0)
            weighted_loss += float(loss.item()) * bs
            items += bs
        scheduler.step()
        by_snr, counts_by_snr = {}, {}
        for snr, loader in val_loaders.items():
            by_snr[snr], counts_by_snr[snr] = evaluate(model, loader, cfg, device)
        mean_hter = float(np.mean([m['hter'] for m in by_snr.values()]))
        row = dict(epoch=epoch, train_ce=weighted_loss / items,
                   val_mean_hter=mean_hter, val_by_snr=by_snr)
        history.append(row)
        print(f'{args.attention_mode} seed={args.seed} epoch={epoch}: '
              f'CE={weighted_loss/items:.4f} val_mean_HTER={mean_hter:.4f}', flush=True)
        if mean_hter < best:
            best = mean_hter
            torch.save(dict(
                state_dict=model.state_dict(), config=vars(cfg), train_args=vars(args),
                attention_mode=args.attention_mode, parameter_count=total_params,
                val_metrics={'mean_hter': mean_hter, 'by_snr': by_snr},
                val_counts=counts_by_snr, epoch=epoch, class_weights=weights.tolist(),
                label_warning='Energy-estimated speech intervals, not verified ground truth.',
            ), path)
            print('  saved best:', path, flush=True)
    history_path = path.with_suffix('.json')
    history_path.write_text(json.dumps(history, indent=2))
    print('Completed', args.epochs, 'epochs; best validation mean HTER:', best,
          'history:', history_path)

if __name__ == '__main__':
    main()
