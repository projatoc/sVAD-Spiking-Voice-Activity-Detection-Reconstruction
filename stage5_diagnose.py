"""Stage 5: reproducible attention and loss diagnostics on validation audio.

Place beside model.py, data.py, metrics.py. No model weights are updated.
This script loads existing Stage 5 checkpoints and reports per-SNR losses,
spike activity, mask behavior, and confusion rates for identical audio.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import Config, SVAD, frame_audio
from data import PairedAudioDataset
from metrics import confusion_counts, rates


def checkpoint_config(raw):
    return Config(**raw) if isinstance(raw, dict) else raw


def device_from_arg(request):
    if request != 'auto':
        return request
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def describe_checkpoint(ckpt):
    args = ckpt.get('train_args', {})
    if not isinstance(args, dict):
        args = vars(args)
    return {
        'epoch': ckpt.get('epoch'),
        'lambda_mse': args.get('lambda_mse'),
        'clean_target': args.get('clean_target'),
        'recurrent_type': args.get('recurrent_type'),
        'saved_validation_metrics': ckpt.get('val_metrics'),
    }


def evaluate(model, loader, cfg, device, weights):
    model.eval()
    totals = dict(tp=0, fn=0, fp=0, tn=0)
    sums = {name: 0. for name in [
        'ce', 'weighted_ce', 'mse', 'feature_rate', 'mask_rate',
        'attended_rate', 'all_zero_mask_rate', 'mask_given_active_feature',
        'mask_on_speech', 'mask_on_nonspeech'
    ]}
    n_frames = n_feature_slots = n_active_features = 0
    n_speech_frames = n_nonspeech_frames = 0
    with torch.no_grad():
        for batch in loader:
            noisy = frame_audio(batch['noisy'].to(device), cfg)
            clean = frame_audio(batch['clean'].to(device), cfg)
            labels = batch['labels'].to(device)
            out = model(noisy, clean)
            logits = out['logits']
            feats, mask, attended, reference = (
                out['features'], out['mask'], out['attended'], out['clean_features'])
            assert logits.shape[:2] == labels.shape, 'Labels and frames are misaligned'
            assert feats.shape == mask.shape == attended.shape == reference.shape
            b, t, f = mask.shape
            n_frames += b * t
            n_feature_slots += b * t * f
            active = feats > 0.5
            speech = labels == 1
            nonspeech = labels == 0
            n_active_features += int(active.sum().item())
            n_speech_frames += int(speech.sum().item())
            n_nonspeech_frames += int(nonspeech.sum().item())
            sums['ce'] += float(F.cross_entropy(logits.reshape(-1,2), labels.reshape(-1), reduction='sum').item())
            # Weighted CE uses the same class weights saved by training; PyTorch
            # weighted mean normalizes by the sum of weights, not batch size.
            flat_labels = labels.reshape(-1)
            weighted_losses = F.cross_entropy(logits.reshape(-1,2), flat_labels, weight=weights, reduction='none')
            sums['weighted_ce'] += float(weighted_losses.sum().item())
            sums['mse'] += float(F.mse_loss(attended, reference, reduction='sum').item())
            sums['feature_rate'] += float(feats.sum().item())
            sums['mask_rate'] += float(mask.sum().item())
            sums['attended_rate'] += float(attended.sum().item())
            sums['all_zero_mask_rate'] += float((mask.sum(dim=-1) == 0).sum().item())
            sums['mask_given_active_feature'] += float((mask * active).sum().item())
            sums['mask_on_speech'] += float(mask[speech].sum().item())
            sums['mask_on_nonspeech'] += float(mask[nonspeech].sum().item())
            counts = confusion_counts(logits, labels)
            for k in totals:
                totals[k] += counts[k]

    if n_frames == 0:
        raise ValueError('No validation frames')
    # Normalized weighted CE uses the weight sum as the denominator.
    # Actual weighted denominator based on true label counts.
    weight_denom = (totals['tp'] + totals['fn']) * weights[1].item() + (totals['tn'] + totals['fp']) * weights[0].item()
    return {
        **rates(totals),
        'ce': sums['ce'] / n_frames,
        'weighted_ce': sums['weighted_ce'] / weight_denom,
        'mse': sums['mse'] / n_feature_slots,
        'encoder_spike_rate': sums['feature_rate'] / n_feature_slots,
        'attention_mask_rate': sums['mask_rate'] / n_feature_slots,
        'attended_spike_rate': sums['attended_rate'] / n_feature_slots,
        'all_zero_mask_frame_fraction': sums['all_zero_mask_rate'] / n_frames,
        'attention_pass_rate_on_active_features': sums['mask_given_active_feature'] / max(1, n_active_features),
        'mask_rate_on_speech_frames': sums['mask_on_speech'] / max(1, n_speech_frames * cfg.features),
        'mask_rate_on_nonspeech_frames': sums['mask_on_nonspeech'] / max(1, n_nonspeech_frames * cfg.features),
        'counts': totals,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--checkpoint', action='append', required=True, help='Repeat for A, B, C')
    p.add_argument('--snrs', type=float, nargs='+', default=[-10, 0, 15])
    p.add_argument('--split', choices=['val','test'], default='val')
    p.add_argument('--segment-s', type=float, default=1.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', choices=['auto','cpu','mps','cuda'], default='auto')
    p.add_argument('--output', default='runs/stage5_attention_diagnostics.json')
    args = p.parse_args()
    device = device_from_arg(args.device)
    results = {'split': args.split, 'snrs': args.snrs, 'models': {}}
    loaders = {}
    for path in args.checkpoint:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        cfg = checkpoint_config(ckpt['config'])
        model = SVAD(cfg).to(device)
        model.load_state_dict(ckpt['state_dict'], strict=True)
        w = torch.tensor(ckpt.get('class_weights', [1., 1.]), dtype=torch.float32, device=device)
        print(f'\n{path} | {describe_checkpoint(ckpt)}', flush=True)
        by_snr = {}
        for snr in args.snrs:
            # Dataset seed and SNR are matched across architectures; this uses
            # existing Stage 5 validation data logic, without modifying the data.
            ds = PairedAudioDataset(args.manifest, args.split, cfg,
                                    args.segment_s, [snr], random_crop=False, seed=args.seed)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
            result = evaluate(model, loader, cfg, device, w)
            by_snr[str(snr)] = result
            print(f'  {snr:+g} dB: HTER={result["hter"]:.4f} '
                  f'CE={result["ce"]:.4f} MSE={result["mse"]:.4f} '
                  f'encoder={result["encoder_spike_rate"]:.3f} '
                  f'mask={result["attention_mask_rate"]:.3f} '
                  f'attended={result["attended_spike_rate"]:.3f} '
                  f'zero-mask-frames={result["all_zero_mask_frame_fraction"]:.3f}', flush=True)
        results['models'][path] = {'checkpoint': describe_checkpoint(ckpt), 'metrics': by_snr}
    path = Path(args.output)
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(results, indent=2, default=str))
    print(f'\nSaved: {path}', flush=True)

if __name__ == '__main__':
    main()
