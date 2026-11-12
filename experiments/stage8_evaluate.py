"""Stage 8: evaluate saved sVAD checkpoints with no retraining or model selection.

Place beside the modified Stage 6 train script, model.py, data.py and metrics.py.
Requires existing Stage 6/7 checkpoints. Writes JSON for later documentation.
"""
import argparse
import json
import statistics
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import Config
from data import PairedAudioDataset
from stage6_train import build_model, evaluate

FAMILIES = {
    'full': 'stage6_full_seed{seed}.pt',
    'light_bypass': 'stage6_bypass_seed{seed}.pt',
    'matched_bypass': 'stage7_matched_bypass_seed{seed}.pt',
}
EXPECTED = {
    'full': {'attention_mode': 'full', 'recurrent_type': 'diagonal', 'recurrent_hidden': 32, 'parameter_count': 4313},
    'light_bypass': {'attention_mode': 'bypass', 'recurrent_type': 'diagonal', 'recurrent_hidden': 32, 'parameter_count': 2030},
    'matched_bypass': {'attention_mode': 'bypass', 'recurrent_type': 'dense', 'recurrent_hidden': 45, 'parameter_count': 4322},
}


def inspect_checkpoint(ckpt, path, family):
    c = ckpt.get('config', {})
    if isinstance(c, Config):
        c = vars(c)
    mode = ckpt.get('attention_mode')
    expected = EXPECTED[family]
    observed = {
        'attention_mode': mode,
        'recurrent_type': c.get('recurrent_type'),
        'recurrent_hidden': c.get('recurrent_hidden'),
        'parameter_count': ckpt.get('parameter_count'),
    }
    mismatches = [
        f'{key}: expected {value!r}, found {observed[key]!r}'
        for key, value in expected.items() if observed[key] != value
    ]
    if mismatches:
        raise ValueError(f'Checkpoint configuration mismatch at {path}: ' + '; '.join(mismatches))
    if 'state_dict' not in ckpt:
        raise KeyError(f'Missing state_dict in {path}')
    return c


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--split', default='test', choices=['val', 'test'])
    p.add_argument('--runs', default='runs')
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 2026])
    p.add_argument('--snrs', nargs='+', type=float, default=[-10, -5, 0, 5, 10, 15])
    p.add_argument('--segment-s', type=float, default=1.5)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    p.add_argument('--output', default='runs/stage8_final_evaluation.json')
    args = p.parse_args()

    device = args.device
    if device == 'auto':
        device = ('cuda' if torch.cuda.is_available() else
                  'mps' if torch.backends.mps.is_available() else 'cpu')

    output = {'protocol': {
        'split': args.split,
        'manifest': args.manifest,
        'seeds': args.seeds,
        'snrs_db': args.snrs,
        'segment_s': args.segment_s,
        'batch_size': args.batch_size,
        'noise_seed_policy': 'dataset seed equals checkpoint training seed; paired models use the same seed',
        'caveat': 'Exploratory held-out split previously inspected; not a pristine independent confirmatory test.',
    }, 'runs': {}, 'summary': {}, 'paired_effects': {}}

    for family, pattern in FAMILIES.items():
        output['runs'][family] = {}
        for seed in args.seeds:
            path = Path(args.runs) / pattern.format(seed=seed)
            if not path.is_file():
                raise FileNotFoundError(f'Missing model checkpoint: {path}')
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            cfg_dict = inspect_checkpoint(ckpt, path, family)
            cfg = Config(**cfg_dict)
            model = build_model(cfg, ckpt['attention_mode'])
            model.load_state_dict(ckpt['state_dict'], strict=True)
            model = model.to(device)
            by_snr = {}
            hters = []
            for snr in args.snrs:
                data = PairedAudioDataset(
                    args.manifest, args.split, cfg, args.segment_s,
                    [snr], random_crop=False, seed=seed,
                )
                loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=0)
                m, counts = evaluate(model, loader, cfg, device)
                by_snr[str(snr)] = dict(m, counts=counts)
                hters.append(float(m['hter']))
                print(f'{family:14s} seed={seed:4d} {snr:+5.0f}dB '
                      f'MR={m["mr"]:.4f} FAR={m["far"]:.4f} HTER={m["hter"]:.4f}', flush=True)
            output['runs'][family][str(seed)] = {
                'checkpoint': str(path), 'epoch': ckpt.get('epoch'),
                'parameter_count': ckpt.get('parameter_count'),
                'attention_mode': ckpt.get('attention_mode'),
                'recurrent_type': cfg.recurrent_type,
                'recurrent_hidden': cfg.recurrent_hidden,
                'mean_hter': statistics.mean(hters),
                'by_snr': by_snr,
            }
            del model

        vals = [output['runs'][family][str(seed)]['mean_hter'] for seed in args.seeds]
        output['summary'][family] = {
            'mean_hter': statistics.mean(vals),
            'sample_sd_hter': statistics.stdev(vals) if len(vals) > 1 else None,
            'by_snr': {},
        }
        for snr in args.snrs:
            key = str(snr)
            output['summary'][family]['by_snr'][key] = {
                metric: statistics.mean(output['runs'][family][str(seed)]['by_snr'][key][metric]
                                        for seed in args.seeds)
                for metric in ('mr', 'far', 'hter')
            }

    for family in ('light_bypass', 'matched_bypass'):
        diffs = {
            str(seed): output['runs'][family][str(seed)]['mean_hter']
                       - output['runs']['full'][str(seed)]['mean_hter']
            for seed in args.seeds
        }
        output['paired_effects'][f'{family}_minus_full'] = {
            'by_seed': diffs,
            'mean': statistics.mean(diffs.values()),
        }

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2))
    print('\nTHREE-SEED SUMMARY (test characterization; not untouched holdout)' if args.split == 'test'
          else '\nTHREE-SEED SUMMARY (validation)')
    for family, s in output['summary'].items():
        sd = f'{s["sample_sd_hter"]*100:.3f}' if s['sample_sd_hter'] is not None else 'n/a'
        print(f'{family:14s}: mean HTER={s["mean_hter"]*100:.3f}% SD={sd} pp')
    for comparison, effect in output['paired_effects'].items():
        print(f'{comparison}: {effect["mean"]*100:+.3f} pp (positive favors full)')
    print(f'JSON saved: {path}')


if __name__ == '__main__':
    main()
