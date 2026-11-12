"""Audio/annotation loader for paired clean/noisy sVAD experiments.

Manifest is JSONL; each line must include:
 {"clean_path":"/absolute/or/relative/to/manifest.wav", "split":"train",
  "speech_intervals":[[0.1,0.7],[1.0,1.35]], "source_id":"speaker-01",
  "noise_path":"optional/noise.wav"}

Use annotation-derived intervals, NOT assumed full-clip speech unless known.
Labels mark a frame as speech if the speech-time intersection / frame duration
is at least label_overlap_threshold (default 0.5).
"""
import json
import math
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from torch.utils.data import Dataset


def load_mono(path, sample_rate):
    wav, sr = sf.read(str(path), dtype='float32', always_2d=True)
    wave = wav.mean(axis=1)
    if sr != sample_rate:
        divisor = math.gcd(int(sr), int(sample_rate))
        wave = resample_poly(wave, sample_rate // divisor, int(sr) // divisor).astype('float32')
    return np.asarray(wave, dtype='float32')


def frame_labels(intervals, n_samples, sample_rate, frame_samples, hop_samples, threshold=0.5):
    if n_samples < frame_samples:
        raise ValueError('Segment shorter than analysis frame')
    starts = np.arange(0, n_samples - frame_samples + 1, hop_samples, dtype='int64')
    coverage = np.zeros(len(starts), dtype='float64')
    for start_s, end_s in intervals:
        lo = max(0.0, float(start_s) * sample_rate)
        hi = min(float(n_samples), float(end_s) * sample_rate)
        if hi <= lo:
            continue
        coverage += np.maximum(0.0, np.minimum(starts + frame_samples, hi) - np.maximum(starts, lo))
    # Multiple intervals must not double count overlapping annotation regions.
    # We reject overlap in the loader rather than silently inflate coverage.
    return torch.as_tensor((coverage / frame_samples >= threshold).astype('int64'))


def mix_at_snr(clean, noise, snr_db, eps=1e-8, silence_reference_rms=0.1):
    """Scale additive noise to desired global active-signal power SNR."""
    clean = clean.astype('float32', copy=False)
    noise = noise.astype('float32', copy=False)
    clean_power = float(np.mean(clean ** 2))
    noise_power = float(np.mean(noise ** 2))
    if noise_power < eps:
        return clean.copy()
    # SNR is undefined for silence-only clips. Still add background noise,
    # using a documented reference RMS instead of returning perfect silence.
    reference_power = clean_power if clean_power >= eps else silence_reference_rms**2
    desired_noise_power = reference_power / 10 ** (float(snr_db) / 10)
    scale = np.sqrt(desired_noise_power / max(noise_power, eps))
    return (clean + scale * noise).astype('float32')


class PairedAudioDataset(Dataset):
    def __init__(self, manifest, split, cfg, segment_s=1.5, snrs=(-10, -5, 0, 5, 10, 15),
                 label_overlap_threshold=0.5, random_crop=False, seed=42):
        self.manifest = Path(manifest).resolve()
        self.cfg = cfg
        self.split = split
        self.segment_samples = max(cfg.frame_samples, int(round(segment_s * cfg.sample_rate)))
        self.snrs = tuple(float(s) for s in snrs)
        self.label_overlap_threshold = label_overlap_threshold
        self.random_crop = random_crop
        self.seed = seed
        rows = [json.loads(line) for line in self.manifest.read_text().splitlines() if line.strip()]
        self.rows = [row for row in rows if row['split'] == split]
        if not self.rows:
            raise ValueError(f'No records with split={split!r} in {manifest}')
        for row in self.rows:
            if 'speech_intervals' not in row:
                raise ValueError('Every entry needs speech_intervals, including [] for silence')
            ranges = sorted(row['speech_intervals'])
            if any(float(b) <= float(a) for a,b in ranges) or any(float(ranges[i][1]) > float(ranges[i+1][0]) for i in range(len(ranges)-1)):
                raise ValueError('Speech intervals must be non-empty and nonoverlapping')
        # Prevent same source being assigned to both splits. For real projects,
        # source_id should be a speaker or original-recording identity.
        group_to_split = {}
        for row in rows:
            group = row.get('source_id')
            if group is not None:
                if group in group_to_split and group_to_split[group] != row['split']:
                    raise ValueError(f'Leakage: source_id {group!r} appears in multiple splits')
                group_to_split[group] = row['split']

    def _resolve(self, p):
        p = Path(p)
        return p if p.is_absolute() else self.manifest.parent / p

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        # Random crop in training, deterministic crop for validation/test.
        rng = np.random.default_rng(None if self.random_crop else self.seed + index)
        clean = load_mono(self._resolve(row['clean_path']), self.cfg.sample_rate)
        if len(clean) < self.segment_samples:
            clean = np.pad(clean, (0, self.segment_samples - len(clean)))
        max_start = len(clean) - self.segment_samples
        start = int(rng.integers(0, max_start + 1)) if self.random_crop and max_start else 0
        stop = start + self.segment_samples
        clip = clean[start:stop].copy()
        offset_sec = start / self.cfg.sample_rate
        intervals = [(float(a) - offset_sec, float(b) - offset_sec) for a,b in row['speech_intervals']]
        labels = frame_labels(intervals, self.segment_samples, self.cfg.sample_rate,
                              self.cfg.frame_samples, self.cfg.hop_samples,
                              self.label_overlap_threshold)
        if row.get('noise_path'):
            raw_noise = load_mono(self._resolve(row['noise_path']), self.cfg.sample_rate)
            if len(raw_noise) == 0:
                raise ValueError('Noise clip is empty')
            if len(raw_noise) < self.segment_samples:
                repeats = (self.segment_samples + len(raw_noise) - 1)//len(raw_noise)
                raw_noise = np.tile(raw_noise, repeats)
            max_noise_start = len(raw_noise) - self.segment_samples
            noise_start = int(rng.integers(0,max_noise_start+1)) if self.random_crop and max_noise_start else 0
            noise = raw_noise[noise_start:noise_start+self.segment_samples]
        else:
            # Gaussian noise is a pipeline fallback, not a QUT-NOISE-TIMIT reproduction.
            noise = rng.normal(0, 1, self.segment_samples).astype('float32')
        # Fixed SNR for val/test when using a single value; randomly sample train set.
        snr_db = float(rng.choice(self.snrs))
        noisy = mix_at_snr(clip, noise, snr_db)
        return {'clean': torch.from_numpy(clip), 'noisy': torch.from_numpy(noisy),
                'labels': labels, 'snr_db': torch.tensor(snr_db, dtype=torch.float32)}
