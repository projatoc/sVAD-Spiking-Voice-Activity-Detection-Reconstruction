# sVAD — Spiking Voice Activity Detection

A PyTorch reconstruction of **sVAD: A Robust, Low-Power, and Light-Weight Voice Activity Detection with Spiking Neural Networks** (Qu Yang *et al.*, ICASSP 2024).

The model performs **frame-level speech/non-speech detection** using a compact spiking neural network. This repository includes training and evaluation code, pretrained checkpoints, and experiments comparing spiking attention with smaller and parameter-matched alternatives.

> **Independent reconstruction:** This is not the authors' official implementation. Results below are from a Google Speech Commands pilot with synthetic Gaussian noise, **not** the paper's QUT-NOISE-TIMIT benchmark. Hardware power consumption was not measured.

## Model overview

```text
16-kHz audio
    ↓ 30-ms frames, 15-ms hop
SincNet-style filters + spiking convolution
    ↓ 20 encoded features
Spiking attention mask (optional)
    ↓
Spiking recurrent classifier
    ↓
Speech / non-speech (per frame)
```

The full model has **4,313 parameters** and a 32-neuron diagonal recurrent layer. We also evaluate two models without learned attention:

| Model | Recurrent layer | Parameters |
| --- | --- | ---: |
| Full attention | Diagonal, 32 neurons | 4,313 |
| Lightweight bypass | Diagonal, 32 neurons | 2,030 |
| Parameter-matched bypass | Dense, 45 neurons | 4,322 |

## Quick start

### 1. Install dependencies

Use Python with a compatible PyTorch build (CPU, CUDA, or Apple Silicon MPS):

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Main dependencies: `torch`, `numpy`, `scipy`, and `soundfile`. Exact versions may depend on your operating system and accelerator.

### 2. Prepare the audio data

Evaluation expects a **Google Speech Commands v0.02** pilot dataset and a manifest at:

```text
real_data/speech_commands_pilot/manifest.jsonl
```

The project uses 16-kHz recordings, 1.5-second segments, and energy-derived **estimated speech intervals** (not manually annotated ground truth). Manifest records use fields such as:

```text
clean_path, split, source_id, speech_intervals,
annotation_type, original_path
```

The audio files must exist locally and the paths in the manifest must resolve correctly. Download Speech Commands separately and use the same pilot split/manifest to reproduce the reported numbers. **The manifest is not interchangeable with the original full Speech Commands dataset**, and the commands below will not run without the prepared pilot data. If the repository does not include a data-preparation script and the fixed manifest, the reported dataset split cannot yet be recreated from scratch.

### 3. Run evaluation with pretrained models

Place the nine `.pt` checkpoint files in `runs/` with their original filenames (see layout below), then run:

```bash
python stage8_evaluate.py \
  --manifest real_data/speech_commands_pilot/manifest.jsonl \
  --split test \
  --snrs -10 -5 0 5 10 15 \
  --output runs/stage8_final_evaluation.json
```

This evaluates all three architectures at six SNRs for seeds `42`, `123`, and `2026`, reporting miss rate (MR), false-alarm rate (FAR), and half total error rate (HTER). Evaluation adds Gaussian noise at the specified SNRs. **This is a dataset evaluation command**, not a microphone or single-file inference interface.

For available options:

```bash
python stage8_evaluate.py --help
```

## Train a model

The training script uses 60 epochs, 1.5-second segments, six noise levels, and classification cross-entropy for the final architectural comparisons.

**Full attention:**

```bash
python stage6_train.py \
  --manifest real_data/speech_commands_pilot/manifest.jsonl \
  --attention-mode full \
  --epochs 60 --batch-size 8 --segment-s 1.5 \
  --snrs -10 -5 0 5 10 15 \
  --val-snrs -10 -5 0 5 10 15 \
  --seed 42 --output runs/stage6_full_seed42.pt
```

**Lightweight bypass:** Replace `--attention-mode full` with `--attention-mode bypass` and set the output to `runs/stage6_bypass_seed42.pt`.

**Parameter-matched dense bypass** (requires the version of `stage6_train.py` supporting `--recurrent-type` and `--recurrent-hidden`):

```bash
python stage6_train.py \
  --manifest real_data/speech_commands_pilot/manifest.jsonl \
  --attention-mode bypass \
  --recurrent-type dense --recurrent-hidden 45 \
  --epochs 60 --batch-size 8 --segment-s 1.5 \
  --snrs -10 -5 0 5 10 15 \
  --val-snrs -10 -5 0 5 10 15 \
  --seed 42 --output runs/stage7_matched_bypass_seed42.pt
```

Repeat with seeds `123` and `2026` for the three-seed comparison. Checkpoint files contain model weights and configuration metadata; keep the original filenames for automatic evaluation.

## Results

Mean HTER over **three seeds and six SNRs** on the Speech Commands pilot test split (lower is better):

| Architecture | Mean HTER | SD across seeds |
| --- | ---: | ---: |
| **Full attention** | **13.915%** | 0.458 percentage points |
| Lightweight bypass | 14.316% | 0.993 percentage points |
| Parameter-matched bypass | 14.495% | 0.598 percentage points |

| SNR | Full attention | Lightweight bypass | Matched bypass |
| --- | ---: | ---: | ---: |
| −10 dB | 20.06% | 20.84% | **18.90%** |
| −5 dB | **16.12%** | 16.64% | 16.36% |
| 0 dB | **13.35%** | 13.97% | 14.61% |
| +5 dB | **11.65%** | 12.03% | 12.81% |
| +10 dB | **11.01%** | 11.05% | 12.06% |
| +15 dB | **11.29%** | 11.37% | 12.23% |

`HTER = (MR + FAR) / 2`, where MR is the speech-frame miss rate and FAR is the non-speech-frame false-alarm rate.

The full-attention model has the lowest *overall* average HTER, while the matched bypass performs best at −10 dB. Differences are modest and based on three seeds. The test split had been inspected during development, so these figures are **characterization results, not performance on an untouched holdout**.

## Repository layout

Keep source files at the project root unless you also update their imports. The evaluation script expects the checkpoint filenames shown here under `runs/`:

```text
.
├── README.md
├── requirements.txt
├── model.py
├── data.py
├── metrics.py                     # If imported by your training/evaluation code
├── stage5_train.py               # Optional: attention-reconstruction experiments
├── stage5_diagnose.py            # Optional: spike/mask diagnostics
├── stage6_train.py               # Full/bypass training (with dense/hidden options)
├── stage8_evaluate.py            # Three-architecture evaluation
├── real_data/
│   └── speech_commands_pilot/
│       └── manifest.jsonl
├── runs/
│   ├── stage6_full_seed42.pt
│   ├── stage6_full_seed123.pt
│   ├── stage6_full_seed2026.pt
│   ├── stage6_bypass_seed42.pt
│   ├── stage6_bypass_seed123.pt
│   ├── stage6_bypass_seed2026.pt
│   ├── stage7_matched_bypass_seed42.pt
│   ├── stage7_matched_bypass_seed123.pt
│   ├── stage7_matched_bypass_seed2026.pt
│   └── stage8_final_evaluation.json
└── docs/                         # Optional experiment reports
```

If checkpoints are stored in a different folder, pass `--runs PATH` to `stage8_evaluate.py`. The `runs/` directory should contain only files you intend to share; do not commit downloaded audio or virtual environments.

## Notes and limitations

- The encoder and attention architecture reconstructs the published concept; some paper-specific implementation details remain uncertain.
- Speech/non-speech frame labels were estimated from signal energy rather than hand-annotated.
- Noise was synthesized using additive Gaussian noise instead of the original benchmark's recorded noise conditions.
- Model size alone does **not** establish energy use, inference latency, or neuromorphic hardware efficiency.
- The capacity-matched bypass changes recurrent connectivity and hidden width as well as removing attention; it is not a perfectly isolated attention ablation.
- A clean-install test from this packaged repository has **not yet been independently verified**. Check the dataset manifest paths, training-script version, and checkpoint metadata before reporting exact reproduction.

## Reference

Qu Yang *et al.*, **“sVAD: A Robust, Low-Power, and Light-Weight Voice Activity Detection with Spiking Neural Networks,”** ICASSP 2024.

This repository is an independent educational/research reconstruction and is not affiliated with the original authors.
