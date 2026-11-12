# sVAD — Spiking Voice Activity Detection Reconstruction

A PyTorch research reconstruction of **sVAD: A Robust, Low-Power, and Light-Weight Voice Activity Detection with Spiking Neural Networks** (Qu Yang *et al.*, ICASSP 2024).

This project implements and evaluates a compact spiking neural network (SNN) for frame-level speech/non-speech detection. It explores the effects of recurrent connectivity, spiking attention, and an auxiliary clean-feature reconstruction objective.

> **Reproduction status:** This is an independent **software reconstruction**, not an official implementation or an exact reproduction of the paper's published benchmark or power figures. Several implementation details were not fully specified in the information available during this project.

## Highlights

- Reconstructed full-attention model: **4,313 parameters**.
- Audio pipeline: SincNet-style filtering → spiking convolutional encoder → spiking attention → recurrent spiking classifier → two-class readout.
- 16 kHz recordings, 30 ms frames, 15 ms hop, 20 encoded features, and 32 recurrent hidden neurons in the baseline full model.
- Training and evaluation with a pilot subset of **Google Speech Commands v0.02**, synthetic additive Gaussian noise, and six SNR conditions: **−10, −5, 0, +5, +10, +15 dB**.
- Controlled ablations across **three random seeds (42, 123, 2026)**.
- Final evaluation of full attention, lightweight attention bypass, and near-parameter-matched dense recurrent bypass.

## Final results

Three-seed **test-set characterization** (average HTER over six SNR conditions; lower is better):

| Architecture | Parameters | Mean HTER | SD across seeds |
| --- | ---: | ---: | ---: |
| Full spiking attention | 4,313 | **13.915%** | **0.458 percentage points** |
| Lightweight attention bypass | 2,030 | 14.316% | 0.993 percentage points |
| Parameter-matched dense bypass | 4,322 (expected) | 14.495% | 0.598 percentage points |

The full model improves average HTER relative to the lightweight bypass by **0.401 percentage points** and relative to the matched bypass by **0.580 percentage points**. However, differences are modest and based on three seeds. The matched bypass performs better at the most severe tested noise level (−10 dB), and the test set was examined during development; these results **must not be presented as confirmation on an untouched holdout**.

### Per-SNR HTER (three-seed test averages)

| SNR | Full attention | Lightweight bypass | Matched dense bypass |
| --- | ---: | ---: | ---: |
| −10 dB | 20.06% | 20.84% | **18.90%** |
| −5 dB | **16.12%** | 16.64% | 16.36% |
| 0 dB | **13.35%** | 13.97% | 14.61% |
| +5 dB | **11.65%** | 12.03% | 12.81% |
| +10 dB | **11.01%** | 11.05% | 12.06% |
| +15 dB | **11.29%** | 11.37% | 12.23% |

**HTER** (Half Total Error Rate) is computed as:

\[
\mathrm{HTER} = \tfrac12(\mathrm{MR}+\mathrm{FAR}),
\]

where **MR** is the proportion of speech frames classified as non-speech, and **FAR** is the proportion of non-speech frames classified as speech.

## Suggested submission layout

The final folder can be arranged as follows. **The scripts listed below were developed in the project; copy the versions that actually work together on your machine.** They are not automatically included with this README.

```text
sVAD_Final/
├── README.md
├── requirements.txt                    # Generated from the working environment
├── model.py                            # Network and Config
├── data.py                             # Dataset and frame labeling
├── metrics.py                          # MR, FAR, HTER (if used by local scripts)
├── stage5_train.py                     # Reconstruction-loss experiments
├── stage5_diagnose.py                  # Feature / mask / spike diagnostics
├── stage6_train.py                     # Full and bypass training; updated for Stage 7
├── stage6_compare.py                   # Stage 6 comparison (optional)
├── stage7_compare.py                   # Capacity-matched comparison (optional)
├── stage8_evaluate.py                  # Final nine-checkpoint evaluation
├── real_data/
│   └── speech_commands_pilot/
│       └── manifest.jsonl              # Include only if paths/data access are reproducible
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
├── results/
│   └── summary_tables.csv              # Optional summarized table
└── docs/
    └── ...                             # Stage reports and final research report
```

> The evaluation scripts used `runs/`-relative checkpoint paths during development. If you move checkpoints to a separate `checkpoints/` directory, update those paths in the evaluation script first. Keeping the original `runs/` folder is the simplest reproducible option.

## Requirements and setup

- Python and a compatible PyTorch installation.
- A supported execution device: CPU, Apple Silicon MPS, or CUDA (depending on the selected PyTorch build).
- The project's Python dependencies (at minimum, PyTorch and the packages imported by the submitted scripts). The development environment was on macOS with Apple Silicon and used a shared virtual environment from an earlier project folder.

Create an isolated environment in the submission folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Before submission:** Generate and test `requirements.txt` from the working project environment. Avoid assuming the development environment's exact package versions are portable across operating systems and hardware. For example, `python -m pip freeze > requirements.txt` records all currently installed packages, though a curated direct-dependency file is preferable.

The project has **not yet been verified here in a fresh clean environment**. That check should be completed before final submission.

## Dataset and preprocessing

The pilot uses Google Speech Commands v0.02 audio recordings, sampled at 16 kHz. Recordings were divided into approximately **30 ms frames with 15 ms hop**. The project builds frame-level speech/non-speech targets using **automatically estimated speech intervals based on signal energy**, not manually verified word boundaries.

The manifest developed for the project used fields such as:

```text
clean_path, split, source_id, speech_intervals,
annotation_type, original_path
```

For robustness tests, noisy versions are generated by adding synthetic Gaussian noise to the original recordings at selected SNRs. The **clean reference** used for reconstruction-loss experiments is the original dataset audio, which may itself contain incidental recording noise. The dataset is **not supplied by this README**; obtain and prepare it separately and ensure paths in the manifest resolve correctly.

> **Reproducibility warning:** A manifest alone may not be sufficient if it contains paths relative to the original project layout. Include the dataset-preparation code and a documented fixed split, or adjust paths and check the manifest before sharing. Follow dataset licensing and redistribution conditions.

## Model variants

| Variant | Attention | Recurrence | Hidden neurons | Parameters |
| --- | --- | --- | ---: | ---: |
| Full model | Learned spiking mask | Diagonal | 32 | 4,313 |
| Lightweight bypass | Identity mask (no attention) | Diagonal | 32 | 2,030 |
| Capacity-matched bypass | Identity mask (no attention) | Dense | 45 | 4,322 (expected) |

In the full model, the encoded feature representation \(Y\) is masked by a learned binary spiking attention mask \(M\): \(\hat S = Y \odot M\). In the bypass models, the mask is an identity gate: \(\hat S = Y\).

Parameter matching changes the recurrent architecture and hidden size, so the full-versus-matched comparison is **not a perfectly isolated test of attention**.

## Training and reproduction

All Stage 6/7 final architectures were trained for 60 epochs with batch size 8, 1.5 s audio segments, SNR choices −10/−5/0/+5/+10/+15 dB, and seeds 42, 123, 2026. Stage 6/7 use \(\lambda_{\mathrm{MSE}}=0\), i.e. CE-only classification loss.

The following examples assume the scripts are kept at the project root, as they were during development.

### Full attention

```bash
python stage6_train.py \
  --manifest real_data/speech_commands_pilot/manifest.jsonl \
  --attention-mode full \
  --epochs 60 --batch-size 8 --segment-s 1.5 \
  --snrs -10 -5 0 5 10 15 \
  --val-snrs -10 -5 0 5 10 15 \
  --seed 42 \
  --output runs/stage6_full_seed42.pt
```

### Lightweight bypass

```bash
python stage6_train.py \
  --manifest real_data/speech_commands_pilot/manifest.jsonl \
  --attention-mode bypass \
  --epochs 60 --batch-size 8 --segment-s 1.5 \
  --snrs -10 -5 0 5 10 15 \
  --val-snrs -10 -5 0 5 10 15 \
  --seed 42 \
  --output runs/stage6_bypass_seed42.pt
```

### Parameter-matched dense bypass

This requires the **Stage 7-updated** `stage6_train.py` with CLI options `--recurrent-type` and `--recurrent-hidden`, and a model builder that uses them.

```bash
python stage6_train.py \
  --manifest real_data/speech_commands_pilot/manifest.jsonl \
  --attention-mode bypass \
  --recurrent-type dense --recurrent-hidden 45 \
  --epochs 60 --batch-size 8 --segment-s 1.5 \
  --snrs -10 -5 0 5 10 15 \
  --val-snrs -10 -5 0 5 10 15 \
  --seed 42 \
  --output runs/stage7_matched_bypass_seed42.pt
```

Repeat these commands with `--seed 123` and `--seed 2026` and the correspondingly named output checkpoints. Each model should initialize from scratch, rather than resume from a previous run.

### Final evaluation

With all nine checkpoints under `runs/`:

```bash
python stage8_evaluate.py \
  --manifest real_data/speech_commands_pilot/manifest.jsonl \
  --split test \
  --snrs -10 -5 0 5 10 15 \
  --output runs/stage8_final_evaluation.json
```

The evaluation script should reconstruct each model using the saved `config`, `attention_mode`, and `parameter_count` metadata in the checkpoint. Confirm that the exact paths and metadata handling match the versions of your submitted code.

## Summary of experimental stages

| Stage | Main objective | Outcome |
| --- | --- | --- |
| 1 | Implement the compact spiking VAD | Functional 4,313-parameter full model |
| 2 | Validate forward/backward pass and learning with synthetic data | Synthetic fixed-batch overfitting and training tests passed |
| 3 | Move to real recordings and multi-SNR testing | Speech Commands pilot training and noisy-speech evaluation established |
| 4 | Compare diagonal and dense recurrent connections | Performance depended on parameter allocation, seed, and SNR |
| 5 | Study clean-reference MSE reconstruction loss | No reproducible three-seed accuracy gain; detached targets showed lower observed variability |
| 6 | Ablate learned attention | Full model modestly outperformed smaller bypass on average |
| 7 | Compare attention against similar parameter budget | Full and matched dense bypass had near-identical validation means |
| 8 | Characterize all final models on test split | Full model had best average HTER; matched bypass won at −10 dB |
| 9 | Consolidate reporting and submission | Packaging and documentation |

### Stage 5 auxiliary objective

The reconstruction-loss experiment used:

\[
L=L_{\mathrm{CE}}+\lambda L_{\mathrm{MSE}},\qquad
L_{\mathrm{MSE}}=\frac{1}{N}\sum_i(\hat S_i-S_i)^2,
\]

where \(S\) is the encoder representation of the original recording and \(\hat S\) is the attended noisy representation. Both *joint* gradients and *detached clean-reference* targets were tested. Across seeds 42, 123, and 2026, no robust mean-HTER advantage over \(\lambda=0\) was established.

## Comparison with the original publication

The original paper reports HTER of approximately **4.0% / 11.9% / 19.1%** for its low-/medium-/high-noise evaluation groups on **QUT-NOISE-TIMIT**, and a low-power estimate associated with specialized neuromorphic hardware.

**Those figures cannot be directly compared to this project's HTER values**, because this reconstruction uses a different audio dataset, synthetic Gaussian noise, approximate speech-boundary annotations, and a different evaluation protocol. This project also **does not measure neuromorphic hardware power**. It validates a software reconstruction and its relative ablation behavior, not the original paper's exact accuracy or power claims.

## Limitations

1. Missing or uncertain architectural details in the reconstruction, including exact spiking-time behavior and some attention/encoder design choices.
2. Dataset mismatch: Speech Commands pilot instead of QUT-NOISE-TIMIT.
3. Synthetic Gaussian noise instead of the paper's recorded environmental noise.
4. Energy-derived pseudo-labels instead of human-verified frame boundaries.
5. Small pilot and only three random seeds per final architecture.
6. Prior inspection of the test split during development; the final scores are **test characterization**, not a pristine holdout result.
7. Parameter count is not a power, latency, or energy measurement.
8. Stage 7 changes both parameter allocation and recurrent structure; causal claims about attention remain limited.

## Submission checklist

- [ ] Include compatible source code and a tested `requirements.txt`.
- [ ] Keep all nine final checkpoints with their original names.
- [ ] Include `runs/stage8_final_evaluation.json` and a concise CSV results table.
- [ ] Supply the manifest and sufficient data-preparation instructions to reproduce dataset splits.
- [ ] Add the completed Stage 1–8 research notes and consolidated Stage 9 report.
- [ ] Run evaluation from a newly created environment before submitting.
- [ ] Exclude `.venv/`, caches, unnecessary intermediate checkpoints, and unlicensed dataset copies.

## Citation

If you use this project in an academic submission, cite the original sVAD paper and clearly identify this repository as an independent reconstruction. Add the full bibliographic entry/DOI from the official paper in the final report or a `CITATION.cff` file.
