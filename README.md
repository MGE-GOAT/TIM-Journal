# On-Device Keyword-Spotting: Confusable-Speech False Alarms and a Measured-Microphone Domain-Shift Study

Code, trained model, and the measured microphone transfer function accompanying our
IEEE Sensors Journal manuscript on a deployed on-device wake-word (keyword-spotting)
sensor. The work is **empirical/deployment-focused**: it reuses established building
blocks (depthwise-separable convolutions, squeeze-and-excitation, coordinate attention,
a streaming PCEN front-end, INT8 quantization-aware training) and contributes (1) a
per-source attribution of false alarms to confusable speech vs. ambient noise, (2) a
label-formulation / class-imbalance study, and (3) a **digital-microphone-twin** method
that turns a one-time measured transfer function of the deployment MEMS microphone into
a training-time augmentation to close the microphone-domain gap.

## What's here
- `src/` — training, augmentation, and evaluation code:
  - `common.py` — feature front-end (32-band mel + streaming PCEN) and the Noban-V17 model.
  - `build_cache.py` — feature extraction / caching; `train_ablation.py` — the 3-class vs binary vs FRR label-scheme study.
  - `build_twin_filter.py` — builds the digital-microphone-twin FIR from a measured log-sweep.
  - `build_twin_cache.py` — synthesizes an INMP441-domain corpus by filtering conventional audio through the twin.
  - `train_job.py`, `dispatch_ladder.py` — the A0–A5 ablation ladder, runnable across multiple GPUs.
  - `analyze_domainshift.py`, `fan_analysis.py`, `spec_compare.py`, `gen_sweep.py` — microphone-characterization utilities.
  - `stats_review.py`, `eval_far.py`, `macs.py` — statistics (Clopper–Pearson, Fisher, McNemar), FAR@FRR, and MAC counting.
- `models/3class_qat_final_int8.tflite` — the deployed INT8 model (≈182 KB, 75,075 parameters).
- `measurement/twin_fir.npy` — the measured conventional→INMP441 transfer-function FIR (16 kHz).
- `measurement/reference.wav` — the log-sine sweep stimulus (50 Hz–7.5 kHz) used to measure the transfer function.

## Data availability
The **non-keyword** classes derive from publicly available corpora; the **keyword
recordings** are identifiable human-voice data collected under consent that does not
permit public redistribution, and are **withheld to protect speaker privacy**. The
released pipeline reproduces the full method and can be applied to any keyword-spotting
corpus (e.g., the public Google Speech Commands dataset), so the approach is reproducible
without the private recordings.

## Reproducing the method on your own data
1. Point `common.py` (`WAKE_ROOT` / `OTHER_ROOT` / `NOISE_ROOT`) at your keyword / non-keyword-speech / noise folders.
2. `python build_cache.py` to extract features.
3. `python train_ablation.py --scheme 3class --qat` for the model + label-scheme study.
4. To reproduce the microphone-twin study: measure your deployment microphone with `measurement/reference.wav`, build the FIR with `build_twin_filter.py`, synthesize the twin corpus with `build_twin_cache.py`, and run the ladder with `dispatch_ladder.py`.

Requires TensorFlow 2.13, NumPy, SciPy, scikit-learn, and librosa.

## Citation
Please cite the accompanying manuscript (details to be added on publication).

## License
MIT — see `LICENSE`.
