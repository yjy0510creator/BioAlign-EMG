# BioAlign-EMG Complete Reproducibility Package v1.0

This package implements the paper-facing computational pipeline for BioAlign-EMG: SeNic data acquisition and audit, training-only normalization, RingAug, matched TCN/SE/CBAM baselines, the final BioAlign model, the 30-participant × 3-seed main experiment, full-retraining alignment ablation, measured-angle robustness analysis, participant-level statistics, plotting, TorchScript export, and CPU latency benchmarking.

This is a cleaned reproducibility release reconstructed from the confirmed project scripts and frozen manuscript protocol. It is not a byte-for-byte archive of the original `D:\\BioSelect_EMG` directory. The SeNic dataset and trained checkpoints are intentionally not bundled.

## Frozen protocol

- SeNic h0-h29, session 0 only.
- 8 channels, 200 Hz, seven gestures.
- Training/calibration: p0 repetitions r0+r1 (14 trials per participant).
- Ideal-position test: p0-r2.
- Shift test: all p1-p10 trials.
- Window: 250 ms (50 samples), increment 50 ms (10 samples).
- Seeds: 42, 2026, 3407.
- Primary endpoint: pooled all-shift trial-level Macro-F1.
- Final BioAlign objective: gesture cross-entropy only.

See `README_CN.md` for the step-by-step Windows workflow and `docs/EXPERIMENT_PROTOCOL.md` for the formal protocol.
