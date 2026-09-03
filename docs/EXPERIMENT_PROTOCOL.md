# Frozen experiment protocol

## Cohort and data

The primary cohort is SeNic h0-h29. Only session 0 is used. Each selected session contains 11 electrode positions (p0-p10), seven retained gestures, and three repetitions, for 231 trials per participant.

## Split

- Calibration/training: p0-r0 and p0-r1.
- Ideal-position evaluation: p0-r2.
- Electrode-shift evaluation: all repetitions from p1-p10.
- No real shifted trial enters supervised training.
- Normalization mean and standard deviation are estimated only from p0-r0/r1 windows for the same participant.

## Signal representation

- Sampling frequency: 200 Hz.
- Window duration: 250 ms = 50 samples.
- Window increment: 50 ms = 10 samples.
- Adjacent windows overlap by 200 ms.
- The supplied dataset signal is windowed directly; no test-position-specific processing is permitted.

## Models

- TCN_plain: lightweight depthwise-separable TCN, no RingAug.
- TCN_RingAug: same TCN, matched augmentation.
- SE_RingAug: SE-TCN, matched augmentation.
- CBAM_RingAug: CBAM-TCN, matched augmentation.
- BioAlign_RingAug: per-channel multiscale encoder, latent eight-way cyclic correspondence, differentiable soft inverse-roll canonicalization, lightweight TCN classifier.

## Augmentation

For RingAug models, each training window receives:

1. a random cyclic channel roll k in {0,...,7};
2. independent channel gain jitter around 1.0, clipped to 0.65-1.35;
3. with probability 0.25, attenuation of one random channel by a factor in 0.15-0.60.

## Optimization

- AdamW, learning rate 1e-3.
- Weight decay 1e-4.
- Batch size 256.
- 20 epochs.
- Cosine annealing.
- Gradient norm clipping at 5.0.
- Seeds 42, 2026, 3407.
- Final BioAlign loss: gesture cross-entropy only.

## Evaluation and statistics

Window posteriors are averaged within each complete trial. The class with the highest mean posterior is the trial prediction. The primary endpoint is pooled p1-p10 trial-level Macro-F1.

Metrics are averaged across the three seeds within participant before inferential testing. The participant is the statistical unit. The frozen manuscript uses one-sided paired Wilcoxon signed-rank tests (BioAlign > comparator) with Holm correction across the four primary baseline comparisons.

## Measured-angle robustness

Physical electrode coordinates are read from `Angle_<subject>_0.xlsx`. The median absolute circular displacement of CH1-CH8 relative to p0 represents the position's measured shift. Planned rotations p0-p8 form the nAUPC analysis; random displacement positions p9-p10 are excluded from the monotonic rotation summary.
