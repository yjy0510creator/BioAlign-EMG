# Changelog V2.1

This version fixes the critical issues identified after V2.0:

- Adds directed SeNic angle parsing from `Angle_h*_0.xlsx`.
- Adds standardized NPZ builder with `angle_deg`.
- Separates augmentation, shift-supervision, and feature-consistency switches in training.
- Prevents negative controls from receiving unintended shift/consistency losses.
- Adds clean model variants for augmentation-only, shift-auxiliary-only, classification-only alignment, and full alignment.
- Evaluates classical baselines with trial-level probability aggregation.
- Adds xcorr, oracle integer, oracle continuous, wrong-direction integer, and wrong-direction continuous correction baselines.
- Stores compact pre/post feature descriptors during evaluation for pattern-preservation analysis.
- Uses trial-group split for latent disentanglement diagnostics.
- Adds aggregate results runner with subject-level statistics, two-sided Wilcoxon, Holm correction, and bootstrap CI.
- Adds h0 quick and full 30-subject batch scripts.

Important: V2.1 is a code-corrected experiment package. It does not contain final real-data results. Run h0 quick validation first, then h0-h29 locked training.
