# Output files

## Main experiment (`14_final_main_*`)

- `raw_metrics.csv`: one row per participant/model/seed/position plus aggregate position -1.
- `subject_summary.csv`: seed-averaged participant metrics.
- `model_summary.csv`: cohort mean, SD and 95% CI.
- `paired_stats.csv`: participant-level Wilcoxon tests and Holm adjustment.
- `seed_stability.csv`: model performance for each random seed.
- `report.md`: text summary.
- figures: shift boxplot, position curve, and paired scatter.

## Evidence pack (`15_*`)

- full-retraining alignment ablation outputs;
- physical-angle position table;
- participant nAUPC table;
- paired angle statistics;
- angle-performance and nAUPC figures.

## Deployment (`16_*`)

- CPU latency CSV;
- benchmark report;
- TorchScript files under `deploy/14_final/`.
