from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .statistics import holm_adjust, paired_wilcoxon


def circular_deg_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a - b + 180.0) % 360.0) - 180.0


def subject_angle_table(subject: str, subjects_root: Path) -> pd.DataFrame:
    subject_dir = subjects_root / subject
    candidates = sorted(subject_dir.rglob(f"Angle_{subject}_0.xlsx"))
    if not candidates:
        return pd.DataFrame()
    table = pd.read_excel(candidates[0])
    channels = [f"CH{i}" for i in range(1, 9)]
    if "ID" not in table.columns or not all(channel in table.columns for channel in channels):
        return pd.DataFrame()
    table = table.loc[table["ID"].between(0, 10)].copy()
    reference_rows = table.loc[table["ID"] == 0, channels]
    if reference_rows.empty:
        return pd.DataFrame()
    reference = reference_rows.iloc[0].to_numpy(float)
    rows = []
    for _, row in table.iterrows():
        current = row[channels].to_numpy(float)
        delta = circular_deg_diff(current, reference)
        rows.append(
            {
                "subject": subject,
                "position": int(row["ID"]),
                "median_abs_shift_deg": float(np.median(np.abs(delta))),
            }
        )
    return pd.DataFrame(rows)


def normalized_angle_auc(x: np.ndarray, y: np.ndarray) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if frame.empty:
        return float("nan")
    frame = frame.groupby("x", as_index=False)["y"].mean().sort_values("x")
    x_values = frame["x"].to_numpy(float)
    y_values = frame["y"].to_numpy(float)
    span = float(x_values.max() - x_values.min())
    if len(x_values) < 2 or span <= 0:
        return float("nan")
    trap = getattr(np, "trapezoid", np.trapz)
    return float(trap(y_values, x_values) / span)


def run_angle_analysis(
    *,
    main_raw_path: Path,
    subjects_root: Path,
    result_dir: Path,
    figure_dir: Path,
    reference_model: str = "BioAlign_RingAug",
) -> dict[str, Path]:
    if not main_raw_path.exists():
        raise FileNotFoundError(
            f"Main raw metrics not found: {main_raw_path}. Run the main experiment first."
        )
    raw = pd.read_csv(main_raw_path, encoding="utf-8-sig")
    subjects = sorted(raw["subject"].astype(str).unique())
    angle_frames = [subject_angle_table(subject, subjects_root) for subject in subjects]
    angle_frames = [frame for frame in angle_frames if not frame.empty]
    if not angle_frames:
        raise RuntimeError("No Angle_<subject>_0.xlsx files were found.")
    angles = pd.concat(angle_frames, ignore_index=True)

    position = raw.loc[raw["position"].between(0, 8)].copy()
    position = (
        position.groupby(["subject", "model", "position"], as_index=False)[
            "trial_macro_f1"
        ]
        .mean()
        .merge(angles, on=["subject", "position"], how="inner")
    )

    auc_rows = []
    for (subject, model), group in position.groupby(["subject", "model"]):
        auc_rows.append(
            {
                "subject": subject,
                "model": model,
                "nAUPC": normalized_angle_auc(
                    group["median_abs_shift_deg"].to_numpy(float),
                    group["trial_macro_f1"].to_numpy(float),
                ),
            }
        )
    auc = pd.DataFrame(auc_rows)

    reference = auc.loc[auc["model"] == reference_model].set_index("subject")
    stat_rows = []
    for model in sorted(set(auc["model"]) - {reference_model}):
        comparator = auc.loc[auc["model"] == model].set_index("subject")
        common = sorted(set(reference.index) & set(comparator.index))
        x = reference.loc[common, "nAUPC"].to_numpy(float)
        y = comparator.loc[common, "nAUPC"].to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        statistic, p_value = paired_wilcoxon(x, y, alternative="greater")
        difference = x - y
        stat_rows.append(
            {
                "comparison": f"{reference_model} > {model}",
                "n_subjects": len(x),
                "reference_nAUPC": float(x.mean()),
                "baseline_nAUPC": float(y.mean()),
                "mean_paired_gain": float(difference.mean()),
                "reference_wins": int(np.sum(difference > 0)),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )
    stats = pd.DataFrame(stat_rows)
    if not stats.empty:
        stats["p_holm"] = holm_adjust(stats["p_value"].tolist())

    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    position_path = result_dir / "15_physical_angle_position_metrics.csv"
    auc_path = result_dir / "15_physical_angle_naupc.csv"
    stats_path = result_dir / "15_physical_angle_paired_stats.csv"
    position.to_csv(position_path, index=False, encoding="utf-8-sig")
    auc.to_csv(auc_path, index=False, encoding="utf-8-sig")
    stats.to_csv(stats_path, index=False, encoding="utf-8-sig")

    bins = np.asarray([0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180.1])
    labels = [
        "0-22.5",
        "22.5-45",
        "45-67.5",
        "67.5-90",
        "90-112.5",
        "112.5-135",
        "135-157.5",
        "157.5-180",
    ]
    plot_frame = position.copy()
    plot_frame["angle_bin"] = pd.cut(
        plot_frame["median_abs_shift_deg"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    summary = (
        plot_frame.groupby(["model", "angle_bin"], observed=False)["trial_macro_f1"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x_positions = np.arange(len(labels))
    for model in sorted(summary["model"].unique()):
        group = summary.loc[summary["model"] == model].copy()
        mapping = {str(row.angle_bin): row for row in group.itertuples(index=False)}
        y = np.asarray([
            float(mapping[label].mean) if label in mapping and mapping[label].count > 0 else np.nan
            for label in labels
        ])
        ci = np.asarray([
            1.96 * float(mapping[label].std) / np.sqrt(max(1, int(mapping[label].count)))
            if label in mapping and int(mapping[label].count) > 1 else 0.0
            for label in labels
        ])
        valid = np.isfinite(y)
        ax.plot(x_positions[valid], y[valid], marker="o", linewidth=1.8, label=model)
        ax.fill_between(x_positions[valid], y[valid] - ci[valid], y[valid] + ci[valid], alpha=0.10)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlabel("Median absolute physical electrode displacement (degrees)")
    ax.set_ylabel("Trial-level Macro-F1")
    ax.set_title("Robustness across planned physical electrode rotations (p0-p8)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    angle_figure = figure_dir / "15_physical_angle_robustness.png"
    fig.savefig(angle_figure, dpi=220, bbox_inches="tight")
    plt.close(fig)

    models = sorted(auc["model"].unique())
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(
        [auc.loc[auc["model"] == model, "nAUPC"].dropna().to_numpy(float) for model in models],
        labels=models,
        showmeans=True,
    )
    ax.set_ylabel("Normalized physical-angle performance AUC (nAUPC)")
    ax.set_title("Participant-level robustness across p0-p8")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    auc_figure = figure_dir / "15_angle_naupc_boxplot.png"
    fig.savefig(auc_figure, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "position": position_path,
        "naupc": auc_path,
        "stats": stats_path,
        "angle_figure": angle_figure,
        "naupc_figure": auc_figure,
    }
