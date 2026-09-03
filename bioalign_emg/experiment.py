from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import PRIMARY_MODELS
from .data import (
    build_subject_cache,
    normalize_from_training,
    parse_subjects,
    training_mask,
)
from .metrics import predict_probabilities, trial_level_metrics
from .models import count_parameters
from .statistics import ci95_half_width, paired_comparisons
from .training import train_model


def ensure_layout(project_root: Path) -> dict[str, Path]:
    paths = {
        "subjects": project_root / "data" / "raw" / "SeNic" / "subjects",
        "cache": project_root / "data" / "processed" / "14_final_session0_cache",
        "checkpoints": project_root / "checkpoints" / "14_final_main",
        "results": project_root / "results",
        "figures": project_root / "figures",
        "logs": project_root / "logs",
    }
    for key, path in paths.items():
        if key != "subjects":
            path.mkdir(parents=True, exist_ok=True)
    return paths


def evaluate_model(
    model,
    payload: dict[str, np.ndarray],
    X_normalized: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> dict:
    y = payload["y"]
    position = payload["position"]
    repetition = payload["repetition"]
    trial_id = payload["trial_id"]

    _, probability = predict_probabilities(
        model,
        X_normalized,
        batch_size=batch_size,
        device=device,
    )

    position_rows = []
    for pos in range(11):
        if pos == 0:
            mask = (position == 0) & (repetition == 2)
        else:
            mask = position == pos
        if not mask.any():
            continue
        metrics = trial_level_metrics(y[mask], probability[mask], trial_id[mask])
        position_rows.append(
            {
                "position": pos,
                "trial_macro_f1": metrics["macro_f1"],
                "trial_accuracy": metrics["accuracy"],
                "trial_balanced_accuracy": metrics["balanced_accuracy"],
                "n_trials": int(len(np.unique(trial_id[mask]))),
            }
        )

    ideal = (position == 0) & (repetition == 2)
    shifted = position > 0
    ideal_metrics = trial_level_metrics(
        y[ideal],
        probability[ideal],
        trial_id[ideal],
    )
    shift_metrics = trial_level_metrics(
        y[shifted],
        probability[shifted],
        trial_id[shifted],
    )
    return {
        "position_rows": position_rows,
        "ideal_macro_f1": ideal_metrics["macro_f1"],
        "shift_macro_f1": shift_metrics["macro_f1"],
        "degradation": ideal_metrics["macro_f1"] - shift_metrics["macro_f1"],
        "shift_n_trials": int(len(np.unique(trial_id[shifted]))),
    }


def summarize(
    raw: pd.DataFrame,
    *,
    models: list[str],
    reference_model: str,
    stats_alternative: str = "greater",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregate = raw.loc[raw["position"] == -1].copy()
    subject = (
        aggregate.groupby(["subject", "model"], as_index=False)
        .agg(
            ideal_macro_f1=("ideal_macro_f1", "mean"),
            shift_macro_f1=("shift_macro_f1", "mean"),
            degradation=("degradation", "mean"),
            seeds=("seed", "nunique"),
        )
    )

    rows = []
    for model, group in subject.groupby("model"):
        row = {"model": model, "n_subjects": int(group["subject"].nunique())}
        for column in ("ideal_macro_f1", "shift_macro_f1", "degradation"):
            values = group[column].to_numpy(float)
            half = ci95_half_width(values)
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{column}_ci95_half_width"] = half
            row[f"{column}_ci95_low"] = float(values.mean() - half)
            row[f"{column}_ci95_high"] = float(values.mean() + half)
        rows.append(row)
    model_summary = pd.DataFrame(rows).sort_values(
        "shift_macro_f1_mean",
        ascending=False,
    )

    comparator_models = [model for model in models if model != reference_model]
    stats = paired_comparisons(
        subject,
        reference_model=reference_model,
        comparator_models=comparator_models,
        metric="shift_macro_f1",
        alternative=stats_alternative,
    )
    return subject, model_summary, stats


def create_figures(
    subject: pd.DataFrame,
    raw: pd.DataFrame,
    figure_dir: Path,
    *,
    prefix: str,
    models: list[str],
    reference_model: str,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)

    values = [
        subject.loc[subject["model"] == model, "shift_macro_f1"].to_numpy(float)
        for model in models
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(values, labels=models, showmeans=True)
    ax.set_ylabel("All-shift trial-level Macro-F1")
    ax.set_title("Participant-level electrode-shift robustness")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / f"{prefix}_shift_macrof1_boxplot.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    position_rows = raw.loc[raw["position"].between(0, 10)].copy()
    position_rows = (
        position_rows.groupby(["subject", "model", "position"], as_index=False)[
            "trial_macro_f1"
        ]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for model in models:
        group = position_rows.loc[position_rows["model"] == model]
        summary = (
            group.groupby("position")["trial_macro_f1"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("position")
        )
        ci = 1.96 * summary["std"] / np.sqrt(summary["count"].clip(lower=1))
        ax.plot(summary["position"], summary["mean"], marker="o", linewidth=1.8, label=model)
        ax.fill_between(
            summary["position"],
            summary["mean"] - ci,
            summary["mean"] + ci,
            alpha=0.10,
        )
    ax.set_xlabel("Electrode position index (p0 = reference)")
    ax.set_ylabel("Trial-level Macro-F1")
    ax.set_title("Position-wise robustness (participant mean ± 95% CI)")
    ax.set_xticks(range(11))
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / f"{prefix}_position_curve.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    reference = subject.loc[subject["model"] == reference_model].set_index("subject")
    others = [model for model in models if model != reference_model]
    if others:
        best = max(
            others,
            key=lambda model: subject.loc[
                subject["model"] == model,
                "shift_macro_f1",
            ].mean(),
        )
        comparator = subject.loc[subject["model"] == best].set_index("subject")
        common = sorted(set(reference.index) & set(comparator.index))
        x = comparator.loc[common, "shift_macro_f1"].to_numpy(float)
        y = reference.loc[common, "shift_macro_f1"].to_numpy(float)
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.scatter(x, y, s=42)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"{best} all-shift Macro-F1")
        ax.set_ylabel(f"{reference_model} all-shift Macro-F1")
        ax.set_title("Subject-wise paired robustness")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{prefix}_paired.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def run_experiment(
    *,
    project_root: Path,
    subject_spec: str,
    seeds: list[int],
    models: list[str],
    reference_model: str,
    output_prefix: str,
    epochs: int,
    batch_size: int,
    lr: float,
    threads: int,
    device: str = "cpu",
    rebuild_cache: bool = False,
    force_retrain: bool = False,
    strict_data: bool = True,
) -> dict[str, Path]:
    paths = ensure_layout(project_root)
    if output_prefix != "14_final_main":
        checkpoint_root = project_root / "checkpoints" / output_prefix
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    else:
        checkpoint_root = paths["checkpoints"]

    result_dir = paths["results"]
    figure_dir = paths["figures"]
    log_dir = paths["logs"]
    raw_path = result_dir / f"{output_prefix}_raw_metrics.csv"
    subject_path = result_dir / f"{output_prefix}_subject_summary.csv"
    model_path = result_dir / f"{output_prefix}_model_summary.csv"
    stats_path = result_dir / f"{output_prefix}_paired_stats.csv"
    seed_path = result_dir / f"{output_prefix}_seed_stability.csv"
    report_path = result_dir / f"{output_prefix}_report.md"
    progress_path = log_dir / f"{output_prefix}_progress.txt"

    subjects = parse_subjects(subject_spec)
    total_jobs = len(subjects) * len(models) * len(seeds)
    started = time.time()

    if raw_path.exists():
        raw = pd.read_csv(raw_path, encoding="utf-8-sig")
    else:
        raw = pd.DataFrame()
    completed = set()
    if not raw.empty:
        aggregate = raw.loc[raw["position"] == -1]
        for row in aggregate.itertuples(index=False):
            completed.add((str(row.subject), str(row.model), int(row.seed)))
    rows = raw.to_dict("records") if not raw.empty else []

    def write_progress(current: str) -> None:
        done = len(completed)
        elapsed = time.time() - started
        eta = ""
        if done > 0 and total_jobs > done:
            eta_seconds = elapsed * (total_jobs - done) / done
            eta = f"\nETA_minutes: {eta_seconds / 60:.2f}"
        progress_path.write_text(
            "\n".join(
                [
                    f"Current: {current}",
                    f"Progress: {done}/{total_jobs}",
                    f"Percent: {100 * done / max(1, total_jobs):.2f}%",
                    f"Elapsed_minutes: {elapsed / 60:.2f}{eta}",
                    f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                ]
            ),
            encoding="utf-8",
        )

    for subject in subjects:
        payload = build_subject_cache(
            subject,
            paths["subjects"],
            paths["cache"],
            rebuild=rebuild_cache,
            strict=strict_data,
        )
        train = training_mask(payload["position"], payload["repetition"])
        X_normalized, _, _ = normalize_from_training(payload["X"], train)
        X_train = X_normalized[train]
        y_train = payload["y"][train]

        for seed in seeds:
            for model_name in models:
                key = (subject, model_name, seed)
                current = f"{subject} | {model_name} | seed={seed}"
                if key in completed:
                    write_progress(current + " [cached]")
                    continue
                write_progress(current)
                model, train_seconds, checkpoint = train_model(
                    subject=subject,
                    model_name=model_name,
                    seed=seed,
                    X_train=X_train,
                    y_train=y_train,
                    checkpoint_root=checkpoint_root,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    threads=threads,
                    device=device,
                    force_retrain=force_retrain,
                )
                result = evaluate_model(
                    model,
                    payload,
                    X_normalized,
                    batch_size=batch_size,
                    device=device,
                )
                for position_row in result["position_rows"]:
                    rows.append(
                        {
                            "subject": subject,
                            "model": model_name,
                            "seed": seed,
                            **position_row,
                            "ideal_macro_f1": result["ideal_macro_f1"],
                            "shift_macro_f1": result["shift_macro_f1"],
                            "degradation": result["degradation"],
                            "train_seconds": train_seconds,
                            "trainable_parameters": count_parameters(model),
                            "checkpoint": str(checkpoint),
                        }
                    )
                rows.append(
                    {
                        "subject": subject,
                        "model": model_name,
                        "seed": seed,
                        "position": -1,
                        "trial_macro_f1": result["shift_macro_f1"],
                        "trial_accuracy": np.nan,
                        "trial_balanced_accuracy": np.nan,
                        "n_trials": result["shift_n_trials"],
                        "ideal_macro_f1": result["ideal_macro_f1"],
                        "shift_macro_f1": result["shift_macro_f1"],
                        "degradation": result["degradation"],
                        "train_seconds": train_seconds,
                        "trainable_parameters": count_parameters(model),
                        "checkpoint": str(checkpoint),
                    }
                )
                completed.add(key)
                pd.DataFrame(rows).to_csv(raw_path, index=False, encoding="utf-8-sig")
                write_progress(current)

    raw = pd.DataFrame(rows)
    raw = raw.drop_duplicates(
        subset=["subject", "model", "seed", "position"],
        keep="last",
    )
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")

    subject, model_summary, stats = summarize(
        raw,
        models=models,
        reference_model=reference_model,
    )
    subject.to_csv(subject_path, index=False, encoding="utf-8-sig")
    model_summary.to_csv(model_path, index=False, encoding="utf-8-sig")
    stats.to_csv(stats_path, index=False, encoding="utf-8-sig")

    aggregate = raw.loc[raw["position"] == -1]
    seed_summary = (
        aggregate.groupby(["model", "seed"], as_index=False)
        .agg(
            n_subjects=("subject", "nunique"),
            shift_macro_f1_mean=("shift_macro_f1", "mean"),
            ideal_macro_f1_mean=("ideal_macro_f1", "mean"),
            degradation_mean=("degradation", "mean"),
        )
    )
    seed_summary.to_csv(seed_path, index=False, encoding="utf-8-sig")
    create_figures(
        subject,
        raw,
        figure_dir,
        prefix=output_prefix,
        models=models,
        reference_model=reference_model,
    )

    report = [
        f"# {output_prefix}",
        "",
        "## Protocol",
        "",
        f"- Subjects: {subject_spec}",
        f"- Seeds: {seeds}",
        f"- Models: {models}",
        "- Training: p0-r0/r1 only.",
        "- Ideal test: p0-r2.",
        "- Shift test: p1-p10.",
        "- Primary endpoint: pooled all-shift trial-level Macro-F1.",
        "- Statistical unit: participant after averaging seeds.",
        "",
        "## Model summary",
        "",
        model_summary.to_string(index=False),
        "",
        "## Paired statistics",
        "",
        stats.to_string(index=False),
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")
    write_progress("completed")

    return {
        "raw": raw_path,
        "subject": subject_path,
        "model": model_path,
        "stats": stats_path,
        "seed": seed_path,
        "report": report_path,
        "progress": progress_path,
    }
