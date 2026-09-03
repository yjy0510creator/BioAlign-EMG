from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def ci95_half_width(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if len(array) <= 1:
        return 0.0
    return float(1.96 * array.std(ddof=1) / math.sqrt(len(array)))


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=float)
    if len(values) == 0:
        return []
    order = np.argsort(values)
    adjusted_sorted = np.empty(len(values), dtype=float)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        adjusted = (m - rank) * values[index]
        running = max(running, adjusted)
        adjusted_sorted[rank] = min(1.0, running)
    output = np.empty(len(values), dtype=float)
    for rank, index in enumerate(order):
        output[index] = adjusted_sorted[rank]
    return output.tolist()


def paired_wilcoxon(
    reference: np.ndarray,
    comparator: np.ndarray,
    *,
    alternative: str = "greater",
) -> tuple[float, float]:
    difference = np.asarray(reference, dtype=float) - np.asarray(comparator, dtype=float)
    if np.allclose(difference, 0):
        return 0.0, 1.0
    result = wilcoxon(
        difference,
        alternative=alternative,
        zero_method="wilcox",
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


def paired_comparisons(
    subject_summary: pd.DataFrame,
    *,
    reference_model: str,
    comparator_models: list[str],
    metric: str = "shift_macro_f1",
    alternative: str = "greater",
) -> pd.DataFrame:
    reference = subject_summary.loc[
        subject_summary["model"] == reference_model
    ].set_index("subject")
    rows = []
    for comparator_model in comparator_models:
        comparator = subject_summary.loc[
            subject_summary["model"] == comparator_model
        ].set_index("subject")
        common = sorted(set(reference.index) & set(comparator.index))
        if not common:
            continue
        x = reference.loc[common, metric].to_numpy(float)
        y = comparator.loc[common, metric].to_numpy(float)
        difference = x - y
        statistic, p_value = paired_wilcoxon(
            x,
            y,
            alternative=alternative,
        )
        rows.append(
            {
                "comparison": f"{reference_model} > {comparator_model}",
                "n_subjects": len(common),
                "reference_mean": float(x.mean()),
                "comparator_mean": float(y.mean()),
                "mean_paired_difference": float(difference.mean()),
                "reference_wins": int(np.sum(difference > 0)),
                "ties": int(np.sum(np.isclose(difference, 0))),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )
    output = pd.DataFrame(rows)
    if not output.empty:
        output["p_holm"] = holm_adjust(output["p_value"].tolist())
    return output
