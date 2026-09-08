from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_gesture_heatmap(df: pd.DataFrame, metric: str, position, out: str | Path) -> None:
    sub = df[(df.metric == metric) & (df.position.astype(str) == str(position))]
    pivot = sub.pivot(index='gesture', columns='channel', values='value').sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(pivot.to_numpy(), aspect='auto')
    ax.set_xlabel('Electrode channel')
    ax.set_ylabel('Gesture')
    ax.set_xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_entropy_performance(entropy: np.ndarray, correct: np.ndarray, out: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(entropy, correct.astype(float), alpha=0.2)
    bins = np.linspace(0, 1, 11)
    centers = (bins[:-1] + bins[1:]) / 2
    means = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (entropy >= lo) & (entropy < hi)
        means.append(np.nan if not m.any() else correct[m].mean())
    ax.plot(centers, means, marker='o')
    ax.set_xlabel('Normalized correspondence entropy')
    ax.set_ylabel('Window accuracy')
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
