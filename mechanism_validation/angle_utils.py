from __future__ import annotations

"""Angle parsing utilities for SeNic electrode-position files.

The original Angle_<subject>_0.xlsx table stores one angular coordinate per
physical electrode for positions p0-p10.  For mechanism validation we need a
*directed* rotation label, not only a median absolute displacement.  The helper
below estimates the rigid component of the ring rotation relative to p0 by a
circular mean of the eight per-channel differences, and also reports rigidity
quality diagnostics.  This label is an analysis target, not a training feature.
"""

import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

CHANNEL_COLUMNS = [f"CH{i}" for i in range(1, 9)]


def wrap_deg(x: np.ndarray | float) -> np.ndarray | float:
    """Wrap degrees to [-180, 180)."""
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


def circular_mean_deg(values_deg: np.ndarray) -> float:
    values = np.asarray(values_deg, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    r = np.deg2rad(values)
    return float(np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))


def circular_resultant(values_deg: np.ndarray) -> float:
    values = np.asarray(values_deg, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    r = np.deg2rad(values)
    return float(np.sqrt(np.sin(r).mean() ** 2 + np.cos(r).mean() ** 2))


def circular_mad_deg(values_deg: np.ndarray, center_deg: float) -> float:
    values = np.asarray(values_deg, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0 or not np.isfinite(center_deg):
        return float("nan")
    return float(np.median(np.abs(wrap_deg(values - center_deg))))


def find_angle_file(subject_dir: Path, subject: str, session: int = 0) -> Optional[Path]:
    candidates = sorted(subject_dir.rglob(f"Angle_{subject}_{session}.xlsx"))
    if candidates:
        return candidates[0]
    # fallback for copied/renamed files
    candidates = sorted(subject_dir.rglob(f"Angle*{subject}*{session}*.xlsx"))
    return candidates[0] if candidates else None


def parse_subject_angle_table(subject: str, subjects_root: str | Path, session: int = 0) -> pd.DataFrame:
    subjects_root = Path(subjects_root)
    subject_dir = subjects_root / subject
    path = find_angle_file(subject_dir, subject, session=session)
    if path is None:
        raise FileNotFoundError(f"No Angle_{subject}_{session}.xlsx found under {subject_dir}")
    table = pd.read_excel(path)
    if "ID" not in table.columns or not all(c in table.columns for c in CHANNEL_COLUMNS):
        raise ValueError(f"Angle file lacks ID/CH1-CH8 columns: {path}")
    table = table.loc[table["ID"].between(0, 10)].copy()
    if table.empty:
        raise ValueError(f"Angle file has no ID 0-10 rows: {path}")
    ref_rows = table.loc[table["ID"] == 0, CHANNEL_COLUMNS]
    if ref_rows.empty:
        raise ValueError(f"Angle file has no p0 reference row: {path}")
    reference = ref_rows.iloc[0].to_numpy(float)
    rows = []
    for _, row in table.iterrows():
        pos = int(row["ID"])
        current = row[CHANNEL_COLUMNS].to_numpy(float)
        delta = wrap_deg(current - reference)
        directed = circular_mean_deg(delta)
        rows.append({
            "subject": subject,
            "position": f"p{pos}",
            "position_id": pos,
            "angle_deg": directed,
            "angle_abs_median_deg": float(np.median(np.abs(delta))),
            "angle_resultant": circular_resultant(delta),
            "angle_mad_deg": circular_mad_deg(delta, directed),
            "angle_file": str(path),
            **{f"delta_ch{i+1}_deg": float(delta[i]) for i in range(len(delta))},
        })
    out = pd.DataFrame(rows).sort_values("position_id").reset_index(drop=True)
    # Force p0 to exact zero for downstream equality checks.
    out.loc[out["position_id"] == 0, "angle_deg"] = 0.0
    return out


def attach_window_angles(position: np.ndarray, subject: np.ndarray, subjects_root: str | Path) -> np.ndarray:
    """Return window-level directed angle labels matched by subject and position."""
    position = np.asarray(position).astype(str)
    subject = np.asarray(subject).astype(str)
    angles = np.full(len(position), np.nan, dtype=float)
    for sub in np.unique(subject):
        table = parse_subject_angle_table(sub, subjects_root).set_index("position")
        for pos in np.unique(position[subject == sub]):
            key = pos if pos.startswith("p") else f"p{int(pos)}"
            if key in table.index:
                angles[(subject == sub) & (position == pos)] = float(table.loc[key, "angle_deg"])
    return angles


def write_angle_audit(subjects: Iterable[str], subjects_root: str | Path, out_csv: str | Path) -> pd.DataFrame:
    frames = []
    for subject in subjects:
        try:
            frames.append(parse_subject_angle_table(subject, subjects_root))
        except Exception as exc:
            frames.append(pd.DataFrame([{"subject": subject, "status": f"ERROR: {exc}"}]))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return df
