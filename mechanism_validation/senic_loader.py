from __future__ import annotations

"""SeNic session-0 reader used by BioAlign-EMG mechanism validation.

This module replaces the old ``legacy_v1`` dependency.  It reads the public
SeNic subject folders directly, parses the standard session-0 CSV files, and
creates 8-channel overlapping windows with trial-level metadata.  The reader is
kept intentionally narrow: it supports the exact SeNic naming convention used in
this project instead of carrying unused legacy experiment code.
"""

from pathlib import Path
from typing import Iterable
import re

import numpy as np
import pandas as pd

FS = 200.0
WINDOW_LEN = 50
WINDOW_STEP = 10
N_CHANNELS = 8
EXPECTED_SESSION0_TRIALS = 11 * 7 * 3
GESTURES = [
    "eversion",
    "fist",
    "open_hand",
    "pinch_forefinger",
    "pinch_middlefinger",
    "two",
    "varus",
]
GESTURE_TO_INDEX = {gesture: index for index, gesture in enumerate(GESTURES)}
FILENAME_RE = re.compile(
    r"^emg_p(?P<position>\d+)_r(?P<rep>\d+)_(?P<gesture>.+)\.csv$",
    re.IGNORECASE,
)


def parse_subjects(spec: str) -> list[str]:
    spec = spec.strip()
    if not spec:
        raise ValueError("Subject specification is empty.")
    if "-" in spec and "," not in spec:
        left, right = spec.split("-", 1)
        if not (left.startswith("h") and right.startswith("h")):
            raise ValueError(f"Invalid subject range: {spec}")
        start, stop = int(left[1:]), int(right[1:])
        if stop < start:
            raise ValueError(f"Invalid subject range: {spec}")
        return [f"h{i}" for i in range(start, stop + 1)]
    values = [item.strip() for item in spec.split(",") if item.strip()]
    if not all(item.startswith("h") and item[1:].isdigit() for item in values):
        raise ValueError(f"Invalid subject list: {spec}")
    return values


def locate_session(path: Path) -> str:
    for part in reversed(path.parts[:-1]):
        if part.isdigit():
            return part
    return "0"


def find_session0_csvs(subject_dir: Path) -> list[Path]:
    return [path for path in sorted(subject_dir.rglob("*.csv")) if locate_session(path) == "0"]


def read_8ch_csv(path: Path) -> np.ndarray:
    """Return a [time, 8] float32 array from a SeNic trial CSV."""
    encodings = ("utf-8-sig", "utf-8", "gb18030", "latin1")
    separators = (",", "\t", ";", r"\s+")
    for encoding in encodings:
        for separator in separators:
            try:
                frame = pd.read_csv(
                    path,
                    header=None,
                    sep=separator,
                    engine="python",
                    encoding=encoding,
                    on_bad_lines="skip",
                )
                if frame.empty:
                    continue
                if frame.shape[1] == 1 and separator != r"\s+":
                    continue
                numeric = frame.apply(pd.to_numeric, errors="coerce")
                numeric = numeric.dropna(axis=1, how="all")
                numeric = numeric.loc[:, numeric.notna().mean(axis=0) >= 0.80]
                if numeric.shape[0] < WINDOW_LEN or numeric.shape[1] < N_CHANNELS:
                    continue

                # Drop obvious time/sample-count columns when more than 8 numeric
                # columns are present.  SeNic trial files are expected to contain
                # eight EMG channels; auxiliary monotonic columns are not channels.
                if numeric.shape[1] > N_CHANNELS:
                    keep = []
                    for column in numeric.columns:
                        values = numeric[column].dropna().to_numpy(float)
                        if len(values) < 10:
                            continue
                        unique_ratio = len(np.unique(values)) / len(values)
                        diffs = np.diff(values)
                        monotonic = bool(np.all(diffs >= 0) or np.all(diffs <= 0))
                        if monotonic and unique_ratio > 0.95:
                            continue
                        keep.append(column)
                    if len(keep) >= N_CHANNELS:
                        numeric = numeric.loc[:, keep]

                if numeric.shape[1] > N_CHANNELS:
                    scores: dict[object, float] = {}
                    for column in numeric.columns:
                        values = numeric[column].dropna().to_numpy(float)
                        scores[column] = float(np.std(np.diff(values))) if len(values) > 2 else -np.inf
                    selected = sorted(numeric.columns, key=lambda c: scores[c], reverse=True)[:N_CHANNELS]
                    selected = sorted(selected, key=lambda c: list(numeric.columns).index(c))
                    numeric = numeric.loc[:, selected]

                numeric = numeric.iloc[:, :N_CHANNELS].interpolate(limit_direction="both").fillna(0.0)
                array = numeric.to_numpy(np.float32)
                if array.shape[1] == N_CHANNELS and np.isfinite(array).all():
                    return array
            except Exception:
                continue
    raise ValueError(f"Could not parse eight-channel sEMG CSV: {path}")


def make_windows(signal: np.ndarray, window_len: int = WINDOW_LEN, step: int = WINDOW_STEP) -> np.ndarray:
    """Split [time, channels] signal into [windows, channels, samples]."""
    if signal.ndim != 2 or signal.shape[1] != N_CHANNELS:
        raise ValueError(f"Expected [time,{N_CHANNELS}] signal, got {signal.shape}")
    if len(signal) < window_len:
        signal = np.pad(signal, ((0, window_len - len(signal)), (0, 0)), mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(signal, window_len, axis=0)
    starts = np.arange(0, len(signal) - window_len + 1, step, dtype=int)
    if len(starts) == 0:
        starts = np.asarray([0], dtype=int)
    return np.ascontiguousarray(view[starts], dtype=np.float32)


def subject_cache_path(cache_dir: Path, subject: str) -> Path:
    return cache_dir / f"{subject}_session0_windows.npz"


def build_subject_cache(
    subject: str,
    subjects_root: Path,
    cache_dir: Path,
    *,
    rebuild: bool = False,
    strict: bool = True,
) -> dict[str, np.ndarray]:
    """Read/cache one subject's session-0 windows and metadata."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = subject_cache_path(cache_dir, subject)
    if rebuild and cache.exists():
        cache.unlink()
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            payload = {
                "X": z["X"].astype(np.float32),
                "y": z["y"].astype(np.int64),
                "position": z["position"].astype(np.int64),
                "repetition": z["repetition"].astype(np.int64),
                "trial_id": z["trial_id"].astype(str),
            }
        if payload["X"].ndim == 3 and payload["X"].shape[1:] == (N_CHANNELS, WINDOW_LEN):
            return payload
        cache.unlink()

    subject_dir = subjects_root / subject
    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")
    files = find_session0_csvs(subject_dir)
    if strict and len(files) != EXPECTED_SESSION0_TRIALS:
        raise RuntimeError(
            f"{subject}: expected {EXPECTED_SESSION0_TRIALS} session-0 CSV files, found {len(files)}."
        )
    if not files:
        raise RuntimeError(f"{subject}: no session-0 CSV files found.")

    x_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    position_blocks: list[np.ndarray] = []
    repetition_blocks: list[np.ndarray] = []
    trial_blocks: list[np.ndarray] = []

    for path in files:
        match = FILENAME_RE.match(path.name)
        if not match:
            continue
        gesture = match.group("gesture").lower()
        if gesture not in GESTURE_TO_INDEX:
            continue
        position = int(match.group("position"))
        repetition = int(match.group("rep"))
        label = GESTURE_TO_INDEX[gesture]
        trial_id = path.relative_to(subjects_root).as_posix()
        windows = make_windows(read_8ch_csv(path))
        n_windows = len(windows)
        x_blocks.append(windows.astype(np.float32))
        y_blocks.append(np.full(n_windows, label, dtype=np.int8))
        position_blocks.append(np.full(n_windows, position, dtype=np.int8))
        repetition_blocks.append(np.full(n_windows, repetition, dtype=np.int8))
        trial_blocks.append(np.full(n_windows, trial_id, dtype="<U220"))

    if not x_blocks:
        raise RuntimeError(f"{subject}: no valid gesture files were parsed.")
    payload = {
        "X": np.concatenate(x_blocks).astype(np.float32),
        "y": np.concatenate(y_blocks).astype(np.int64),
        "position": np.concatenate(position_blocks).astype(np.int64),
        "repetition": np.concatenate(repetition_blocks).astype(np.int64),
        "trial_id": np.concatenate(trial_blocks).astype(str),
    }
    np.savez_compressed(cache, **payload, fs=np.asarray([FS], dtype=np.float32))
    return payload


def audit_senic_dataset(subjects_root: Path, subjects: Iterable[str]) -> pd.DataFrame:
    rows = []
    for subject in subjects:
        subject_dir = subjects_root / subject
        files = find_session0_csvs(subject_dir) if subject_dir.exists() else []
        positions, repetitions, gestures = set(), set(), set()
        named = 0
        for path in files:
            match = FILENAME_RE.match(path.name)
            if not match:
                continue
            named += 1
            positions.add(int(match.group("position")))
            repetitions.add(int(match.group("rep")))
            gestures.add(match.group("gesture").lower())
        angle_files = list(subject_dir.rglob(f"Angle_{subject}_0.xlsx")) if subject_dir.exists() else []
        status = "OK"
        if not subject_dir.exists():
            status = "MISSING_SUBJECT_DIR"
        elif len(files) != EXPECTED_SESSION0_TRIALS:
            status = "CHECK_SESSION0_COUNT"
        elif named != EXPECTED_SESSION0_TRIALS:
            status = "CHECK_FILENAMES"
        elif positions != set(range(11)):
            status = "CHECK_POSITIONS"
        elif repetitions != {0, 1, 2}:
            status = "CHECK_REPETITIONS"
        elif gestures != set(GESTURE_TO_INDEX):
            status = "CHECK_GESTURES"
        rows.append({
            "subject": subject,
            "session0_csv": len(files),
            "recognized_filenames": named,
            "position_count": len(positions),
            "repetition_count": len(repetitions),
            "gesture_count": len(gestures),
            "angle_file_count": len(angle_files),
            "status": status,
        })
    return pd.DataFrame(rows)
