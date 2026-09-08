from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .angle_utils import parse_subject_angle_table, write_angle_audit
from .data import StandardWindows, save_standard_npz

from .senic_loader import build_subject_cache, parse_subjects


def build_subject(subject: str, project_root: Path, output_dir: Path, rebuild: bool = False) -> Path:
    subjects_root = project_root / "data" / "raw" / "SeNic" / "subjects"
    cache_dir = project_root / "data" / "processed" / "legacy_session0"
    payload = build_subject_cache(
        subject=subject,
        subjects_root=subjects_root,
        cache_dir=cache_dir,
        rebuild=rebuild,
        strict=True,
    )
    x = payload["X"].astype(np.float32)
    gesture = payload["y"].astype(np.int64)
    position = np.asarray([f"p{int(v)}" for v in payload["position"]], dtype="<U3")
    repetition = np.asarray([f"r{int(v)}" for v in payload["repetition"]], dtype="<U2")
    trial = payload["trial_id"].astype(str)
    subj = np.repeat(subject, len(x))

    atab = parse_subject_angle_table(subject, subjects_root)
    amap = dict(zip(atab["position"].astype(str), atab["angle_deg"].astype(float)))
    angle_deg = np.asarray([amap.get(p, np.nan) for p in position], dtype=float)

    data = StandardWindows(x, gesture, position, repetition, trial, subj, angle_deg).validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{subject}.npz"
    save_standard_npz(data, out_path)
    return out_path


def main():
    p = argparse.ArgumentParser(description="Build standardized NPZ files from real SeNic subject folders")
    p.add_argument("--project-root", required=True, help="e.g. D:\\BioSelect_EMG")
    p.add_argument("--code-root", default=None, help="root of this V2.1 package; defaults to current package root")
    p.add_argument("--subjects", default="h0", help="h0,h1 or h0-h29")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args()

    project_root = Path(args.project_root)
    # --code-root is accepted for backward compatibility but no longer required.
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "data" / "processed_v21"

    subjects = parse_subjects(args.subjects)
    rows = []
    for subject in subjects:
        out = build_subject(subject, project_root, output_dir, rebuild=args.rebuild)
        with np.load(out, allow_pickle=True) as z:
            trials = np.asarray(z["trial"]).astype(str)
            pos = np.asarray(z["position"]).astype(str)
            rep = np.asarray(z["repetition"]).astype(str)
            rows.append({
                "subject": subject,
                "output": str(out),
                "windows": int(len(z["x"])),
                "trials": int(len(np.unique(trials))),
                "train_trials": int(len(np.unique(trials[(pos == "p0") & np.isin(rep, ["r0", "r1"])]))),
                "ideal_trials": int(len(np.unique(trials[(pos == "p0") & (rep == "r2")]))),
                "shift_trials": int(len(np.unique(trials[pos != "p0"]))),
                "finite_angle_windows": int(np.isfinite(z["angle_deg"]).sum()),
            })
        print(rows[-1])
    summary = pd.DataFrame(rows)
    summary_path = output_dir / "build_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_angle_audit(subjects, project_root / "data" / "raw" / "SeNic" / "subjects", output_dir / "angle_audit.csv")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
