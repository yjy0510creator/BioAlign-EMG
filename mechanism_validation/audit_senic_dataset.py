from __future__ import annotations

import argparse
from pathlib import Path

from .senic_loader import audit_senic_dataset, parse_subjects


def main() -> None:
    p = argparse.ArgumentParser(description="Audit SeNic session-0 subject folders before training")
    p.add_argument("--project-root", required=True, help="e.g. D:\\BioSelect_EMG")
    p.add_argument("--subjects", default="h0-h29")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    project_root = Path(args.project_root)
    subjects_root = project_root / "data" / "raw" / "SeNic" / "subjects"
    subjects = parse_subjects(args.subjects)
    df = audit_senic_dataset(subjects_root, subjects)
    out = Path(args.out) if args.out else project_root / "results_v21" / "00_senic_dataset_audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
