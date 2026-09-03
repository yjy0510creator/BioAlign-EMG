from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bioalign_emg.data import audit_dataset, parse_subjects


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--subjects", default="h0-h29")
    args = parser.parse_args()

    subjects_root = args.project_root / "data" / "raw" / "SeNic" / "subjects"
    result_dir = args.project_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    frame = audit_dataset(subjects_root, parse_subjects(args.subjects))
    output = result_dir / "00_senic_dataset_audit.csv"
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    print(frame.to_string(index=False))
    print(f"\nSaved: {output}")
    bad = frame.loc[frame["status"] != "OK"]
    if not bad.empty:
        raise SystemExit(f"Dataset audit found {len(bad)} subject(s) requiring attention.")


if __name__ == "__main__":
    main()
