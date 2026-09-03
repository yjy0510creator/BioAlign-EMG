from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bioalign_emg.angle import run_angle_analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="Main raw metrics CSV; defaults to results/14_final_main_raw_metrics.csv",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    raw = args.raw or (root / "results" / "14_final_main_raw_metrics.csv")
    outputs = run_angle_analysis(
        main_raw_path=raw,
        subjects_root=root / "data" / "raw" / "SeNic" / "subjects",
        result_dir=root / "results",
        figure_dir=root / "figures",
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
