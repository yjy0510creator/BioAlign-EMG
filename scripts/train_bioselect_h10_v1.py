"""Compatibility entry point for a one-subject prototype run.

The paper-facing final run is scripts/train_bioalign_final_30subjects_3seeds.py.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bioalign_emg.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--subject", default="h10")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=max(2, min(8, (os.cpu_count() or 8)//2)))
    args = parser.parse_args()
    run_experiment(
        project_root=args.project_root.resolve(),
        subject_spec=args.subject,
        seeds=[args.seed],
        models=["TCN_plain", "TCN_RingAug", "SE_RingAug", "CBAM_RingAug", "BioAlign_RingAug"],
        reference_model="BioAlign_RingAug",
        output_prefix="07_h10_prototype",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=1e-3,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
