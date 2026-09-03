from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bioalign_emg.constants import PRIMARY_MODELS
from bioalign_emg.data import parse_subjects
from bioalign_emg.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="BioAlign-EMG final main experiment")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--subjects", default="h0-h29")
    parser.add_argument("--seeds", default="42,2026,3407")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--threads",
        type=int,
        default=max(2, min(8, (os.cpu_count() or 8) // 2)),
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--models", default=",".join(PRIMARY_MODELS))
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--allow-incomplete-data", action="store_true")
    args = parser.parse_args()

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    parse_subjects(args.subjects)

    outputs = run_experiment(
        project_root=args.project_root.resolve(),
        subject_spec=args.subjects,
        seeds=seeds,
        models=models,
        reference_model="BioAlign_RingAug",
        output_prefix="14_final_main",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        threads=args.threads,
        device=args.device,
        rebuild_cache=args.rebuild_cache,
        force_retrain=args.force_retrain,
        strict_data=not args.allow_incomplete_data,
    )
    print("\nFINAL MAIN EXPERIMENT COMPLETED")
    for key, path in outputs.items():
        print(f"{key:>10}: {path}")


if __name__ == "__main__":
    main()
