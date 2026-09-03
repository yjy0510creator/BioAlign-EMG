from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bioalign_emg.angle import run_angle_analysis
from bioalign_emg.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal alignment ablation and angle evidence")
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
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]

    ablation_outputs = run_experiment(
        project_root=project_root,
        subject_spec=args.subjects,
        seeds=seeds,
        models=["BioAlign_RingAug", "BioAlign_NoAlignment"],
        reference_model="BioAlign_RingAug",
        output_prefix="15_alignment_ablation",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        threads=args.threads,
        device=args.device,
        force_retrain=args.force_retrain,
    )

    angle_outputs = run_angle_analysis(
        main_raw_path=project_root / "results" / "14_final_main_raw_metrics.csv",
        subjects_root=project_root / "data" / "raw" / "SeNic" / "subjects",
        result_dir=project_root / "results",
        figure_dir=project_root / "figures",
    )

    print("\nFINAL EVIDENCE PACK COMPLETED")
    print("\nAblation:")
    for key, path in ablation_outputs.items():
        print(f"  {key}: {path}")
    print("\nAngle analysis:")
    for key, path in angle_outputs.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
