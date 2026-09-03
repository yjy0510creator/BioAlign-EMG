from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bioalign_emg.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optional no-alignment, uniform-mixture, hard-argmax and soft controls"
    )
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
    args = parser.parse_args()

    outputs = run_experiment(
        project_root=args.project_root.resolve(),
        subject_spec=args.subjects,
        seeds=[int(value) for value in args.seeds.split(",")],
        models=[
            "BioAlign_RingAug",
            "BioAlign_NoAlignment",
            "BioAlign_UniformMixture",
            "BioAlign_HardArgmax",
        ],
        reference_model="BioAlign_RingAug",
        output_prefix="17_mechanism_controls",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        threads=args.threads,
        device=args.device,
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
