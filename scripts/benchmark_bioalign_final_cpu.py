from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bioalign_emg.constants import PRIMARY_MODELS
from bioalign_emg.deployment import run_cpu_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--subject", default="h0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--models", default=",".join(PRIMARY_MODELS))
    parser.add_argument("--threads", default="1")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--runs", type=int, default=1500)
    parser.add_argument("--max-windows", type=int, default=1500)
    args = parser.parse_args()

    models = [value.strip() for value in args.models.split(",") if value.strip()]
    threads = [int(value.strip()) for value in args.threads.split(",") if value.strip()]
    outputs = run_cpu_benchmark(
        project_root=args.project_root.resolve(),
        subject=args.subject,
        seed=args.seed,
        epochs=args.epochs,
        models=models,
        thread_settings=threads,
        warmup=args.warmup,
        runs=args.runs,
        max_windows=args.max_windows,
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
