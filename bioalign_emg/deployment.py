from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import build_subject_cache, normalize_from_training, training_mask
from .models import build_model, count_parameters


class LogitsOnly(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["logits"]


def benchmark_model(
    model: nn.Module,
    X: np.ndarray,
    *,
    threads: int,
    warmup: int,
    runs: int,
) -> dict[str, float]:
    torch.set_num_threads(max(1, threads))
    wrapper = LogitsOnly(model).eval()
    x = torch.from_numpy(X.astype(np.float32))
    with torch.inference_mode():
        for index in range(warmup):
            wrapper(x[index % len(x) : index % len(x) + 1])
        timings = []
        for index in range(runs):
            sample = x[index % len(x) : index % len(x) + 1]
            started = time.perf_counter_ns()
            wrapper(sample)
            timings.append((time.perf_counter_ns() - started) / 1e6)
    values = np.asarray(timings, dtype=float)
    return {
        "mean_latency_ms": float(values.mean()),
        "p50_latency_ms": float(np.percentile(values, 50)),
        "p95_latency_ms": float(np.percentile(values, 95)),
        "p99_latency_ms": float(np.percentile(values, 99)),
        "windows_per_second": float(1000.0 / values.mean()),
    }


def run_cpu_benchmark(
    *,
    project_root: Path,
    subject: str = "h0",
    seed: int = 2026,
    epochs: int = 20,
    models: list[str],
    thread_settings: list[int],
    warmup: int = 200,
    runs: int = 1500,
    max_windows: int = 1500,
) -> dict[str, Path]:
    subjects_root = project_root / "data" / "raw" / "SeNic" / "subjects"
    cache_dir = project_root / "data" / "processed" / "14_final_session0_cache"
    checkpoint_root = project_root / "checkpoints" / "14_final_main"
    result_dir = project_root / "results"
    deploy_dir = project_root / "deploy" / "14_final"
    result_dir.mkdir(parents=True, exist_ok=True)
    deploy_dir.mkdir(parents=True, exist_ok=True)

    payload = build_subject_cache(subject, subjects_root, cache_dir)
    train = training_mask(payload["position"], payload["repetition"])
    X, _, _ = normalize_from_training(payload["X"], train)
    X = X[~train]
    if len(X) > max_windows:
        indices = np.linspace(0, len(X) - 1, max_windows).astype(int)
        X = X[indices]

    rows = []
    for model_name in models:
        checkpoint = (
            checkpoint_root
            / subject
            / f"{model_name}_seed{seed}_epoch{epochs}.pt"
        )
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}. Run the main experiment first."
            )
        model = build_model(model_name)
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])
        model.eval()

        example = torch.from_numpy(X[:1])
        wrapper = LogitsOnly(model).eval()
        with torch.inference_mode():
            traced = torch.jit.trace(wrapper, example)
            traced = torch.jit.freeze(traced)
        torchscript_path = deploy_dir / f"{model_name}_{subject}_seed{seed}_torchscript.pt"
        traced.save(str(torchscript_path))

        for threads in thread_settings:
            result = benchmark_model(
                model,
                X,
                threads=threads,
                warmup=warmup,
                runs=runs,
            )
            rows.append(
                {
                    "model": model_name,
                    "subject": subject,
                    "seed": seed,
                    "threads": threads,
                    "logical_cpu_cores": os.cpu_count(),
                    "real_windows_used": len(X),
                    "trainable_parameters": count_parameters(model),
                    "checkpoint_mb": checkpoint.stat().st_size / 1024**2,
                    "torchscript_mb": torchscript_path.stat().st_size / 1024**2,
                    "input_shape": "1x8x50",
                    **result,
                }
            )

    frame = pd.DataFrame(rows)
    csv_path = result_dir / "16_final_cpu_latency.csv"
    report_path = result_dir / "16_final_cpu_latency_report.md"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    report_path.write_text(
        "\n".join(
            [
                "# BioAlign-EMG CPU deployment benchmark",
                "",
                "- Batch size: 1",
                "- Input: real normalized SeNic session-0 windows (1×8×50)",
                f"- Warm-up calls: {warmup}",
                f"- Timed calls: {runs}",
                "",
                frame.to_string(index=False),
                "",
                "Timing covers neural forward inference only.",
            ]
        ),
        encoding="utf-8",
    )
    return {"csv": csv_path, "report": report_path, "deploy": deploy_dir}
