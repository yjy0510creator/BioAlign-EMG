from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .augmentation import ring_augment
from .models import AUGMENTED_MODELS, build_model, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_path(
    checkpoint_root: Path,
    subject: str,
    model_name: str,
    seed: int,
    epochs: int,
) -> Path:
    folder = checkpoint_root / subject
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{model_name}_seed{seed}_epoch{epochs}.pt"


def train_model(
    *,
    subject: str,
    model_name: str,
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    checkpoint_root: Path,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    gradient_clip: float = 5.0,
    threads: int = 8,
    device: str = "cpu",
    force_retrain: bool = False,
) -> tuple[torch.nn.Module, float, Path]:
    set_seed(seed)
    if device == "cpu":
        torch.set_num_threads(max(1, threads))

    model = build_model(model_name).to(device)
    checkpoint = checkpoint_path(
        checkpoint_root,
        subject,
        model_name,
        seed,
        epochs,
    )

    if checkpoint.exists() and not force_retrain:
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model"])
        model.eval()
        return model, 0.0, checkpoint

    dataset = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
    )

    start = time.time()
    progress = tqdm(
        range(1, epochs + 1),
        desc=f"{subject} {model_name} s{seed}",
        unit="ep",
        dynamic_ncols=True,
        leave=False,
    )
    for epoch in progress:
        model.train()
        loss_sum = 0.0
        accuracy_sum = 0.0
        batches = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            if model_name in AUGMENTED_MODELS:
                xb, _ = ring_augment(xb)
            output = model(xb)
            loss = F.cross_entropy(output["logits"], yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            loss_sum += float(loss.item())
            accuracy_sum += float(
                (output["logits"].argmax(dim=1) == yb).float().mean().item()
            )
            batches += 1
        scheduler.step()
        progress.set_postfix(
            loss=f"{loss_sum / max(1, batches):.4f}",
            acc=f"{accuracy_sum / max(1, batches):.3f}",
            lr=f"{scheduler.get_last_lr()[0]:.1e}",
        )

    elapsed = time.time() - start
    model.eval()
    torch.save(
        {
            "model": model.state_dict(),
            "subject": subject,
            "model_name": model_name,
            "seed": seed,
            "epochs": epochs,
            "trainable_parameters": count_parameters(model),
            "train_seconds": elapsed,
            "protocol": "session0; train p0-r0/r1; ideal p0-r2; shift p1-p10",
        },
        checkpoint,
    )
    return model, elapsed, checkpoint
