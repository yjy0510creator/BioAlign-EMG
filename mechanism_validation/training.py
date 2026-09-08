from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .cyclic_ops import circular_mean_from_probs, continuous_circular_shift
from .losses import circular_distribution_loss, circular_regression_loss, normalized_feature_consistency


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    # Experimental fairness controls.  V2.0 trained every model with the same
    # synthetic shifted input and accidentally allowed auxiliary shift losses for
    # some negative controls.  V2.1 makes these switches explicit and stores them
    # in the checkpoint hash.
    use_synthetic_shift: bool = True
    use_shift_loss: bool = True
    use_consistency_loss: bool = True
    lambda_shift: float = 0.5
    lambda_consistency: float = 0.2
    max_synthetic_shift_channels: float = 4.0
    continuous_augmentation: bool = True
    shift_kappa: float = 8.0
    num_workers: int = 0

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def synthetic_shift(x: Tensor, max_abs: float, continuous: bool = True) -> Tuple[Tensor, Tensor]:
    b, c = x.shape[:2]
    if continuous:
        shift = (torch.rand(b, device=x.device) * 2 - 1) * float(max_abs)
    else:
        max_int = int(round(max_abs))
        shift = torch.randint(-max_int, max_int + 1, (b,), device=x.device).to(x.dtype)
    return continuous_circular_shift(x, shift), shift.remainder(c)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer, cfg: TrainConfig, device: torch.device) -> Dict[str, float]:
    model.train()
    total = {'loss': 0.0, 'gesture': 0.0, 'shift': 0.0, 'consistency': 0.0, 'n': 0}
    for batch in loader:
        x = batch['x'].to(device, non_blocking=True)
        y = batch['gesture'].to(device, non_blocking=True)

        if cfg.use_synthetic_shift:
            x_in, target_shift = synthetic_shift(x, cfg.max_synthetic_shift_channels, cfg.continuous_augmentation)
        else:
            x_in = x
            target_shift = torch.zeros(x.shape[0], dtype=x.dtype, device=device)

        reference_features = None
        if cfg.use_consistency_loss and hasattr(model, 'encode') and cfg.use_synthetic_shift:
            with torch.no_grad():
                reference_features = model.encode(x)

        logits, aux = model(x_in, return_aux=True)
        loss_g = F.cross_entropy(logits, y)
        loss_s = torch.zeros((), device=device)
        loss_c = torch.zeros((), device=device)

        if cfg.use_shift_loss and cfg.lambda_shift > 0 and 'shift_logits' in aux:
            loss_dist = circular_distribution_loss(aux['shift_logits'], target_shift, kappa=cfg.shift_kappa)
            pred_shift, _, _ = circular_mean_from_probs(F.softmax(aux['shift_logits'], dim=-1))
            loss_reg = circular_regression_loss(pred_shift, target_shift, x.shape[1])
            loss_s = 0.5 * loss_dist + 0.5 * loss_reg

        if cfg.use_consistency_loss and cfg.lambda_consistency > 0 and reference_features is not None and 'features_post' in aux:
            loss_c = normalized_feature_consistency(aux['features_post'], reference_features)

        loss = loss_g + cfg.lambda_shift * loss_s + cfg.lambda_consistency * loss_c

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        n = len(x)
        total['loss'] += float(loss.detach()) * n
        total['gesture'] += float(loss_g.detach()) * n
        total['shift'] += float(loss_s.detach()) * n
        total['consistency'] += float(loss_c.detach()) * n
        total['n'] += n
    return {k: v / max(1, total['n']) for k, v in total.items() if k != 'n'}


def fit(model: nn.Module, loader: DataLoader, cfg: TrainConfig, device: Optional[str] = None) -> list[dict]:
    seed_everything(cfg.seed)
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1))
    history = []
    for epoch in range(cfg.epochs):
        metrics = train_epoch(model, loader, optimizer, cfg, dev)
        scheduler.step()
        metrics.update({'epoch': epoch + 1, 'lr': scheduler.get_last_lr()[0]})
        history.append(metrics)
    return history


def save_checkpoint(model: nn.Module, path: str | Path, cfg: TrainConfig, extra: Optional[dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'train_config': asdict(cfg),
        'config_digest': cfg.digest(),
        'extra': extra or {},
    }, path)


def checkpoint_matches(path: str | Path, cfg: TrainConfig) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    obj = torch.load(path, map_location='cpu')
    return obj.get('config_digest') == cfg.digest()
