from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .cyclic_ops import continuous_circular_shift
from .models import BioAlignV2
from .training import TrainConfig, fit, seed_everything
from .metrics import classification_metrics, circular_mae_deg


class SyntheticRingDataset(Dataset):
    def __init__(self, n: int = 1400, channels: int = 8, samples: int = 50, classes: int = 7, seed: int = 0):
        rng = np.random.default_rng(seed)
        templates = rng.normal(size=(classes, channels, samples)).astype(np.float32)
        # Smooth templates over time and channels.
        templates = (templates + np.roll(templates, 1, -1) + np.roll(templates, -1, -1)) / 3
        y = rng.integers(0, classes, n)
        x0 = templates[y] + rng.normal(scale=0.15, size=(n, channels, samples)).astype(np.float32)
        shifts = rng.uniform(0, channels, n).astype(np.float32)
        x = continuous_circular_shift(torch.from_numpy(x0), torch.from_numpy(shifts)).numpy()
        self.x = x.astype(np.float32)
        self.y = y.astype(np.int64)
        self.shifts = shifts

    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        return {'x': torch.from_numpy(self.x[i]), 'gesture': torch.tensor(self.y[i]), 'position': 'synthetic', 'repetition': '0', 'trial': str(i), 'index': i}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='synthetic_validation')
    p.add_argument('--epochs', type=int, default=8)
    args = p.parse_args()
    seed_everything(42)
    data = SyntheticRingDataset()
    train = torch.utils.data.Subset(data, range(1000))
    test = torch.utils.data.Subset(data, range(1000, len(data)))
    model = BioAlignV2(canonicalizer='continuous')
    cfg = TrainConfig(epochs=args.epochs, batch_size=128, seed=42, lambda_shift=1.0, lambda_consistency=0.2)
    fit(model, DataLoader(train, batch_size=128, shuffle=True), cfg, device='cpu')
    model.eval()
    preds, ys, pshift, tshift = [], [], [], []
    with torch.no_grad():
        for batch in DataLoader(test, batch_size=128):
            logits, aux = model(batch['x'], return_aux=True)
            preds.append(logits.argmax(1).numpy())
            ys.append(batch['gesture'].numpy())
            pshift.append(aux['expected_shift_channels'].numpy())
            tshift.append(data.shifts[np.asarray(batch['index'])])
    pred = np.concatenate(preds); y = np.concatenate(ys)
    ps = np.concatenate(pshift); ts = np.concatenate(tshift)
    report = classification_metrics(y, pred)
    report['shift_circular_mae_deg'] = circular_mae_deg(ps * 45.0, ts * 45.0)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    (out / 'synthetic_smoke_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
