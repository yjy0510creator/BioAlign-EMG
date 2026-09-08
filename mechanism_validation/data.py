from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


ALIASES = {
    'x': ('x', 'X', 'windows', 'signals', 'data'),
    'gesture': ('gesture', 'gestures', 'y', 'labels', 'gesture_id'),
    'position': ('position', 'positions', 'p', 'position_id'),
    'repetition': ('repetition', 'repetitions', 'rep', 'r', 'repetition_id'),
    'trial': ('trial', 'trials', 'trial_id'),
    'subject': ('subject', 'subjects', 'subject_id'),
    'angle_deg': ('angle_deg', 'angles_deg', 'measured_angle_deg', 'shift_angle_deg'),
}


def _pick(npz: Mapping[str, np.ndarray], key: str, required: bool = True):
    for candidate in ALIASES[key]:
        if candidate in npz:
            return np.asarray(npz[candidate])
    if required:
        raise KeyError(f'Missing {key}; accepted keys: {ALIASES[key]}; found: {list(npz)}')
    return None


@dataclass
class StandardWindows:
    x: np.ndarray
    gesture: np.ndarray
    position: np.ndarray
    repetition: np.ndarray
    trial: np.ndarray
    subject: np.ndarray
    angle_deg: Optional[np.ndarray] = None

    def validate(self) -> 'StandardWindows':
        self.x = np.asarray(self.x, dtype=np.float32)
        if self.x.ndim != 3:
            raise ValueError(f'x must be [N,C,T], got {self.x.shape}')
        n = self.x.shape[0]
        for name in ('gesture', 'position', 'repetition', 'trial', 'subject'):
            arr = np.asarray(getattr(self, name))
            if len(arr) != n:
                raise ValueError(f'{name} length {len(arr)} != N {n}')
            setattr(self, name, arr)
        if self.angle_deg is not None and len(self.angle_deg) != n:
            raise ValueError('angle_deg length mismatch')
        return self

    def subset(self, mask: np.ndarray) -> 'StandardWindows':
        return StandardWindows(
            self.x[mask], self.gesture[mask], self.position[mask], self.repetition[mask],
            self.trial[mask], self.subject[mask], None if self.angle_deg is None else self.angle_deg[mask]
        ).validate()


def load_standard_npz(path: str | Path) -> StandardWindows:
    path = Path(path)
    with np.load(path, allow_pickle=True) as z:
        x = _pick(z, 'x')
        gesture = _pick(z, 'gesture')
        position = _pick(z, 'position')
        repetition = _pick(z, 'repetition')
        trial = _pick(z, 'trial', required=False)
        if trial is None:
            trial = np.arange(len(x))
        subject = _pick(z, 'subject', required=False)
        if subject is None:
            subject = np.repeat(path.stem, len(x))
        angle = _pick(z, 'angle_deg', required=False)
    return StandardWindows(x, gesture, position, repetition, trial, subject, angle).validate()


def save_standard_npz(data: StandardWindows, path: str | Path) -> None:
    data.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'x': data.x.astype(np.float32),
        'gesture': data.gesture,
        'position': data.position,
        'repetition': data.repetition,
        'trial': data.trial,
        'subject': data.subject,
    }
    if data.angle_deg is not None:
        payload['angle_deg'] = data.angle_deg
    np.savez_compressed(path, **payload)


def protocol_masks(data: StandardWindows) -> Dict[str, np.ndarray]:
    p = np.asarray(data.position).astype(str)
    r = np.asarray(data.repetition).astype(str)
    p0 = np.isin(p, ['0', 'p0', 'P0'])
    train = p0 & np.isin(r, ['0', '1', 'r0', 'r1', 'R0', 'R1'])
    ideal = p0 & np.isin(r, ['2', 'r2', 'R2'])
    shift = ~p0
    if not train.any() or not shift.any():
        raise ValueError('Could not form p0 r0+r1 train and p1+ shift masks; check metadata encoding')
    return {'train': train, 'ideal': ideal, 'shift': shift}


def normalize_from_train(data: StandardWindows, train_mask: np.ndarray) -> Tuple[StandardWindows, np.ndarray, np.ndarray]:
    x_train = data.x[train_mask]
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x = (data.x - mean) / std
    out = StandardWindows(x, data.gesture, data.position, data.repetition, data.trial, data.subject, data.angle_deg).validate()
    return out, mean.squeeze(), std.squeeze()


class WindowDataset(Dataset):
    def __init__(self, data: StandardWindows, indices: Sequence[int]):
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = self.indices[i]
        item = {
            'x': torch.from_numpy(self.data.x[j]),
            'gesture': torch.as_tensor(int(self.data.gesture[j]), dtype=torch.long),
            'position': str(self.data.position[j]),
            'repetition': str(self.data.repetition[j]),
            'trial': str(self.data.trial[j]),
            'index': int(j),
        }
        if self.data.angle_deg is not None:
            item['angle_deg'] = torch.as_tensor(float(self.data.angle_deg[j]), dtype=torch.float32)
        return item
