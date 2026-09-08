from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .classical_baselines import fit_evaluate_classical
from .correction_baselines import (
    XCorrTemplateCorrector,
    angle_to_continuous_channels,
    angle_to_integer_bins,
    apply_continuous_inverse,
    apply_integer_inverse,
)
from .data import StandardWindows, WindowDataset, load_standard_npz, normalize_from_train, protocol_masks
from .evaluation import evaluate_windows
from .models import BioAlignV2, CircularConvTCN, TinyTCN
from .training import TrainConfig, checkpoint_matches, fit, save_checkpoint


CANONICALIZER_VARIANTS = ['none', 'uniform', 'uniform_safe', 'random', 'hard_ste', 'soft', 'continuous']
DEFAULT_MODELS = [
    'tcn_plain',
    'tcn_aug',
    'circular_tcn_plain',
    'circular_tcn_aug',
    'none',
    'none_shift_aux',
    'uniform',
    'random',
    'hard_ste',
    'soft_cls_only',
    'soft',
    'continuous_cls_only',
    'continuous',
]
DEFAULT_CORRECTIONS = ['xcorr_tcn', 'oracle_integer_tcn', 'oracle_continuous_tcn', 'wrong_integer_tcn', 'wrong_continuous_tcn']


def expand_models(models: Iterable[str]) -> list[str]:
    out = []
    for m in models:
        if m == 'all':
            out.extend(DEFAULT_MODELS)
        else:
            out.append(m)
    seen = []
    for m in out:
        if m not in seen:
            seen.append(m)
    return seen


def make_model(name: str, n_channels: int, n_classes: int):
    alias = {
        'tcn': 'tcn_plain',
        'circular_tcn': 'circular_tcn_plain',
        'none_cls': 'none_plain',
    }.get(name, name)
    if alias in ('tcn_plain', 'tcn_aug'):
        return TinyTCN(n_channels, n_classes)
    if alias in ('circular_tcn_plain', 'circular_tcn_aug'):
        return CircularConvTCN(n_channels, n_classes)
    if alias == 'none_plain':
        return BioAlignV2(n_channels, n_classes, canonicalizer='none')
    if alias == 'none':
        return BioAlignV2(n_channels, n_classes, canonicalizer='none')
    if alias == 'none_shift_aux':
        return BioAlignV2(n_channels, n_classes, canonicalizer='none')
    if alias in ('soft_cls_only', 'continuous_cls_only'):
        return BioAlignV2(n_channels, n_classes, canonicalizer=alias.replace('_cls_only', ''))
    if alias in CANONICALIZER_VARIANTS:
        return BioAlignV2(n_channels, n_classes, canonicalizer=alias)
    raise ValueError(f'Unknown model: {name}')


def config_for_model(name: str, args) -> TrainConfig:
    alias = {
        'tcn': 'tcn_plain',
        'circular_tcn': 'circular_tcn_plain',
        'none_cls': 'none_plain',
    }.get(name, name)
    # Baselines are deliberately separated so that augmentation, shift auxiliary
    # supervision, and true alignment are not confounded.
    if alias in ('tcn_plain', 'circular_tcn_plain', 'none_plain'):
        return TrainConfig(
            epochs=args.epochs, batch_size=args.batch_size, seed=args._seed,
            use_synthetic_shift=False, use_shift_loss=False, use_consistency_loss=False,
            lambda_shift=0.0, lambda_consistency=0.0,
        )
    if alias in ('tcn_aug', 'circular_tcn_aug', 'none', 'uniform', 'uniform_safe', 'random', 'soft_cls_only', 'continuous_cls_only'):
        return TrainConfig(
            epochs=args.epochs, batch_size=args.batch_size, seed=args._seed,
            use_synthetic_shift=True, use_shift_loss=False, use_consistency_loss=False,
            lambda_shift=0.0, lambda_consistency=0.0,
            max_synthetic_shift_channels=args.max_synthetic_shift_channels,
            continuous_augmentation=not args.integer_augmentation,
        )
    if alias == 'none_shift_aux':
        return TrainConfig(
            epochs=args.epochs, batch_size=args.batch_size, seed=args._seed,
            use_synthetic_shift=True, use_shift_loss=True, use_consistency_loss=False,
            lambda_shift=args.lambda_shift, lambda_consistency=0.0,
            max_synthetic_shift_channels=args.max_synthetic_shift_channels,
            continuous_augmentation=not args.integer_augmentation,
            shift_kappa=args.shift_kappa,
        )
    if alias in ('hard_ste', 'soft', 'continuous'):
        return TrainConfig(
            epochs=args.epochs, batch_size=args.batch_size, seed=args._seed,
            use_synthetic_shift=True, use_shift_loss=True, use_consistency_loss=True,
            lambda_shift=args.lambda_shift, lambda_consistency=args.lambda_consistency,
            max_synthetic_shift_channels=args.max_synthetic_shift_channels,
            continuous_augmentation=not args.integer_augmentation,
            shift_kappa=args.shift_kappa,
        )
    raise ValueError(alias)


def train_or_load(name: str, seed: int, args, data: StandardWindows, masks, out: Path):
    n_classes = len(np.unique(data.gesture))
    n_channels = data.x.shape[1]
    args._seed = seed
    cfg = config_for_model(name, args)
    model = make_model(name, n_channels, n_classes)
    train_idx = np.flatnonzero(masks['train'])
    loader = DataLoader(WindowDataset(data, train_idx), batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    ckpt = out / 'checkpoints' / f'{name}_seed{seed}_{cfg.digest()}.pt'
    latest = out / 'checkpoints' / f'{name}_seed{seed}.pt'
    if not (checkpoint_matches(ckpt, cfg) and not args.force):
        history = fit(model, loader, cfg, device=args.device)
        save_checkpoint(model, ckpt, cfg, {'model': name, 'history': history})
        save_checkpoint(model, latest, cfg, {'model': name, 'history': history, 'canonical_checkpoint': str(ckpt)})
    else:
        obj = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(obj['state_dict'])
    return model, cfg


def subset_with_override(data: StandardWindows, idx: np.ndarray, x_override: np.ndarray) -> StandardWindows:
    return StandardWindows(
        x_override.astype(np.float32),
        data.gesture[idx],
        data.position[idx],
        data.repetition[idx],
        data.trial[idx],
        data.subject[idx],
        None if data.angle_deg is None else data.angle_deg[idx],
    ).validate()


def evaluate_on_array(model, data: StandardWindows, idx: np.ndarray, x_override: np.ndarray, batch_size: int, device: str | None):
    temp = subset_with_override(data, idx, x_override)
    loader = DataLoader(WindowDataset(temp, np.arange(len(temp.x))), batch_size=batch_size, shuffle=False)
    return evaluate_windows(model, loader, device=device)


def run_correction_baselines(seed: int, args, data: StandardWindows, masks, out: Path) -> list[dict]:
    rows = []
    requested = [] if not args.correction_baselines else args.correction_baselines
    if 'all' in requested:
        requested = DEFAULT_CORRECTIONS
    if not requested:
        return rows

    # Correction baselines answer a different question: if the input is corrected
    # by an explicit rotation estimator/oracle, how much shift robustness can a
    # plain p0-trained classifier recover?
    base_name = 'tcn_plain'
    base_model, base_cfg = train_or_load(base_name, seed, args, data, masks, out)
    idx = np.flatnonzero(masks['shift'])
    x_eval = data.x[idx]

    for corr in requested:
        corr = corr.strip()
        try:
            if corr == 'xcorr_tcn':
                corrector = XCorrTemplateCorrector.fit(data.x[masks['train']], data.gesture[masks['train']])
                x_corr, shift_bins = corrector.correct(x_eval)
                extra = {'mean_estimated_shift_bins': float(np.mean(shift_bins))}
            elif corr in ('oracle_integer_tcn', 'oracle_continuous_tcn', 'wrong_integer_tcn', 'wrong_continuous_tcn'):
                if data.angle_deg is None or not np.isfinite(data.angle_deg[idx]).any():
                    rows.append({'model': corr, 'seed': seed, 'split': 'shift', 'status': 'SKIPPED_NO_ANGLE'})
                    continue
                angle = data.angle_deg[idx]
                if corr == 'oracle_integer_tcn':
                    bins = angle_to_integer_bins(angle, data.x.shape[1])
                    x_corr = apply_integer_inverse(x_eval, bins)
                    extra = {'angle_source': 'directed_measured_angle', 'correction': 'nearest_integer_inverse'}
                elif corr == 'oracle_continuous_tcn':
                    shift = angle_to_continuous_channels(angle, data.x.shape[1])
                    x_corr = apply_continuous_inverse(x_eval, shift)
                    extra = {'angle_source': 'directed_measured_angle', 'correction': 'continuous_inverse'}
                elif corr == 'wrong_integer_tcn':
                    bins = angle_to_integer_bins(-angle, data.x.shape[1])
                    x_corr = apply_integer_inverse(x_eval, bins)
                    extra = {'angle_source': 'directed_measured_angle', 'correction': 'wrong_direction_integer'}
                else:
                    shift = angle_to_continuous_channels(-angle, data.x.shape[1])
                    x_corr = apply_continuous_inverse(x_eval, shift)
                    extra = {'angle_source': 'directed_measured_angle', 'correction': 'wrong_direction_continuous'}
            else:
                rows.append({'model': corr, 'seed': seed, 'split': 'shift', 'status': 'UNKNOWN_CORRECTION'})
                continue
            result = evaluate_on_array(base_model, data, idx, x_corr, args.batch_size, args.device)
            row = {'model': corr, 'seed': seed, 'split': 'shift', 'status': 'OK', **result['trial_metrics'], **extra}
            rows.append(row)
            np.savez_compressed(
                out / f'predictions_{corr}_seed{seed}_shift.npz',
                **{k: v for k, v in result.items() if isinstance(v, np.ndarray)},
            )
        except Exception as exc:
            rows.append({'model': corr, 'seed': seed, 'split': 'shift', 'status': f'ERROR: {exc}'})
    return rows


def run(args):
    data = load_standard_npz(args.data)
    masks = protocol_masks(data)
    data, mean, std = normalize_from_train(data, masks['train'])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'checkpoints').mkdir(exist_ok=True)

    rows = []
    train_idx = np.flatnonzero(masks['train'])
    shift_idx = np.flatnonzero(masks['shift'])
    if args.classical:
        cres = fit_evaluate_classical(data, train_idx, shift_idx)
        for name, metrics in cres.items():
            rows.append({'model': name, 'seed': np.nan, 'split': 'shift', 'status': 'OK', **metrics})

    models = expand_models(args.models)
    for seed in args.seeds:
        for name in models:
            model, cfg = train_or_load(name, seed, args, data, masks, out)
            for split in ('ideal', 'shift'):
                idx = np.flatnonzero(masks[split])
                if len(idx) == 0:
                    continue
                ev_loader = DataLoader(WindowDataset(data, idx), batch_size=args.batch_size, shuffle=False)
                result = evaluate_windows(model, ev_loader, device=args.device)
                metrics = result['trial_metrics']
                rows.append({'model': name, 'seed': seed, 'split': split, 'status': 'OK', **metrics, 'config_digest': cfg.digest()})
                np.savez_compressed(
                    out / f'predictions_{name}_seed{seed}_{split}.npz',
                    **{k: v for k, v in result.items() if isinstance(v, np.ndarray)},
                )
        rows.extend(run_correction_baselines(seed, args, data, masks, out))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / 'metrics_all.csv', index=False, encoding='utf-8-sig')
    protocol = {
        'data': str(args.data),
        'n_windows': int(len(data.x)),
        'n_trials': int(len(np.unique(data.trial))),
        'train_trials': int(len(np.unique(data.trial[masks['train']]))),
        'ideal_trials': int(len(np.unique(data.trial[masks['ideal']]))),
        'shift_trials': int(len(np.unique(data.trial[masks['shift']]))),
        'has_angle_deg': bool(data.angle_deg is not None and np.isfinite(data.angle_deg).any()),
        'models': models,
        'correction_baselines': args.correction_baselines,
        'normalization': 'channel-wise z-score fitted only on p0 r0+r1 windows',
    }
    (out / 'protocol_audit.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    (out / 'run_config.json').write_text(json.dumps({k:v for k,v in vars(args).items() if not k.startswith('_')}, indent=2), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description='Run BioAlign mechanism-validation experiment on standardized SeNic windows')
    p.add_argument('--data', required=True, help='Subject-level standardized NPZ with x/gesture/position/repetition/trial and optional angle_deg')
    p.add_argument('--output', required=True)
    p.add_argument('--models', nargs='+', default=DEFAULT_MODELS, help='Use "all" for the full default neural set')
    p.add_argument('--correction-baselines', nargs='*', default=[], help='Use "all" or names: ' + ', '.join(DEFAULT_CORRECTIONS))
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 2026, 3407])
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--device', default=None)
    p.add_argument('--classical', action='store_true')
    p.add_argument('--force', action='store_true')
    p.add_argument('--lambda-shift', type=float, default=0.5)
    p.add_argument('--lambda-consistency', type=float, default=0.2)
    p.add_argument('--max-synthetic-shift-channels', type=float, default=4.0)
    p.add_argument('--integer-augmentation', action='store_true', help='Use integer synthetic rolls instead of continuous Fourier shifts')
    p.add_argument('--shift-kappa', type=float, default=8.0)
    run(p.parse_args())


if __name__ == '__main__':
    main()
