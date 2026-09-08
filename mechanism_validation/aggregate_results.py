from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def parse_subjects(spec: str) -> list[str]:
    if '-' in spec and ',' not in spec:
        a, b = spec.split('-', 1)
        return [f'h{i}' for i in range(int(a[1:]), int(b[1:]) + 1)]
    return [s.strip() for s in spec.split(',') if s.strip()]


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj.tolist()


def bootstrap_ci(x, seed=0, n_boot=5000):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    vals = np.asarray([rng.choice(x, len(x), replace=True).mean() for _ in range(n_boot)])
    return tuple(np.percentile(vals, [2.5, 97.5]).astype(float))


def summarize(df: pd.DataFrame, reference: str, split: str, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df[(df['split'].astype(str) == split) & (df['status'].fillna('OK').astype(str).str.startswith('OK'))].copy()
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()
    # For seedless classical rows, average by subject/model only.  For neural rows,
    # first average seeds within subject to keep subject as inferential unit.
    group_cols = ['subject', 'model']
    subj = d.groupby(group_cols, as_index=False)[metric].mean()
    rows=[]
    for model, g in subj.groupby('model'):
        vals = g[metric].to_numpy(float)
        lo, hi = bootstrap_ci(vals)
        rows.append({'model': model, 'split': split, 'metric': metric, 'n_subjects': int(len(vals)), 'mean': float(np.nanmean(vals)), 'ci95_low': lo, 'ci95_high': hi})
    summary = pd.DataFrame(rows).sort_values('mean', ascending=False)

    ref = subj[subj['model'] == reference].set_index('subject')[metric]
    comp_rows=[]
    pvals=[]
    for model in sorted(set(subj['model']) - {reference}):
        other = subj[subj['model'] == model].set_index('subject')[metric]
        common = sorted(set(ref.index) & set(other.index))
        if not common:
            continue
        a = ref.loc[common].to_numpy(float)
        b = other.loc[common].to_numpy(float)
        valid = np.isfinite(a) & np.isfinite(b)
        a, b = a[valid], b[valid]
        if len(a) == 0:
            continue
        diff = a - b
        try:
            stat, p = wilcoxon(a, b, alternative='two-sided', zero_method='wilcox')
        except ValueError:
            stat, p = np.nan, 1.0
        pvals.append(float(p))
        comp_rows.append({'comparison': f'{reference} vs {model}', 'n_subjects': int(len(a)), 'reference_mean': float(a.mean()), 'comparator_mean': float(b.mean()), 'mean_paired_gain': float(diff.mean()), 'reference_wins': int(np.sum(diff > 0)), 'wilcoxon_statistic': float(stat) if np.isfinite(stat) else np.nan, 'p_two_sided': float(p)})
    comps = pd.DataFrame(comp_rows)
    if not comps.empty:
        comps['p_holm'] = holm(pvals)
    return summary, comps


def run_subjects(args):
    subjects = parse_subjects(args.subjects)
    code_root = Path(args.code_root)
    data_dir = Path(args.data_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    for subject in subjects:
        data_path = data_dir / f'{subject}.npz'
        if not data_path.exists():
            raise FileNotFoundError(f'Missing standardized subject NPZ: {data_path}. Run build_senic_npz first.')
        out = output_root / subject
        cmd = [
            py, '-m', 'mechanism_validation.run_experiment',
            '--data', str(data_path),
            '--output', str(out),
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size),
            '--seeds', *[str(s) for s in args.seeds],
            '--models', *args.models,
            '--correction-baselines', *args.correction_baselines,
        ]
        if args.classical:
            cmd.append('--classical')
        if args.force:
            cmd.append('--force')
        if args.device:
            cmd.extend(['--device', args.device])
        print('RUN', ' '.join(cmd))
        subprocess.run(cmd, cwd=str(code_root), check=True)


def collect(output_root: Path) -> pd.DataFrame:
    frames=[]
    for p in sorted(output_root.glob('h*/metrics_all.csv')):
        df=pd.read_csv(p)
        df.insert(0,'subject',p.parent.name)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f'No h*/metrics_all.csv under {output_root}')
    return pd.concat(frames, ignore_index=True)


def main():
    p=argparse.ArgumentParser(description='Run/collect 30-subject BioAlign V2.1 results and paired statistics')
    p.add_argument('--mode', choices=['run','collect','run_collect'], default='collect')
    p.add_argument('--code-root', default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--subjects', default='h0-h29')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--seeds', type=int, nargs='+', default=[42,2026,3407])
    p.add_argument('--models', nargs='+', default=['all'])
    p.add_argument('--correction-baselines', nargs='*', default=['all'])
    p.add_argument('--classical', action='store_true')
    p.add_argument('--force', action='store_true')
    p.add_argument('--device', default=None)
    p.add_argument('--reference', default='continuous')
    p.add_argument('--metric', default='macro_f1')
    args=p.parse_args()
    if args.mode in ('run','run_collect'):
        run_subjects(args)
    if args.mode in ('collect','run_collect'):
        root=Path(args.output_root)
        df=collect(root)
        root.mkdir(parents=True,exist_ok=True)
        df.to_csv(root/'all_subject_metrics_long.csv', index=False, encoding='utf-8-sig')
        for split in ['ideal','shift']:
            summary, comps=summarize(df,args.reference,split,args.metric)
            summary.to_csv(root/f'summary_{split}_{args.metric}.csv', index=False, encoding='utf-8-sig')
            comps.to_csv(root/f'paired_{split}_{args.metric}.csv', index=False, encoding='utf-8-sig')
        (root/'aggregate_config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
        print(f'Wrote aggregate results under {root}')

if __name__=='__main__':
    main()
