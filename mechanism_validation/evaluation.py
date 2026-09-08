from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .metrics import aggregate_trial_probabilities, classification_metrics


@torch.no_grad()
def evaluate_windows(model, loader: DataLoader, device: Optional[str] = None, store_features: bool = True) -> Dict[str, np.ndarray]:
    """Evaluate a neural model and aggregate overlapping windows to trial level."""
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(dev).eval()
    out = defaultdict(list)
    for batch in loader:
        x = batch['x'].to(dev)
        logits, aux = model(x, return_aux=True)
        out['probs'].append(F.softmax(logits, dim=-1).cpu().numpy())
        out['gesture'].append(batch['gesture'].numpy())
        for key in ('embedding', 'expected_shift_channels', 'expected_shift_radians', 'entropy', 'confidence', 'probs'):
            if key in aux:
                name = 'shift_probs' if key == 'probs' else key
                out[name].append(aux[key].detach().cpu().numpy())
        if store_features:
            # Full [B,C,D,T] features can become large; store compact pooled
            # descriptors for pattern-preservation analyses.
            for key, name in (('features_pre', 'features_pre_pool'), ('features_post', 'features_post_pool')):
                if key in aux:
                    fp = aux[key].detach().mean(dim=-1).flatten(1).cpu().numpy()
                    out[name].append(fp)
        if 'angle_deg' in batch:
            out['angle_deg'].append(batch['angle_deg'].numpy())
        out['trial'].extend(list(batch['trial']))
        out['position'].extend(list(batch['position']))
        out['repetition'].extend(list(batch['repetition']))
        out['index'].extend(np.asarray(batch['index']).tolist())
    result = {}
    for k, v in out.items():
        if k in ('trial', 'position', 'repetition', 'index'):
            result[k] = np.asarray(v)
        else:
            result[k] = np.concatenate(v, axis=0)
    result['pred'] = result['probs'].argmax(axis=1)
    result['window_metrics'] = classification_metrics(result['gesture'], result['pred'])
    tprob, ty, tids = aggregate_trial_probabilities(result['probs'], result['gesture'], result['trial'])
    result['trial_probs'] = tprob
    result['trial_gesture'] = ty
    result['trial_ids'] = tids
    result['trial_pred'] = tprob.argmax(axis=1)
    result['trial_metrics'] = classification_metrics(ty, result['trial_pred'])
    return result
