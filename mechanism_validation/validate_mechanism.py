from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .cyclic_ops import continuous_circular_shift
from .data import WindowDataset, load_standard_npz, normalize_from_train, protocol_masks
from .evaluation import evaluate_windows
from .figures import save_entropy_performance, save_gesture_heatmap
from .mechanism_analyses import gesture_channel_summary, latent_disentanglement, pre_post_pattern_preservation, real_angle_agreement, synthetic_roll_recovery
from .metrics import classification_metrics, fisher_ratio, safe_silhouette
from .models import BioAlignV2


def load_model(checkpoint: Path, variant: str, channels: int, classes: int, device):
    model=BioAlignV2(channels,classes,canonicalizer=variant)
    obj=torch.load(checkpoint,map_location='cpu')
    state=obj['state_dict'] if 'state_dict' in obj else obj
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def synthetic_probe(model, x: np.ndarray, y: np.ndarray, shifts: np.ndarray, device, batch_size=256):
    xs=[]; ys=[]; true=[]
    base=torch.as_tensor(x,dtype=torch.float32)
    for s in shifts:
        xs.append(continuous_circular_shift(base, torch.full((len(base),),float(s))).numpy())
        ys.append(y); true.append(np.full(len(base),float(s)))
    X=np.concatenate(xs); Y=np.concatenate(ys); T=np.concatenate(true)
    pred_g=[]; pred_s=[]; ent=[]
    for start in range(0,len(X),batch_size):
        xb=torch.from_numpy(X[start:start+batch_size]).to(device)
        logits,aux=model(xb,return_aux=True)
        pred_g.append(logits.argmax(1).cpu().numpy())
        pred_s.append(aux['expected_shift_channels'].cpu().numpy())
        ent.append(aux['entropy'].cpu().numpy())
    pg=np.concatenate(pred_g); ps=np.concatenate(pred_s); entropy=np.concatenate(ent)
    rep=synthetic_roll_recovery(ps,T,x.shape[1])
    rep.update({f'gesture_{k}':v for k,v in classification_metrics(Y,pg).items()})
    return rep, {'true_shift':T,'pred_shift':ps,'gesture':Y,'pred_gesture':pg,'entropy':entropy}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--variant',default='continuous')
    p.add_argument('--output',required=True)
    p.add_argument('--device',default=None)
    p.add_argument('--synthetic-step',type=float,default=0.5)
    args=p.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    data=load_standard_npz(args.data); masks=protocol_masks(data); data,_,_=normalize_from_train(data,masks['train'])
    dev=torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model=load_model(Path(args.checkpoint),args.variant,data.x.shape[1],len(np.unique(data.gesture)),dev)

    # Direct known-transform test uses held-out p0-r2 only.
    base_mask=masks['ideal'] if masks['ideal'].any() else masks['train']
    shifts=np.arange(0,data.x.shape[1],args.synthetic_step,dtype=float)
    srep,sraw=synthetic_probe(model,data.x[base_mask],data.gesture[base_mask],shifts,dev)
    np.savez_compressed(out/'synthetic_shift_probe.npz',**sraw)

    # Real shifted outputs.
    idx=np.flatnonzero(masks['shift'])
    ev=evaluate_windows(model,DataLoader(WindowDataset(data,idx),batch_size=256,shuffle=False),device=str(dev))
    report={'synthetic_shift_recovery':srep,'real_shift_classification':ev['trial_metrics']}
    if data.angle_deg is not None and np.isfinite(data.angle_deg[idx]).any() and 'expected_shift_channels' in ev:
        report['real_angle_agreement']=real_angle_agreement(ev['expected_shift_channels'],data.angle_deg[idx],data.x.shape[1])
    if 'shift_probs' in ev:
        report['latent_disentanglement']=latent_disentanglement(ev['shift_probs'],ev['gesture'],ev['position'], ev.get('trial'))
    if 'embedding' in ev:
        report['embedding_gesture_fisher']=fisher_ratio(ev['embedding'],ev['gesture'])
        report['embedding_gesture_silhouette']=safe_silhouette(ev['embedding'],ev['gesture'])
        report['embedding_position_silhouette']=safe_silhouette(ev['embedding'],ev['position'])
    if 'features_pre_pool' in ev and 'features_post_pool' in ev:
        report['pre_post_pattern_preservation']=pre_post_pattern_preservation(ev['features_pre_pool'],ev['features_post_pool'],ev['gesture'],ev['position'])
    (out/'mechanism_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

    # Raw gesture-channel analysis.
    gdf=gesture_channel_summary(data.x,data.gesture,data.position)
    gdf.to_csv(out/'gesture_channel_features.csv',index=False)
    for pos in np.unique(data.position):
        save_gesture_heatmap(gdf,'RMS',pos,out/f'gesture_RMS_{pos}.png')
    if 'entropy' in ev:
        save_entropy_performance(ev['entropy'],ev['pred']==ev['gesture'],out/'entropy_vs_window_correctness.png')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
