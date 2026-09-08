import numpy as np
import pandas as pd

from mechanism_validation.angle_utils import wrap_deg, circular_mean_deg
from mechanism_validation.classical_baselines import fit_evaluate_classical
from mechanism_validation.data import StandardWindows
from mechanism_validation.run_experiment import config_for_model


class Args:
    epochs=1; batch_size=8; _seed=42
    max_synthetic_shift_channels=4.0; integer_augmentation=False
    lambda_shift=0.5; lambda_consistency=0.2; shift_kappa=8.0


def test_angle_wrap_and_mean():
    assert wrap_deg(190) == -170
    assert abs(circular_mean_deg(np.array([179, -179]))) - 180 < 1e-6


def test_training_configs_do_not_confound_negative_controls():
    a=Args()
    plain=config_for_model('tcn_plain',a)
    assert not plain.use_synthetic_shift and not plain.use_shift_loss
    uniform=config_for_model('uniform',a)
    assert uniform.use_synthetic_shift and not uniform.use_shift_loss and not uniform.use_consistency_loss
    full=config_for_model('continuous',a)
    assert full.use_synthetic_shift and full.use_shift_loss and full.use_consistency_loss
    aux=config_for_model('none_shift_aux',a)
    assert aux.use_synthetic_shift and aux.use_shift_loss and not aux.use_consistency_loss


def test_classical_baselines_report_trial_metrics():
    rng=np.random.default_rng(1)
    n_trials=28; nwin=3
    xs=[]; ys=[]; trials=[]; pos=[]; reps=[]; subj=[]
    for t in range(n_trials):
        g=t%2
        for w in range(nwin):
            x=rng.normal(size=(8,50)).astype('float32')
            x[g,:]+=1.5
            xs.append(x); ys.append(g); trials.append(f't{t}'); pos.append('p0' if t<14 else 'p1'); reps.append('r0' if t<7 else ('r1' if t<14 else 'r2')); subj.append('h0')
    data=StandardWindows(np.stack(xs),np.array(ys),np.array(pos),np.array(reps),np.array(trials),np.array(subj)).validate()
    train=np.arange(14*nwin); test=np.arange(14*nwin, n_trials*nwin)
    out=fit_evaluate_classical(data,train,test)
    assert 'TD-LDA' in out and 'trial_macro_f1' in out['TD-LDA']
    assert 'window_macro_f1' in out['TD-LDA']
