"""
Depression seed-robustness check, matching the same convention already
used for PD/COVID in remaining_fixes.py's fix_d_rf_variance() -- same
10-seed list, same one-point-estimate-per-seed approach, no inner
bootstrap layer (that's a separate, already-verified check).

Built because depression_final_consolidated.py (seed=42 only) gave
AUROC=0.609, which does not match the 0.634 that THREE other existing
result files independently agree on -- suggesting SelectKBest's
feature-selection randomness moves depression AUROC by a real amount at
N_train=107, unlike PD/COVID where seed choice barely matters (already
verified: PD 0.799+/-0.007, COVID 0.757+/-0.001). This script checks
whether that's actually true, using real seed variation rather than
assumption.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, f1_score, confusion_matrix

BASE = Path.home() / 'Desktop/PULSE_IQ_AI'
RESULTS = BASE / 'results/metrics'
SEEDS = [42, 7, 123, 256, 512, 999, 1337, 2024, 31, 88]

def ece(y, p, nb=10):
    b = np.linspace(0, 1, nb + 1)
    e, n = 0.0, len(y)
    for i in range(nb):
        m = (p >= b[i]) & (p < b[i + 1])
        if m.sum():
            e += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return e

def metrics_at_threshold(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return sens, spec, (sens + spec) / 2, f1_score(y, pred, zero_division=0)

print('Loading DAIC-WOZ COVAREP features, official AVEC 2017 split...')
dd    = pd.read_csv(BASE / 'data/features/daic_woz_covarep_allframes.csv')
trn_s = pd.read_csv(BASE / 'data/raw/daic_woz/train_split_Depression_AVEC2017.csv').rename(
            columns={'Participant_ID': 'participant_id'})
dev_s = pd.read_csv(BASE / 'data/raw/daic_woz/dev_split_Depression_AVEC2017.csv').rename(
            columns={'Participant_ID': 'participant_id'})
fc  = [c for c in dd.columns if c.endswith('_mean') or c.endswith('_std')]
tr  = dd[dd.participant_id.isin(trn_s.participant_id)]
dv  = dd[dd.participant_id.isin(dev_s.participant_id)]
y_true = dv.PHQ8_Binary.values

rows = []
print(f'\nRunning {len(SEEDS)} seeds...')
for seed in SEEDS:
    pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc', StandardScaler()),
                      ('sel', SelectKBest(partial(mutual_info_classif, random_state=seed), k=30)),
                      ('svm', SVC(kernel='rbf', probability=True, random_state=seed))])
    pipe.fit(tr[fc].values, tr.PHQ8_Binary.values)
    y_prob = pipe.predict_proba(dv[fc].values)[:, 1]

    auroc = roc_auc_score(y_true, y_prob)
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    t_star = thr[np.argmax(tpr - fpr)]
    sens, spec, bacc, f1 = metrics_at_threshold(y_true, y_prob, t_star)
    bs = brier_score_loss(y_true, y_prob)
    e = ece(y_true, y_prob)

    rows.append({'seed': seed, 'auroc': auroc, 'bacc': bacc, 'f1': f1,
                 'sens': sens, 'spec': spec, 'brier': bs, 'ece': e, 'threshold': t_star})
    print(f'  seed={seed:5d}  AUROC={auroc:.4f}  BACC={bacc:.4f}  F1={f1:.4f}  '
          f'Sens={sens:.4f}  Spec={spec:.4f}')

df = pd.DataFrame(rows)
df.to_csv(RESULTS / 'depression_seed_robustness.csv', index=False)

print('\n' + '=' * 70)
print(' SEED-ROBUSTNESS SUMMARY (mean +/- std across 10 seeds)')
print('=' * 70)
for col in ['auroc', 'bacc', 'f1', 'sens', 'spec', 'brier', 'ece']:
    print(f'  {col:8s}: {df[col].mean():.4f} +/- {df[col].std():.4f}   '
          f'[min={df[col].min():.4f}, max={df[col].max():.4f}]')
print('=' * 70)
print(f"\nSaved -> {RESULTS / 'depression_seed_robustness.csv'}")
print('\nCompare this spread to the already-reported PD (0.799+/-0.007) and')
print("COVID (0.757+/-0.001) seed robustness -- if depression's std is much")
print('larger, that confirms feature-selection instability at N_train=107')
print('is real and needs to be reported explicitly, not averaged away silently.')
