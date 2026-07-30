"""
FINAL depression Table I row: point estimates are the mean across the
10 seeds already confirmed in depression_seed_robustness.py; the AUROC
95% CI combines BOTH uncertainty sources (which seed's feature
selection, and which bootstrap resample of the N=35 dev set) into one
interval, rather than reusing a single seed's CI next to a multi-seed
point estimate -- those would otherwise describe two different things.
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

print('Loading DAIC-WOZ COVAREP features...')
dd    = pd.read_csv(BASE / 'data/features/daic_woz_covarep_allframes.csv')
trn_s = pd.read_csv(BASE / 'data/raw/daic_woz/train_split_Depression_AVEC2017.csv').rename(
            columns={'Participant_ID': 'participant_id'})
dev_s = pd.read_csv(BASE / 'data/raw/daic_woz/dev_split_Depression_AVEC2017.csv').rename(
            columns={'Participant_ID': 'participant_id'})
fc  = [c for c in dd.columns if c.endswith('_mean') or c.endswith('_std')]
tr  = dd[dd.participant_id.isin(trn_s.participant_id)]
dv  = dd[dd.participant_id.isin(dev_s.participant_id)]
y_true = dv.PHQ8_Binary.values

print(f'Fitting all {len(SEEDS)} seeds...')
seed_probs = []
per_seed_rows = []
for seed in SEEDS:
    pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc', StandardScaler()),
                      ('sel', SelectKBest(partial(mutual_info_classif, random_state=seed), k=30)),
                      ('svm', SVC(kernel='rbf', probability=True, random_state=seed))])
    pipe.fit(tr[fc].values, tr.PHQ8_Binary.values)
    y_prob = pipe.predict_proba(dv[fc].values)[:, 1]
    seed_probs.append(y_prob)

    auroc = roc_auc_score(y_true, y_prob)
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    t_star = thr[np.argmax(tpr - fpr)]
    sens, spec, bacc, f1 = metrics_at_threshold(y_true, y_prob, t_star)
    per_seed_rows.append({'auroc': auroc, 'bacc': bacc, 'f1': f1, 'sens': sens,
                           'spec': spec, 'brier': brier_score_loss(y_true, y_prob),
                           'ece': ece(y_true, y_prob)})

seed_df = pd.DataFrame(per_seed_rows)

# ── Combined seed + bootstrap-sampling uncertainty for AUROC CI ───────────────
print('Computing combined seed+sampling bootstrap CI (2,000 draws)...')
rng = np.random.default_rng(42)
combined_auroc = []
for _ in range(2000):
    s = rng.integers(0, len(SEEDS))
    idx = rng.integers(0, len(y_true), len(y_true))
    if len(np.unique(y_true[idx])) < 2:
        continue
    combined_auroc.append(roc_auc_score(y_true[idx], seed_probs[s][idx]))
ci_lo, ci_hi = np.percentile(combined_auroc, [2.5, 97.5])

auroc_mean = seed_df.auroc.mean()
print('\n' + '=' * 70)
print(' FINAL DEPRESSION ROW FOR TABLE I')
print(' (point estimates = mean across 10 seeds; CI = combined seed +')
print('  bootstrap-sampling uncertainty, both drawn from the same 2,000 trials)')
print('=' * 70)
print(f'  AUROC [95% CI] : {auroc_mean:.3f} [{ci_lo:.3f}--{ci_hi:.3f}]')
print(f'  BACC           : {seed_df.bacc.mean():.3f}')
print(f'  F1             : {seed_df.f1.mean():.3f}')
print(f'  Sens.          : {seed_df.sens.mean():.3f}')
print(f'  Spec.          : {seed_df.spec.mean():.3f}')
print(f'  Brier          : {seed_df.brier.mean():.3f}')
print(f'  ECE            : {seed_df.ece.mean():.3f}')
print('=' * 70)
print(f'\nFor comparison, seed-to-seed AUROC range alone was '
      f'[{seed_df.auroc.min():.3f}, {seed_df.auroc.max():.3f}] '
      f'(std={seed_df.auroc.std():.4f})')
print(f'vs. PD (existing, already in paper): 0.799 +/- 0.007')
print(f'vs. COVID (existing, already in paper): 0.757 +/- 0.001')

seed_df.to_csv(RESULTS / 'depression_seed_robustness_final.csv', index=False)
pd.DataFrame([{
    'condition': 'depression', 'N': len(y_true),
    'auroc_mean': round(auroc_mean, 4), 'auroc_ci_lo': round(ci_lo, 4), 'auroc_ci_hi': round(ci_hi, 4),
    'auroc_seed_std': round(seed_df.auroc.std(), 4),
    'bacc': round(seed_df.bacc.mean(), 4), 'f1': round(seed_df.f1.mean(), 4),
    'sens': round(seed_df.sens.mean(), 4), 'spec': round(seed_df.spec.mean(), 4),
    'brier': round(seed_df.brier.mean(), 4), 'ece': round(seed_df.ece.mean(), 4),
}]).to_csv(RESULTS / 'depression_final_table1_row.csv', index=False)
print(f"\nSaved -> depression_seed_robustness_final.csv and depression_final_table1_row.csv")
