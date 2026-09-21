"""
FINAL corrected COVID-19 Table I row: participant-grouped 5-fold CV
(seed=42, matching this paper's canonical-seed convention throughout),
with a proper PARTICIPANT-LEVEL bootstrap CI -- resampling whole
participants together, not individual recordings, since recordings
from the same person are not independent observations.

Replaces the old recording-level-CV row, which is now understood to
have the same subject-level leakage problem this paper's own Section
5.1 criticizes in the PD literature.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score, confusion_matrix, roc_curve

BASE = Path.home() / 'Desktop/PULSE_IQ_AI'
RESULTS = BASE / 'results/metrics'
df = pd.read_csv(BASE / 'data/raw/coswara/voice_dataset_labeled_full.csv')

feat_cols = ['pitch', 'spectral_centroid', 'zcr', 'jitter', 'shimmer', 'hnr'] + \
            [f'mfcc_{i}' for i in range(1, 14)]
X = df[feat_cols].values
y = df['label'].values
groups = df['user_id'].values

def make_rf(seed=42):
    return Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc', StandardScaler()),
                      ('clf', RandomForestClassifier(n_estimators=500, random_state=seed,
                                                      class_weight='balanced', n_jobs=-1))])

def ece(y, p, nb=10):
    b = np.linspace(0, 1, nb + 1)
    e, n = 0.0, len(y)
    for i in range(nb):
        m = (p >= b[i]) & (p < b[i + 1])
        if m.sum():
            e += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return e

print('Fitting participant-grouped 5-fold CV (seed=42, canonical)...')
y_prob = cross_val_predict(make_rf(seed=42), X, y,
                            cv=StratifiedGroupKFold(5, shuffle=True, random_state=42),
                            groups=groups, method='predict_proba')[:, 1]

auroc = roc_auc_score(y, y_prob)
fpr, tpr, thr = roc_curve(y, y_prob)
t_star = thr[np.argmax(tpr - fpr)]
pred = (y_prob >= t_star).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
sens = tp / (tp + fn)
spec = tn / (tn + fp)
bacc = (sens + spec) / 2
f1 = f1_score(y, pred)
bs = brier_score_loss(y, y_prob)
e = ece(y, y_prob)

print(f'Point estimate: AUROC={auroc:.4f} BACC={bacc:.4f} F1={f1:.4f} Sens={sens:.4f} Brier={bs:.4f} ECE={e:.4f}')

print('\nComputing PARTICIPANT-LEVEL bootstrap CI (2000 resamples, whole participants together)...')
unique_participants = np.unique(groups)
rng = np.random.default_rng(42)
boot_auroc = []
for _ in range(2000):
    boot_p = rng.choice(unique_participants, size=len(unique_participants), replace=True)
    idx = np.concatenate([np.where(groups == p)[0] for p in boot_p])
    y_b, p_b = y[idx], y_prob[idx]
    if len(np.unique(y_b)) < 2:
        continue
    boot_auroc.append(roc_auc_score(y_b, p_b))
ci_lo, ci_hi = np.percentile(boot_auroc, [2.5, 97.5])

print('\n' + '=' * 70)
print(' FINAL COVID-19 ROW FOR TABLE I (participant-grouped, corrected)')
print('=' * 70)
print(f'  AUROC [95% CI] : {auroc:.3f} [{ci_lo:.3f}--{ci_hi:.3f}]  (participant-level bootstrap)')
print(f'  BACC           : {bacc:.3f}')
print(f'  F1             : {f1:.3f}')
print(f'  Sens.          : {sens:.3f}')
print(f'  Brier          : {bs:.3f}')
print(f'  ECE            : {e:.3f}')
print(f'  Youden-J threshold: {t_star:.4f}')
print('=' * 70)
print(f'\nFor comparison, OLD (recording-level, leaked) values were:')
print(f'  AUROC 0.758 [0.744--0.771], BACC 0.697, F1 0.667, Sens 0.631, Brier 0.202, ECE 0.066')

pd.DataFrame([{
    'condition': 'respiratory_participant_grouped', 'N': len(y), 'N_participants': len(unique_participants),
    'auroc': round(auroc, 4), 'auroc_ci_lo': round(ci_lo, 4), 'auroc_ci_hi': round(ci_hi, 4),
    'bacc': round(bacc, 4), 'f1': round(f1, 4), 'sens': round(sens, 4),
    'brier': round(bs, 4), 'ece': round(e, 4), 'youden_threshold': round(t_star, 4)
}]).to_csv(RESULTS / 'covid_final_table1_row.csv', index=False)
print(f"\nSaved -> results/metrics/covid_final_table1_row.csv")
