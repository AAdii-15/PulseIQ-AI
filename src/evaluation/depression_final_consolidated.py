"""
Consolidated, single-source-of-truth computation for every depression
number Table I needs: AUROC [95% CI], BACC, F1, Sens, Spec, Brier, ECE.

Built after discovering Table I's depression row (0.626 AUROC, 0.611
BACC, 0.548 F1, 0.832 Sens, 0.225 Brier, 0.114 ECE) does not match any
currently-existing results file -- the closest sources
(bootstrap_metric_cis.csv, calibration_bacc_f1.csv,
depression_optimal_threshold_results.csv) all agree with EACH OTHER
(AUROC~0.634, Sens~0.917) but not with Table I, suggesting Table I's
numbers trace to an earlier, now-superseded run. Rather than continue
chasing an untraceable historical number, this script is the new
canonical source: one seeded, reproducible fit, all six Table I values
computed from it, nothing borrowed from any other file.

Conventions matched to the already-verified PD/COVID rows in Table I:
  - AUROC: single-fit point estimate, with 95% CI from bootstrap
    resampling of the fixed dev-set predictions (NOT a bootstrap mean).
  - BACC / F1 / Sens / Spec: point estimates at the Youden-J-optimal
    threshold from that same single fit.
  - Brier / ECE: bootstrap MEAN across 2,000 resamples (matches the
    methodology already confirmed to reproduce PD/COVID exactly).
mutual_info_classif is seeded via functools.partial for reproducibility
(the unseeded version was the source of earlier run-to-run drift).
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
    bacc = (sens + spec) / 2
    f1 = f1_score(y, pred, zero_division=0)
    return sens, spec, bacc, f1

print('Loading DAIC-WOZ COVAREP features, official AVEC 2017 split...')
dd    = pd.read_csv(BASE / 'data/features/daic_woz_covarep_allframes.csv')
trn_s = pd.read_csv(BASE / 'data/raw/daic_woz/train_split_Depression_AVEC2017.csv').rename(
            columns={'Participant_ID': 'participant_id'})
dev_s = pd.read_csv(BASE / 'data/raw/daic_woz/dev_split_Depression_AVEC2017.csv').rename(
            columns={'Participant_ID': 'participant_id'})
fc  = [c for c in dd.columns if c.endswith('_mean') or c.endswith('_std')]
tr  = dd[dd.participant_id.isin(trn_s.participant_id)]
dv  = dd[dd.participant_id.isin(dev_s.participant_id)]
print(f'Train: {len(tr)} | Dev: {len(dv)} | Dev positive: {dv.PHQ8_Binary.sum()}')

print('Fitting seeded pipeline: median impute -> scale -> SelectKBest(mutual_info, k=30, seeded) -> RBF SVM...')
dep_pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc', StandardScaler()),
                      ('sel', SelectKBest(partial(mutual_info_classif, random_state=42), k=30)),
                      ('svm', SVC(kernel='rbf', probability=True, random_state=42))])
dep_pipe.fit(tr[fc].values, tr.PHQ8_Binary.values)
y_prob = dep_pipe.predict_proba(dv[fc].values)[:, 1]
y_true = dv.PHQ8_Binary.values

# ── AUROC: single-fit point estimate ───────────────────────────────────────────
auroc = roc_auc_score(y_true, y_prob)

# ── Youden-J optimal threshold from this same fit ──────────────────────────────
fpr, tpr, thr = roc_curve(y_true, y_prob)
j_stat = tpr - fpr
t_star = thr[np.argmax(j_stat)]
sens, spec, bacc, f1 = metrics_at_threshold(y_true, y_prob, t_star)

# ── AUROC 95% CI: bootstrap the fixed (y_true, y_prob), not a mean ────────────
rng = np.random.default_rng(42)
auroc_boot = []
for _ in range(2000):
    idx = rng.integers(0, len(y_true), len(y_true))
    if len(np.unique(y_true[idx])) < 2:
        continue
    auroc_boot.append(roc_auc_score(y_true[idx], y_prob[idx]))
auroc_lo, auroc_hi = np.percentile(auroc_boot, [2.5, 97.5])

# ── Brier / ECE: bootstrap MEAN (matches verified PD/COVID methodology) ───────
rng2 = np.random.default_rng(42)
briers, eces = [], []
for _ in range(2000):
    idx = rng2.integers(0, len(y_true), len(y_true))
    if len(np.unique(y_true[idx])) < 2:
        continue
    briers.append(brier_score_loss(y_true[idx], y_prob[idx]))
    eces.append(ece(y_true[idx], y_prob[idx]))
brier_mean, ece_mean = np.mean(briers), np.mean(eces)

print('\n' + '=' * 70)
print(' CONSOLIDATED DEPRESSION ROW FOR TABLE I (single canonical source)')
print('=' * 70)
print(f'  AUROC [95% CI]  : {auroc:.3f} [{auroc_lo:.3f}--{auroc_hi:.3f}]')
print(f'  BACC            : {bacc:.3f}')
print(f'  F1              : {f1:.3f}')
print(f'  Sens.           : {sens:.3f}')
print(f'  Spec.           : {spec:.3f}')
print(f'  Brier           : {brier_mean:.3f}')
print(f'  ECE             : {ece_mean:.3f}')
print(f'  Youden-J threshold t*: {t_star:.4f}')
print('=' * 70)

pd.DataFrame([{
    'condition': 'depression', 'model': 'COVAREP+SelectKBest(30,seeded)+RBF-SVM',
    'protocol': 'AVEC2017 Official Dev Set', 'N': len(y_true),
    'auroc': round(auroc, 4), 'auroc_ci_lo': round(auroc_lo, 4), 'auroc_ci_hi': round(auroc_hi, 4),
    'bacc': round(bacc, 4), 'f1': round(f1, 4), 'sens': round(sens, 4), 'spec': round(spec, 4),
    'brier': round(brier_mean, 4), 'ece': round(ece_mean, 4),
    'youden_threshold': round(t_star, 4)
}]).to_csv(RESULTS / 'depression_final_consolidated.csv', index=False)
print(f"\nSaved -> {RESULTS / 'depression_final_consolidated.csv'}")
print('This file is now the single source of truth for Table I\'s depression row.')
