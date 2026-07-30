"""
Tree-ensemble comparison: Random Forest vs XGBoost vs LightGBM.

Motivation: the manuscript justifies RF on TreeSHAP compatibility, but
XGBoost and LightGBM are equally TreeSHAP-compatible (shap.TreeExplainer
supports all three natively), so that justification alone doesn't rule
them out as classifiers. This script trains all three under the exact
same protocols already used for the primary results -- LOSO for PD
(6-nonlinear-feature set, matching the primary reported result), 5-fold
stratified CV for COVID-19 (19-feature set) -- and reports AUROC with
95% bootstrap CIs and DeLong's test against the existing RF result, so
a reviewer's "why not XGBoost/LightGBM" question has a direct answer
in the paper instead of just a design-choice justification.

Auto-installs xgboost/lightgbm if not already present, matching the
try/except pattern already used elsewhere in this codebase (e.g. for
shap in exp_final_polish.py).
"""
import sys
import subprocess
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

for pkg in ['xgboost', 'lightgbm']:
    try:
        __import__(pkg)
    except ImportError:
        print(f'Installing {pkg}...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                                pkg, '--quiet', '--break-system-packages'])

import xgboost as xgb
import lightgbm as lgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (LeaveOneGroupOut, StratifiedKFold,
                                      cross_val_predict)
from sklearn.metrics import roc_auc_score

BASE = Path.home() / 'Desktop/PULSE_IQ_AI'
sys.path.insert(0, str(BASE / 'src'))
from feature_extraction.data_loaders import load_parkinsons, load_respiratory
RESULTS = BASE / 'results/metrics'

NL6 = ['RPDE', 'DFA', 'PPE', 'spread1', 'spread2', 'D2']


def bootstrap_auroc_ci(y_true, y_prob, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return round(np.mean(aucs), 4), round(lo, 4), round(hi, 4)


def delong_test(y_true, prob_a, prob_b):
    """DeLong's test via the placement-value / structural-component method."""
    def midrank(x):
        order = np.argsort(x)
        ranks = np.empty(len(x))
        ranks[order] = np.arange(1, len(x) + 1)
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        cum = np.cumsum(counts)
        avg = (np.concatenate(([0], cum[:-1])) + cum + 1) / 2.0
        return avg[inv]

    def auc_and_var(y, prob):
        pos = prob[y == 1]
        neg = prob[y == 0]
        n1, n0 = len(pos), len(neg)
        all_scores = np.concatenate([pos, neg])
        ranks = midrank(all_scores)
        r_pos = ranks[:n1]
        auc = (r_pos.sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        v10 = (r_pos - np.arange(1, n1 + 1)) / n0
        r_neg = midrank(-all_scores[n1:])
        v01 = 1 - (r_neg - np.arange(1, n0 + 1)) / n1
        s10 = np.var(v10, ddof=1) / n1
        s01 = np.var(v01, ddof=1) / n0
        return auc, s10 + s01, v10, v01, n1, n0

    y = np.asarray(y_true)
    a_a, var_a, v10_a, v01_a, n1, n0 = auc_and_var(y, prob_a)
    a_b, var_b, v10_b, v01_b, _, _ = auc_and_var(y, prob_b)
    cov = (np.cov(v10_a, v10_b, ddof=1)[0, 1] / n1 +
           np.cov(v01_a, v01_b, ddof=1)[0, 1] / n0)
    se = np.sqrt(max(var_a + var_b - 2 * cov, 1e-12))
    z = (a_a - a_b) / se
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z)))
    return round(z, 4), round(p, 4)


def make_pipe(model):
    return Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc', StandardScaler()),
                      ('clf', model)])


MODELS = {
    'RandomForest': lambda: RandomForestClassifier(
        n_estimators=500, random_state=42, class_weight='balanced', n_jobs=-1),
    'XGBoost': lambda: xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        random_state=42, eval_metric='logloss', verbosity=0),
    'LightGBM': lambda: lgb.LGBMClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        random_state=42, verbosity=-1),
}

if __name__ == '__main__':
    rows = []
    probs = {}

    # ── Parkinson's: LOSO on the primary 6-nonlinear-feature set ──────────────
    print('=' * 60)
    print(' PARKINSON\'S (LOSO, 6-feature nonlinear set)')
    print('=' * 60)
    Xp, yp, mp = load_parkinsons()
    loso = LeaveOneGroupOut()
    for name, ctor in MODELS.items():
        pipe = make_pipe(ctor())
        prob = cross_val_predict(pipe, Xp[NL6].values, yp.values, cv=loso,
                                  groups=mp.subject_id.values,
                                  method='predict_proba')[:, 1]
        probs[('parkinsons', name)] = prob
        auc = roc_auc_score(yp.values, prob)
        mean, lo, hi = bootstrap_auroc_ci(yp.values, prob)
        print(f'  {name:12s} AUROC={auc:.4f} [{lo:.3f}-{hi:.3f}]')
        rows.append({'condition': 'parkinsons', 'protocol': 'LOSO',
                      'model': name, 'auroc': round(auc, 4),
                      'ci_lo': lo, 'ci_hi': hi})

    for name in ['XGBoost', 'LightGBM']:
        z, p = delong_test(yp.values, probs[('parkinsons', 'RandomForest')],
                            probs[('parkinsons', name)])
        print(f'  DeLong RF vs {name}: z={z}, p={p}')
        for r in rows:
            if r['condition'] == 'parkinsons' and r['model'] == name:
                r['delong_z_vs_rf'] = z
                r['delong_p_vs_rf'] = p

    # ── COVID-19: 5-fold stratified CV on the 19-feature set ──────────────────
    print('\n' + '=' * 60)
    print(' COVID-19 (5-fold stratified CV, 19-feature set)')
    print('=' * 60)
    Xr, yr, _ = load_respiratory()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, ctor in MODELS.items():
        pipe = make_pipe(ctor())
        prob = cross_val_predict(pipe, Xr.values, yr.values, cv=skf,
                                  method='predict_proba')[:, 1]
        probs[('respiratory', name)] = prob
        auc = roc_auc_score(yr.values, prob)
        mean, lo, hi = bootstrap_auroc_ci(yr.values, prob)
        print(f'  {name:12s} AUROC={auc:.4f} [{lo:.3f}-{hi:.3f}]')
        rows.append({'condition': 'respiratory', 'protocol': '5-Fold CV',
                      'model': name, 'auroc': round(auc, 4),
                      'ci_lo': lo, 'ci_hi': hi})

    for name in ['XGBoost', 'LightGBM']:
        z, p = delong_test(yr.values, probs[('respiratory', 'RandomForest')],
                            probs[('respiratory', name)])
        print(f'  DeLong RF vs {name}: z={z}, p={p}')
        for r in rows:
            if r['condition'] == 'respiratory' and r['model'] == name:
                r['delong_z_vs_rf'] = z
                r['delong_p_vs_rf'] = p

    df = pd.DataFrame(rows)
    out = RESULTS / 'tree_ensemble_comparison.csv'
    df.to_csv(out, index=False)
    print(f'\nSaved -> {out}')
    print('\n' + '=' * 60)
    print(' SUMMARY')
    print('=' * 60)
    print(df.to_string(index=False))
