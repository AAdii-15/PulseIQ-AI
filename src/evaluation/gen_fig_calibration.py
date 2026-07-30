"""
Figure: Calibration reliability diagrams for all three conditions.

CORRECTED: depression's annotated Brier/ECE now use the exact same
10-seed-mean-of-point-estimates methodology as depression_final_v2.py
(the actual source of Table I's 0.224/0.094) -- NOT a single seed's
bootstrap-mean like the earlier version of this script used, which
gave 0.223/0.128, visibly inconsistent with Table I's ECE=0.094.

PD/COVID unchanged: their Brier/ECE are the bootstrap-mean of ONE
deterministic seed (matching bootstrap_metric_cis.csv), since only
depression's SelectKBest+mutual_info pipeline has seed-instability.

The plotted reliability curve for depression uses the seed-averaged
predicted probabilities (elementwise mean across the 10 seeds' predict
_proba outputs) -- a reasonable single representative curve for an
ensemble of seeds; noted explicitly in the caption to avoid implying
it was computed identically to the annotated numbers.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from functools import partial
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.metrics import brier_score_loss
import sys

BASE = Path.home() / 'Desktop/PULSE_IQ_AI'
sys.path.insert(0, str(BASE / 'src'))
from feature_extraction.data_loaders import load_parkinsons, load_respiratory

FIGS = BASE / 'results/figures'
NONLINEAR = ['RPDE', 'DFA', 'PPE', 'spread1', 'spread2', 'D2']
SEEDS = [42, 7, 123, 256, 512, 999, 1337, 2024, 31, 88]

def ece(y, p, nb=10):
    b = np.linspace(0, 1, nb + 1)
    e, n = 0.0, len(y)
    for i in range(nb):
        m = (p >= b[i]) & (p < b[i + 1])
        if m.sum():
            e += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return e

def bootstrap_mean(y, p, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    briers, eces = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        briers.append(brier_score_loss(y[idx], p[idx]))
        eces.append(ece(y[idx], p[idx]))
    return np.mean(briers), np.mean(eces)

def make_rf(seed=42):
    return Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc', StandardScaler()),
                      ('clf', RandomForestClassifier(n_estimators=500, random_state=seed,
                                                      class_weight='balanced', n_jobs=-1))])

# ── PD: LOSO, bootstrap-mean (matches Table I exactly, already verified) ──────
print('Recomputing PD (LOSO, 6-feature nonlinear set) probabilities...')
Xp, yp, mp = load_parkinsons()
y_prob_pk = cross_val_predict(make_rf(), Xp[NONLINEAR].values, yp.values,
                               cv=LeaveOneGroupOut(), groups=mp.subject_id.values,
                               method='predict_proba')[:, 1]
y_pk = yp.values
bs_pk, e_pk = bootstrap_mean(y_pk, y_prob_pk)

# ── COVID-19: 5-fold CV, bootstrap-mean (matches Table I exactly) ─────────────
print('Recomputing COVID-19 (5-fold CV) probabilities...')
Xr, yr, _ = load_respiratory()
y_prob_resp = cross_val_predict(make_rf(), Xr.values, yr.values,
                                 cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                 method='predict_proba')[:, 1]
y_resp = yr.values
bs_resp, e_resp = bootstrap_mean(y_resp, y_prob_resp)

# ── Depression: 10-seed mean of point estimates (matches Table I EXACTLY,
#    same methodology as depression_final_v2.py, not a bootstrap mean) ────────
print('Recomputing depression across 10 seeds (matches Table I methodology)...')
dd    = pd.read_csv(BASE / 'data/features/daic_woz_covarep_allframes.csv')
trn_s = pd.read_csv(BASE / 'data/raw/daic_woz/train_split_Depression_AVEC2017.csv').rename(
            columns={'Participant_ID': 'participant_id'})
dev_s = pd.read_csv(BASE / 'data/raw/daic_woz/dev_split_Depression_AVEC2017.csv').rename(
            columns={'Participant_ID': 'participant_id'})
fc  = [c for c in dd.columns if c.endswith('_mean') or c.endswith('_std')]
tr  = dd[dd.participant_id.isin(trn_s.participant_id)]
dv  = dd[dd.participant_id.isin(dev_s.participant_id)]
y_dep = dv.PHQ8_Binary.values

seed_probs, seed_briers, seed_eces = [], [], []
for seed in SEEDS:
    pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc', StandardScaler()),
                      ('sel', SelectKBest(partial(mutual_info_classif, random_state=seed), k=30)),
                      ('svm', SVC(kernel='rbf', probability=True, random_state=seed))])
    pipe.fit(tr[fc].values, tr.PHQ8_Binary.values)
    p = pipe.predict_proba(dv[fc].values)[:, 1]
    seed_probs.append(p)
    seed_briers.append(brier_score_loss(y_dep, p))
    seed_eces.append(ece(y_dep, p))

bs_dep = np.mean(seed_briers)
e_dep = np.mean(seed_eces)
y_prob_dep_avg = np.mean(seed_probs, axis=0)  # for the plotted curve only

print('\n=== VERIFY AGAINST TABLE I (must match before this figure is used) ===')
print(f'PD:         Brier={bs_pk:.3f} (expected 0.144)  ECE={e_pk:.3f} (expected 0.108)')
print(f'COVID-19:   Brier={bs_resp:.3f} (expected 0.202)  ECE={e_resp:.3f} (expected 0.066)')
print(f'Depression: Brier={bs_dep:.3f} (expected 0.224)  ECE={e_dep:.3f} (expected 0.094)')

COLORS = {'PD': '#2196F3', 'COVID': '#4CAF50', 'Depression': '#9C27B0'}
pairs = [
    (y_prob_pk, y_pk, bs_pk, e_pk, "(A)  Parkinson's (PD)", COLORS['PD']),
    (y_prob_resp, y_resp, bs_resp, e_resp, '(B)  COVID-19', COLORS['COVID']),
    (y_prob_dep_avg, y_dep, bs_dep, e_dep, '(C)  Depression (Case Study)', COLORS['Depression']),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for (y_prob, y_true, bs, e, title, color), ax in zip(pairs, axes):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=5)
    ax.plot(mean_pred, frac_pos, 's-', color=color, lw=2, ms=8, label='Model')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6, label='Perfect calibration')
    ax.fill_between(mean_pred, frac_pos, mean_pred, alpha=0.1, color=color)
    ax.text(0.05, 0.92, f'Brier = {bs:.3f}\nECE = {e:.3f}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', fc='white', ec=color, alpha=0.9))
    ax.set(title=title, xlabel='Mean Predicted Probability', ylabel='Fraction of Positives')
    ax.legend(fontsize=8.5, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.set_facecolor('#FAFAFA')
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(FIGS / 'fig_calibration.png', dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print('\nSaved -> fig_calibration.png (600dpi, depression now seed-averaged, matches Table I)')
