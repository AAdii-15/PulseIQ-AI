"""
Re-runs the RF vs. XGBoost vs. LightGBM comparison for COVID-19 under
the corrected participant-grouped protocol (tab:treecomp currently
shows the old, unrouted 5-fold CV numbers -- flagged as inconsistent
with the rest of the corrected paper).

All three models share the exact same StratifiedGroupKFold split
(same seed), so their out-of-fold predictions are paired on identical
held-out instances -- DeLong's test is valid here.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

BASE = Path.home() / 'Desktop/PULSE_IQ_AI'
RESULTS = BASE / 'results/metrics'
df = pd.read_csv(BASE / 'data/raw/coswara/voice_dataset_labeled_full.csv')

feat_cols = ['pitch', 'spectral_centroid', 'zcr', 'jitter', 'shimmer', 'hnr'] + \
            [f'mfcc_{i}' for i in range(1, 14)]
X = df[feat_cols].values
y = df['label'].values
groups = df['user_id'].values

def delong_z(y, s1, s2):
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    def auc_var(s):
        pos, neg = s[y == 1], s[y == 0]
        V10 = np.array([((p > neg).sum() + 0.5 * (p == neg).sum()) / n0 for p in pos])
        V01 = np.array([((n < pos).sum() + 0.5 * (n == pos).sum()) / n1 for n in neg])
        return V10, V01, V10.mean()
    V10a, V01a, aucA = auc_var(s1)
    V10b, V01b, aucB = auc_var(s2)
    varA = np.var(V10a, ddof=1) / n1 + np.var(V01a, ddof=1) / n0
    varB = np.var(V10b, ddof=1) / n1 + np.var(V01b, ddof=1) / n0
    covAB = np.cov(V10a, V10b)[0, 1] / n1 + np.cov(V01a, V01b)[0, 1] / n0
    se = np.sqrt(max(varA + varB - 2 * covAB, 1e-12))
    z = (aucA - aucB) / se
    return z, 2 * (1 - norm.cdf(abs(z)))

def participant_bootstrap_ci(y, p, groups, n_boot=2000, seed=42):
    unique_p = np.unique(groups)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        bp = rng.choice(unique_p, size=len(unique_p), replace=True)
        idx = np.concatenate([np.where(groups == pp)[0] for pp in bp])
        yb, pb = y[idx], p[idx]
        if len(np.unique(yb)) < 2: continue
        boot.append(roc_auc_score(yb, pb))
    return np.percentile(boot, [2.5, 97.5])

cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)

print('Fitting RF (participant-grouped 5-fold CV)...')
rf_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler()),
                     ('clf', RandomForestClassifier(n_estimators=500, random_state=42,
                                                     class_weight='balanced', n_jobs=-1))])
prob_rf = cross_val_predict(rf_pipe, X, y, cv=cv, groups=groups, method='predict_proba')[:, 1]

print('Fitting XGBoost (participant-grouped 5-fold CV)...')
xgb_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler()),
                      ('clf', XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1))])
prob_xgb = cross_val_predict(xgb_pipe, X, y, cv=cv, groups=groups, method='predict_proba')[:, 1]

print('Fitting LightGBM (participant-grouped 5-fold CV)...')
lgb_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler()),
                      ('clf', LGBMClassifier(random_state=42, verbosity=-1, n_jobs=-1))])
prob_lgb = cross_val_predict(lgb_pipe, X, y, cv=cv, groups=groups, method='predict_proba')[:, 1]

auc_rf, auc_xgb, auc_lgb = roc_auc_score(y, prob_rf), roc_auc_score(y, prob_xgb), roc_auc_score(y, prob_lgb)
ci_rf = participant_bootstrap_ci(y, prob_rf, groups)
ci_xgb = participant_bootstrap_ci(y, prob_xgb, groups)
ci_lgb = participant_bootstrap_ci(y, prob_lgb, groups)

z_xgb, p_xgb = delong_z(y, prob_rf, prob_xgb)
z_lgb, p_lgb = delong_z(y, prob_rf, prob_lgb)

print('\n' + '=' * 70)
print(' CORRECTED COVID-19 TREE-ENSEMBLE COMPARISON (participant-grouped)')
print('=' * 70)
print(f'  RF        AUROC={auc_rf:.3f} [{ci_rf[0]:.3f}--{ci_rf[1]:.3f}]')
print(f'  XGBoost   AUROC={auc_xgb:.3f} [{ci_xgb[0]:.3f}--{ci_xgb[1]:.3f}]   z={z_xgb:.3f}  p={p_xgb:.4f}')
print(f'  LightGBM  AUROC={auc_lgb:.3f} [{ci_lgb[0]:.3f}--{ci_lgb[1]:.3f}]   z={z_lgb:.3f}  p={p_lgb:.4f}')
print('=' * 70)

pd.DataFrame([
    {'model':'RF','auroc':round(auc_rf,4),'ci_lo':round(ci_rf[0],4),'ci_hi':round(ci_rf[1],4),'z_vs_rf':None,'p_vs_rf':None},
    {'model':'XGBoost','auroc':round(auc_xgb,4),'ci_lo':round(ci_xgb[0],4),'ci_hi':round(ci_xgb[1],4),'z_vs_rf':round(z_xgb,3),'p_vs_rf':round(p_xgb,4)},
    {'model':'LightGBM','auroc':round(auc_lgb,4),'ci_lo':round(ci_lgb[0],4),'ci_hi':round(ci_lgb[1],4),'z_vs_rf':round(z_lgb,3),'p_vs_rf':round(p_lgb,4)},
]).to_csv(RESULTS / 'covid_tree_ensemble_corrected.csv', index=False)
print(f"\nSaved -> results/metrics/covid_tree_ensemble_corrected.csv")
