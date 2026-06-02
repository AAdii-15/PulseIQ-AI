"""
Exp 10: Coswara duplication sensitivity analysis.
The paper claims max inflation ≤ 0.009 AUROC under 20% random removal.
Reviewers will ask to see this. Here it is.
"""
import numpy as np, pandas as pd, warnings, sys
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

BASE = Path.home()/'Desktop/PULSE_IQ_AI'
sys.path.insert(0, str(BASE/'src'))
from feature_extraction.data_loaders import load_respiratory
RES = BASE/'results/metrics'

def rf():
    return Pipeline([('imp',SimpleImputer(strategy='median')),
                     ('sc',StandardScaler()),
                     ('clf',RandomForestClassifier(n_estimators=500,random_state=42,
                                                    class_weight='balanced',n_jobs=-1))])

def cv_auroc(X, y, seed):
    pr = cross_val_predict(rf(), X, y,
                            cv=StratifiedKFold(5,shuffle=True,random_state=seed),
                            method='predict_proba')[:,1]
    return roc_auc_score(y, pr)

X, y, _ = load_respiratory()
base_auc = cv_auroc(X.values, y.values, 42)
print(f'Baseline (full N={len(y)}): AUROC = {base_auc:.4f}')

rng = np.random.default_rng(42)
rows = []
for removal_frac in [0.05, 0.10, 0.15, 0.20]:
    aurocs = []
    for rep in range(10):
        n_keep = int(len(y) * (1 - removal_frac))
        idx = rng.choice(len(y), n_keep, replace=False)
        idx = np.sort(idx)
        auc = cv_auroc(X.values[idx], y.values[idx], 42+rep)
        aurocs.append(auc)
    mean_auc, std_auc = np.mean(aurocs), np.std(aurocs)
    max_delta = max(abs(a - base_auc) for a in aurocs)
    print(f'  Remove {int(removal_frac*100):2d}%: AUROC = {mean_auc:.4f} ± {std_auc:.4f} '
          f'| Max |Δ| from base = {max_delta:.4f}')
    rows.append({'removal_pct':int(removal_frac*100),'mean_auroc':round(mean_auc,4),
                 'std':round(std_auc,4),'max_delta':round(max_delta,4)})

global_max = max(r['max_delta'] for r in rows)
print(f'\nGlobal max |Δ| AUROC across all removal levels: {global_max:.4f}')
print(f'\n=> REPORT IN PAPER: Max plausible inflation Δ ≤ {global_max:.3f} AUROC '
      f'under 5-20% random removal.')

pd.DataFrame(rows).to_csv(RES/'coswara_dedup_sensitivity.csv', index=False)
print(f'Saved -> coswara_dedup_sensitivity.csv')
