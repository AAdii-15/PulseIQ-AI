"""
Exp 8: Recompute proper bootstrap 95% CI for UCI→Sakar cross-dataset validation.
The original CI was a typo (upper bound below point estimate).
"""
import numpy as np, pandas as pd, warnings, sys
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

BASE = Path.home()/'Desktop/PULSE_IQ_AI'
RES  = BASE/'results/metrics'

# Load UCI
uci = pd.read_csv(BASE/'data/raw/uci_parkinsons/parkinsons.csv')
y_uci = uci['status'].values

# Try multiple potential Sakar paths
sakar_paths = [
    BASE/'data/raw/sakar/sakar_parkinsons.csv',
    BASE/'data/raw/sakar_parkinsons/data.csv',
    BASE/'data/raw/sakar/data.csv',
]
sakar = None
for p in sakar_paths:
    if p.exists():
        sakar = pd.read_csv(p); print(f'Loaded Sakar from {p}')
        break

if sakar is None:
    print('Sakar dataset not found locally. Skipping fresh run.')
    print('If you have saved cross-dataset predictions, point to the prediction CSV:')
    pred_path = BASE/'results/metrics/cross_dataset_predictions.csv'
    if pred_path.exists():
        pred = pd.read_csv(pred_path)
        y_true, y_prob = pred['y_true'].values, pred['y_prob'].values
    else:
        print('No saved predictions either. Cannot proceed.')
        sys.exit(1)
else:
    # Find shared features (case-insensitive substring matching)
    uci_cols = set(c.lower() for c in uci.columns)
    sakar_cols = set(c.lower() for c in sakar.columns)
    shared = sorted(uci_cols & sakar_cols)
    feat_candidates = [c for c in shared if any(k in c for k in
                       ['jitter','shimmer','hnr','nhr','f0','pitch','rap','ppq','apq'])]
    print(f'Shared features ({len(feat_candidates)}): {feat_candidates}')
    
    # Match column names case-sensitively
    uci_map = {c.lower(): c for c in uci.columns}
    sakar_map = {c.lower(): c for c in sakar.columns}
    X_uci_train = uci[[uci_map[c] for c in feat_candidates]].values
    X_sakar     = sakar[[sakar_map[c] for c in feat_candidates]].values
    
    # Find Sakar label column
    label_col = [c for c in sakar.columns if c.lower() in ('class','status','label','target')][0]
    y_sakar = sakar[label_col].values
    
    # Train on UCI, test on Sakar
    pipe = Pipeline([('imp',SimpleImputer(strategy='median')),
                     ('sc',StandardScaler()),
                     ('clf',RandomForestClassifier(n_estimators=500,random_state=42,
                                                    class_weight='balanced',n_jobs=-1))])
    pipe.fit(X_uci_train, y_uci)
    y_prob = pipe.predict_proba(X_sakar)[:,1]
    y_true = y_sakar

point_auc = roc_auc_score(y_true, y_prob)
print(f'\nCross-dataset point AUROC: {point_auc:.4f}')

# Proper bootstrap CI
rng = np.random.default_rng(42)
aurocs = []
for _ in range(2000):
    idx = rng.integers(0, len(y_true), len(y_true))
    if len(np.unique(y_true[idx])) < 2: continue
    aurocs.append(roc_auc_score(y_true[idx], y_prob[idx]))
ci_lo, ci_hi = np.percentile(aurocs, [2.5, 97.5])
print(f'95% bootstrap CI: [{ci_lo:.4f}, {ci_hi:.4f}]')
print(f'\n=> REPORT IN PAPER: AUROC {point_auc:.3f} [{ci_lo:.3f}--{ci_hi:.3f}]')

pd.DataFrame([{'auroc':point_auc,'ci_lo':ci_lo,'ci_hi':ci_hi}]).to_csv(
    RES/'cross_dataset_ci_corrected.csv', index=False)
print(f'Saved -> cross_dataset_ci_corrected.csv')
