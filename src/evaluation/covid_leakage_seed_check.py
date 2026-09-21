"""
Seed-robustness check for the COVID-19 participant-grouped leakage
finding. Isolates fold-ASSIGNMENT variance (which participants land in
which fold) from model-training variance -- RF's own training-seed
variance was already shown to be tiny elsewhere in this project
(COVID AUROC 0.757+/-0.001 under the old, unrouped protocol), but that
doesn't guarantee fold-assignment variance is equally small under
grouped CV, since group-based splitting has less "smoothing" room than
sample-based splitting. RF's own seed is held fixed; only the
StratifiedGroupKFold split seed varies.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

BASE = Path.home() / 'Desktop/PULSE_IQ_AI'
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

SPLIT_SEEDS = [42, 7, 123, 256, 512]
print(f'Running participant-grouped 5-fold CV across {len(SPLIT_SEEDS)} different fold-assignment seeds...')
print('(RF training seed held fixed at 42 -- only fold assignment varies)\n')

aucs = []
for seed in SPLIT_SEEDS:
    prob = cross_val_predict(make_rf(seed=42), X, y,
                              cv=StratifiedGroupKFold(5, shuffle=True, random_state=seed),
                              groups=groups, method='predict_proba')[:, 1]
    auc = roc_auc_score(y, prob)
    aucs.append(auc)
    print(f'  split_seed={seed:5d}  AUROC={auc:.4f}')

aucs = np.array(aucs)
print(f'\n{"="*60}')
print(f' RESULT: {aucs.mean():.4f} +/- {aucs.std():.4f}   [min={aucs.min():.4f}, max={aucs.max():.4f}]')
print(f'{"="*60}')
print(f'\nFor comparison: recording-level (leaked) AUROC was 0.7546')
print(f'Original single-seed grouped estimate was 0.7020')
print(f'\nIf std is small (~PD/COVID model-seed levels of +/-0.001-0.01),')
print(f'the leakage finding is solid. If std is large, we need to report')
print(f'a range/CI rather than a single point estimate, same as we did')
print(f'for the depression pipeline earlier.')
