"""
Verify Table II's attribution-RF AUROC (currently stated as 0.626).

Reconstructs the EXACT data-loading and feature-merging logic from
fix2b_shared_space.py (the original script), not an approximation --
same file paths, same column selection, same RF hyperparameters
(500 trees, random_state=42, class_weight='balanced'). Only the SHAP
computation is skipped here (not needed to check AUROC, and much
slower); everything that affects the AUROC number is identical.

Checks against BOTH candidate values: Table II's stated 0.626, and
fix2b_shared_space_shap.csv's saved depression_auroc_shared=0.6123,
which is supposed to be the exact same number and currently isn't.
"""
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

BASE = Path.home() / 'Desktop/PULSE_IQ_AI'

print('Loading COVAREP + shared feature files (same paths as fix2b_shared_space.py)...')
df       = pd.read_csv(BASE / 'data/features/daic_woz_covarep_allframes.csv')
train_df = pd.read_csv(BASE / 'data/raw/daic_woz/train_split_Depression_AVEC2017.csv').rename(
               columns={'Participant_ID': 'participant_id'})
dev_df   = pd.read_csv(BASE / 'data/raw/daic_woz/dev_split_Depression_AVEC2017.csv').rename(
               columns={'Participant_ID': 'participant_id'})

cache = BASE / 'data/features/daic_woz_shared_features.csv'
if not cache.exists():
    print(f'ERROR: {cache} not found. This file should already exist from the original run.')
    raise SystemExit(1)
shared_df = pd.read_csv(cache)
print(f'Loaded cached shared features: {len(shared_df)} sessions, {len(shared_df.columns)} columns')

cov_feats    = [c for c in df.columns if c.endswith('_mean') or c.endswith('_std')]
shared_feats = [c for c in shared_df.columns if c.startswith('shared_')]
print(f'COVAREP features: {len(cov_feats)} | Shared features: {len(shared_feats)}')

merged = df[['participant_id', 'PHQ8_Binary'] + cov_feats].merge(
             shared_df[['participant_id'] + shared_feats], on='participant_id')
print(f'Merged: {len(merged)} sessions')

train = merged[merged.participant_id.isin(train_df.participant_id)]
dev   = merged[merged.participant_id.isin(dev_df.participant_id)]
all_feats = cov_feats + shared_feats
print(f'Total feature dimension: {len(all_feats)}  (paper states 178)')

X_tr, y_tr = train[all_feats].values, train.PHQ8_Binary.values
X_dv, y_dv = dev[all_feats].values, dev.PHQ8_Binary.values

print('\nFitting RF (500 trees, seed=42, class_weight=balanced) -- exact match to original...')
pipe = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('sc', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=500, random_state=42,
                                    class_weight='balanced', n_jobs=-1))
])
pipe.fit(X_tr, y_tr)
y_prob = pipe.predict_proba(X_dv)[:, 1]
auc = roc_auc_score(y_dv, y_prob)

print('\n' + '=' * 70)
print(' VERIFICATION RESULT')
print('=' * 70)
print(f'  Freshly computed AUROC : {auc:.4f}')
print(f'  Table II currently states: 0.626')
print(f'  fix2b_shared_space_shap.csv saved: 0.6123')
print('=' * 70)
if abs(auc - 0.6123) < 0.001:
    print('\n-> Matches the SAVED CSV exactly. Table II\'s 0.626 is the stale/wrong one.')
elif abs(auc - 0.626) < 0.001:
    print('\n-> Matches Table II. The saved CSV (0.6123) must be from an older run.')
else:
    print(f'\n-> Matches NEITHER exactly. This RF may also be seed-sensitive --')
    print('   paste this output back and we will check seed robustness next,')
    print('   same as we just did for the SVM.')
