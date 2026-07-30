"""
Multi-seed permutation test for depression AUROC significance.

The observed statistic (0.621) is the mean AUROC across 10 seeds
(depression_final_v2.py). For a matching null distribution, each of
2,000 trials permutes the true labels ONCE and computes AUROC against
ALL 10 seeds' fitted probabilities, then averages -- exactly mirroring
how the real observed statistic was computed. This is cheap: no
retraining, just permuting labels against already-fitted predictions
(same eq:permtest formula already used elsewhere in the paper).
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
from sklearn.metrics import roc_auc_score

BASE = Path.home() / 'Desktop/PULSE_IQ_AI'
RESULTS = BASE / 'results/metrics'
SEEDS = [42, 7, 123, 256, 512, 999, 1337, 2024, 31, 88]

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
for seed in SEEDS:
    pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc', StandardScaler()),
                      ('sel', SelectKBest(partial(mutual_info_classif, random_state=seed), k=30)),
                      ('svm', SVC(kernel='rbf', probability=True, random_state=seed))])
    pipe.fit(tr[fc].values, tr.PHQ8_Binary.values)
    seed_probs.append(pipe.predict_proba(dv[fc].values)[:, 1])

observed = np.mean([roc_auc_score(y_true, p) for p in seed_probs])
print(f'Observed (10-seed mean) AUROC: {observed:.4f}  (should match 0.621)')

print('\nRunning 2,000-permutation combined null test...')
rng = np.random.default_rng(42)
null_means = []
for _ in range(2000):
    perm_y = rng.permutation(y_true)
    null_means.append(np.mean([roc_auc_score(perm_y, p) for p in seed_probs]))
null_means = np.array(null_means)

p_value = np.mean(null_means >= observed)

print('\n' + '=' * 70)
print(' PERMUTATION TEST RESULT')
print('=' * 70)
print(f'  Observed 10-seed mean AUROC : {observed:.4f}')
print(f'  Null distribution mean      : {null_means.mean():.4f}  (sanity check: should be ~0.500)')
print(f'  Null distribution std       : {null_means.std():.4f}')
print(f'  One-sided p-value           : {p_value:.4f}')
print('=' * 70)

pd.DataFrame([{
    'observed_auroc': round(observed, 4),
    'null_mean': round(null_means.mean(), 4),
    'null_std': round(null_means.std(), 4),
    'p_value': round(p_value, 4),
    'n_permutations': 2000, 'n_seeds': len(SEEDS)
}]).to_csv(RESULTS / 'depression_permutation_test.csv', index=False)
print(f"\nSaved -> {RESULTS / 'depression_permutation_test.csv'}")
