"""
Two things in one script:
  1. Recomputes COVID-19's SHAP Jaccard stability (top-5, top-10) under
     the corrected participant-grouped protocol -- the current Figure 3
     values were computed under the old, unrouted CV.
  2. Builds a real null/random-baseline Jaccard distribution for all
     three conditions (simulated top-k draws from each condition's
     actual feature universe, no real signal), replacing the previous
     unsubstantiated "noise floor" language with an actual comparison.

PD note: with only 6 total features and k=5, ANY two random draws
overlap in >=4 of 5 by pigeonhole -- the null baseline is ~0.72, not
~0. This means PD's observed J5=1.000, while numerically perfect, is
much closer to its own chance baseline than COVID's or depression's
values are to theirs. Reported honestly here, not glossed over.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from functools import partial
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

BASE = Path.home() / 'Desktop/PULSE_IQ_AI'
RESULTS = BASE / 'results/metrics'

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)

def pairwise_jaccard_mean(top_sets):
    vals = []
    for i in range(len(top_sets)):
        for j in range(i+1, len(top_sets)):
            vals.append(jaccard(top_sets[i], top_sets[j]))
    return np.mean(vals)

def null_jaccard(n_universe, k, n_trials=20000, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_trials):
        a = rng.choice(n_universe, min(k, n_universe), replace=False)
        b = rng.choice(n_universe, min(k, n_universe), replace=False)
        vals.append(jaccard(a, b))
    return np.array(vals)

# ── 1. Recompute COVID's real Jaccard under participant-grouped protocol ──────
print('Recomputing COVID-19 SHAP Jaccard stability (participant-grouped, 20 bootstrap replicates)...')
df = pd.read_csv(BASE / 'data/raw/coswara/voice_dataset_labeled_full.csv')
feat_cols = ['pitch', 'spectral_centroid', 'zcr', 'jitter', 'shimmer', 'hnr'] + [f'mfcc_{i}' for i in range(1, 14)]
X_full = df[feat_cols].values
y_full = df['label'].values
groups_full = df['user_id'].values
unique_p = np.unique(groups_full)

top5_sets, top10_sets = [], []
rng = np.random.default_rng(42)
for b in range(20):
    bp = rng.choice(unique_p, size=len(unique_p), replace=True)
    idx = np.concatenate([np.where(groups_full == p)[0] for p in bp])
    Xb, yb = X_full[idx], y_full[idx]

    pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler()),
                      ('clf', RandomForestClassifier(n_estimators=500, random_state=42,
                                                      class_weight='balanced', n_jobs=-1))])
    pipe.fit(Xb, yb)
    explainer = shap.TreeExplainer(pipe['clf'])
    sv = explainer.shap_values(pipe[:-1].transform(Xb))
    if isinstance(sv, list):
        sv = sv[1]
    mean_abs = np.abs(sv).mean(axis=0)
    ranked = np.argsort(-mean_abs)
    top5_sets.append(ranked[:5])
    top10_sets.append(ranked[:10])
    print(f'  replicate {b+1}/20 done')

covid_j5 = pairwise_jaccard_mean(top5_sets)
covid_j10 = pairwise_jaccard_mean(top10_sets)
print(f'\nCOVID-19 (corrected protocol): J5={covid_j5:.3f}  J10={covid_j10:.3f}')
print(f'(previously reported under old protocol: J5=1.000  J10=0.918)')

# ── 2. Null baselines for all three conditions ────────────────────────────────
print('\nSimulating null baselines (20,000 trials each)...')
null_pd_5   = null_jaccard(6, 5)     # PD: 6 total features
null_pd_10  = null_jaccard(6, 10)    # degenerate: k>n, always the full set
null_covid_5  = null_jaccard(19, 5)  # COVID: 19 total features
null_covid_10 = null_jaccard(19, 10)
null_dep_5  = null_jaccard(30, 5)    # Depression: 30 SelectKBest features
null_dep_10 = null_jaccard(30, 10)

results = [
    ('PD',         1.000, null_pd_5,    1.000, null_pd_10),
    ('COVID-19',   covid_j5, null_covid_5, covid_j10, null_covid_10),
    ('Depression', 0.063, null_dep_5,   0.114, null_dep_10),
]

print('\n' + '=' * 90)
print(' JACCARD STABILITY vs. NULL BASELINE (top-5 / top-10)')
print('=' * 90)
rows = []
for name, j5, n5, j10, n10 in results:
    n5_mean, n10_mean = n5.mean(), n10.mean()
    p5 = (n5 >= j5).mean()
    p10 = (n10 >= j10).mean()
    print(f'  {name:10s}  J5={j5:.3f} (null={n5_mean:.3f}, p={p5:.4f})   '
          f'J10={j10:.3f} (null={n10_mean:.3f}, p={p10:.4f})')
    rows.append({'condition': name, 'j5_observed': round(j5,4), 'j5_null_mean': round(n5_mean,4),
                 'j5_p': round(p5,4), 'j10_observed': round(j10,4), 'j10_null_mean': round(n10_mean,4),
                 'j10_p': round(p10,4)})
print('=' * 90)
print('\np = fraction of null-simulation trials with Jaccard >= observed.')
print('Small p means the observed stability is genuinely above chance for that')
print('feature-universe size. Note PD\'s high null baseline (pigeonhole effect).')

pd.DataFrame(rows).to_csv(RESULTS / 'jaccard_null_comparison.csv', index=False)
print(f"\nSaved -> results/metrics/jaccard_null_comparison.csv")
