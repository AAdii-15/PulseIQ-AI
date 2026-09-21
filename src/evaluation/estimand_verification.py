"""
Verifies the AUROC estimand for both PD and COVID-19: recording-level
pooled AUROC (what's currently reported) vs. subject/participant-level
aggregated AUROC (mean prediction per person, then AUROC over that
smaller set of points).

Raised by external review: grouped/LOSO folds prevent leakage in
TRAINING, but do not automatically make the reported AUROC a
subject-level number -- a person with more recordings gets more
influence on a pooled-recording AUROC. This checks how much that
actually matters for both datasets, before deciding whether to
relabel or recompute the paper's primary numbers.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

BASE = Path.home() / 'Desktop/PULSE_IQ_AI'
RESULTS = BASE / 'results/metrics'

def aggregate_and_compare(y, probs, groups, label):
    recording_auc = roc_auc_score(y, probs)
    unique_g = np.unique(groups)
    agg_probs, agg_labels = [], []
    label_mismatches = 0
    for g in unique_g:
        mask = groups == g
        agg_probs.append(probs[mask].mean())
        vals = set(y[mask])
        if len(vals) > 1:
            label_mismatches += 1
        agg_labels.append(y[mask][0])
    agg_auc = roc_auc_score(agg_labels, agg_probs)
    print(f'\n--- {label} ---')
    print(f'  N recordings={len(y)}, N groups={len(unique_g)}')
    print(f'  Recording-level (pooled) AUROC:      {recording_auc:.4f}')
    print(f'  {"Subject" if "PD" in label else "Participant"}-level (aggregated) AUROC: {agg_auc:.4f}')
    print(f'  Difference: {recording_auc - agg_auc:+.4f}')
    if label_mismatches:
        print(f'  WARNING: {label_mismatches} groups have inconsistent labels across their own recordings!')
    return recording_auc, agg_auc

def make_rf(seed=42):
    return Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler()),
                      ('clf', RandomForestClassifier(n_estimators=500, random_state=seed,
                                                      class_weight='balanced', n_jobs=-1))])

# ── PD ──────────────────────────────────────────────────────────────────────
print('Loading UCI Parkinson\'s data...')
pd_df = pd.read_csv(BASE / 'data/raw/uci_parkinsons/parkinsons.csv')
pd_df['subject_id'] = pd_df['name'].str.extract(r'(S\d+)')
nl_features = ['RPDE', 'DFA', 'PPE', 'spread1', 'spread2', 'D2']  # the primary 6-feature model
X_pd = pd_df[nl_features].values
y_pd = pd_df['status'].values
groups_pd = pd_df['subject_id'].values

print('Running LOSO for PD...')
probs_pd = cross_val_predict(make_rf(), X_pd, y_pd,
                              cv=LeaveOneGroupOut(), groups=groups_pd,
                              method='predict_proba')[:, 1]
pd_rec, pd_subj = aggregate_and_compare(y_pd, probs_pd, groups_pd, 'PD (LOSO, 6-feature)')

# ── COVID-19 ────────────────────────────────────────────────────────────────
print('\nLoading Coswara data...')
cov_df = pd.read_csv(BASE / 'data/raw/coswara/voice_dataset_labeled_full.csv')
feat_cols = ['pitch', 'spectral_centroid', 'zcr', 'jitter', 'shimmer', 'hnr'] + [f'mfcc_{i}' for i in range(1, 14)]
X_cov = cov_df[feat_cols].values
y_cov = cov_df['label'].values
groups_cov = cov_df['user_id'].values

print('Running participant-grouped 5-fold CV for COVID-19...')
probs_cov = cross_val_predict(make_rf(), X_cov, y_cov,
                               cv=StratifiedGroupKFold(5, shuffle=True, random_state=42),
                               groups=groups_cov, method='predict_proba')[:, 1]
cov_rec, cov_part = aggregate_and_compare(y_cov, probs_cov, groups_cov, 'COVID-19 (participant-grouped)')

print('\n' + '=' * 70)
print(' SUMMARY')
print('=' * 70)
print(f'  PD:       recording-level {pd_rec:.4f}  vs.  subject-level {pd_subj:.4f}  (diff {pd_rec-pd_subj:+.4f})')
print(f'  COVID-19: recording-level {cov_rec:.4f}  vs.  participant-level {cov_part:.4f}  (diff {cov_rec-cov_part:+.4f})')

pd.DataFrame([
    {'condition': 'PD', 'recording_level_auroc': round(pd_rec,4), 'aggregated_auroc': round(pd_subj,4), 'diff': round(pd_rec-pd_subj,4)},
    {'condition': 'COVID-19', 'recording_level_auroc': round(cov_rec,4), 'aggregated_auroc': round(cov_part,4), 'diff': round(cov_rec-cov_part,4)},
]).to_csv(RESULTS / 'estimand_verification.csv', index=False)
print(f"\nSaved -> results/metrics/estimand_verification.csv")
