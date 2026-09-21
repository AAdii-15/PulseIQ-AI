"""
Participant-clustered inference replacing DeLong's and McNemar's tests
for classifier comparisons, on both COVID-19 (Coswara) and PD (UCI).

Recomputes out-of-fold predictions from raw data (does not depend on
previously saved prediction files), so the CV structure is guaranteed
to match. Includes a sanity check against the published 0.802 (PD) /
0.702 (COVID) AUROC values before trusting anything downstream --
if these don't match closely, STOP and check hyperparameters below
before reading the comparison results.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

BASE = Path.home() / 'Desktop/PULSE_IQ_AI'

# ── Cluster-robust inference functions (verified on synthetic data) ─────────

def participant_bootstrap_auroc_diff(df, prob_a_col, prob_b_col, participant_col,
                                      y_col, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    participants = df[participant_col].unique()
    diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        counts = pd.Series(sampled).value_counts()
        idx = np.concatenate([
            np.repeat(df.index[df[participant_col] == p].values, c)
            for p, c in counts.items()
        ])
        resampled = df.loc[idx]
        if resampled[y_col].nunique() < 2:
            continue
        auc_a = roc_auc_score(resampled[y_col], resampled[prob_a_col])
        auc_b = roc_auc_score(resampled[y_col], resampled[prob_b_col])
        diffs.append(auc_a - auc_b)
    diffs = np.array(diffs)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    p_boot = min(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1.0)
    return diffs.mean(), (ci_lo, ci_hi), p_boot

def cluster_permutation_paired(df, participant_col, correct_a_col, correct_b_col,
                                n_perm=2000, seed=42):
    rng = np.random.default_rng(seed)
    def stat(d):
        b = ((d[correct_a_col] == 1) & (d[correct_b_col] == 0)).sum()
        c = ((d[correct_a_col] == 0) & (d[correct_b_col] == 1)).sum()
        return b - c
    observed = stat(df)
    participants = df[participant_col].unique()
    null_stats = []
    for _ in range(n_perm):
        swap_mask = rng.random(len(participants)) < 0.5
        swap_set = set(participants[swap_mask])
        d = df.copy()
        mask = d[participant_col].isin(swap_set)
        d.loc[mask, [correct_a_col, correct_b_col]] = d.loc[mask, [correct_b_col, correct_a_col]].values
        null_stats.append(stat(d))
    null_stats = np.array(null_stats)
    p = (np.abs(null_stats) >= abs(observed)).mean()
    return observed, p

def run_comparison(name, df, participant_col, y_col, prob_a_col, prob_b_col,
                    correct_a_col, correct_b_col, n_boot=2000, n_perm=2000, seed=42):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    print(f"N recordings = {len(df)}, N participants = {df[participant_col].nunique()}")
    diff, (ci_lo, ci_hi), p_boot = participant_bootstrap_auroc_diff(
        df, prob_a_col, prob_b_col, participant_col, y_col, n_boot=n_boot, seed=seed)
    print(f"\nParticipant-clustered AUROC diff (replaces DeLong):")
    print(f"  Delta AUROC = {diff:+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}], cluster-bootstrap p = {p_boot:.4f}")
    observed, p_perm = cluster_permutation_paired(
        df, participant_col, correct_a_col, correct_b_col, n_perm=n_perm, seed=seed)
    print(f"\nParticipant-clustered paired-decision test (replaces McNemar):")
    print(f"  b - c = {observed}, cluster-permutation p = {p_perm:.4f}")

def make_pipeline(model):
    return Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler()), ('clf', model)])

def get_probs_and_correct(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    correct = (y_pred == y_true).astype(int)
    return correct

# ══════════════════════════════════════════════════════════════════════════
# COVID-19 (Coswara)
# ══════════════════════════════════════════════════════════════════════════
print("Loading Coswara data...")
cov_df = pd.read_csv(BASE / 'data/raw/coswara/voice_dataset_labeled_full.csv')
feat_cols = ['pitch', 'spectral_centroid', 'zcr', 'jitter', 'shimmer', 'hnr'] + [f'mfcc_{i}' for i in range(1, 14)]
X_cov = cov_df[feat_cols].values
y_cov = cov_df['label'].values
groups_cov = cov_df['user_id'].values
cv_cov = StratifiedGroupKFold(5, shuffle=True, random_state=42)

models_cov = {
    'rf':  make_pipeline(RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced', n_jobs=-1)),
    'xgb': make_pipeline(XGBClassifier(random_state=42, eval_metric='logloss')),
    'lgbm': make_pipeline(LGBMClassifier(random_state=42, verbose=-1)),
    'lr':  make_pipeline(LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)),
    'svm': make_pipeline(SVC(probability=True, class_weight='balanced', random_state=42)),
}

print("Running 5 models under participant-grouped CV (this will take a while)...")
probs_cov = {}
for name, model in models_cov.items():
    print(f"  fitting {name}...")
    probs_cov[name] = cross_val_predict(model, X_cov, y_cov, cv=cv_cov, groups=groups_cov, method='predict_proba')[:, 1]

sanity_auc = roc_auc_score(y_cov, probs_cov['rf'])
print(f"\n>>> SANITY CHECK: RF COVID AUROC = {sanity_auc:.4f} (published value: 0.702) <<<")
print(">>> If this doesn't closely match 0.702, STOP and check hyperparameters before trusting results below <<<\n")

cov_base = pd.DataFrame({'user_id': groups_cov, 'label': y_cov})
for other in ['xgb', 'lgbm', 'lr', 'svm']:
    df = cov_base.copy()
    df['prob_rf'] = probs_cov['rf']
    df[f'prob_{other}'] = probs_cov[other]
    df['correct_rf'] = get_probs_and_correct(y_cov, probs_cov['rf'])
    df[f'correct_{other}'] = get_probs_and_correct(y_cov, probs_cov[other])
    run_comparison(f"COVID-19: RF vs {other.upper()}", df, 'user_id', 'label',
                    'prob_rf', f'prob_{other}', 'correct_rf', f'correct_{other}')

# ══════════════════════════════════════════════════════════════════════════
# PD (UCI Parkinson's)
# ══════════════════════════════════════════════════════════════════════════
print("\n\nLoading UCI Parkinson's data...")
pd_df = pd.read_csv(BASE / 'data/raw/uci_parkinsons/parkinsons.csv')
pd_df['subject_id'] = pd_df['name'].str.extract(r'(S\d+)')
nl_features = ['RPDE', 'DFA', 'PPE', 'spread1', 'spread2', 'D2']
X_pd = pd_df[nl_features].values
y_pd = pd_df['status'].values
groups_pd = pd_df['subject_id'].values
cv_pd = LeaveOneGroupOut()

models_pd = {
    'rf':  make_pipeline(RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced', n_jobs=-1)),
    'xgb': make_pipeline(XGBClassifier(random_state=42, eval_metric='logloss')),
    'lgbm': make_pipeline(LGBMClassifier(random_state=42, verbose=-1)),
    'lr':  make_pipeline(LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)),
    'svm': make_pipeline(SVC(probability=True, class_weight='balanced', random_state=42)),
}

print("Running 5 models under LOSO (32 folds each, this will take a while)...")
probs_pd = {}
for name, model in models_pd.items():
    print(f"  fitting {name}...")
    probs_pd[name] = cross_val_predict(model, X_pd, y_pd, cv=cv_pd, groups=groups_pd, method='predict_proba')[:, 1]

sanity_auc_pd = roc_auc_score(y_pd, probs_pd['rf'])
print(f"\n>>> SANITY CHECK: RF PD AUROC = {sanity_auc_pd:.4f} (published value: 0.802) <<<")
print(">>> If this doesn't closely match 0.802, STOP and check hyperparameters before trusting results below <<<\n")

pd_base = pd.DataFrame({'subject_id': groups_pd, 'status': y_pd})
for other in ['xgb', 'lgbm', 'lr', 'svm']:
    df = pd_base.copy()
    df['prob_rf'] = probs_pd['rf']
    df[f'prob_{other}'] = probs_pd[other]
    df['correct_rf'] = get_probs_and_correct(y_pd, probs_pd['rf'])
    df[f'correct_{other}'] = get_probs_and_correct(y_pd, probs_pd[other])
    run_comparison(f"PD: RF vs {other.upper()}", df, 'subject_id', 'status',
                    'prob_rf', f'prob_{other}', 'correct_rf', f'correct_{other}')

print("\n\nDone. Paste the full output back.")
