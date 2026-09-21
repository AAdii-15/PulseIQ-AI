"""
Resolving the Table 2 vs. Figure 4B counterfactual discrepancy
(77.0% vs 77.9% MFCC attribution, with Figure 4B additionally showing
an "Other = 9.7%" category that Table 2 does not report at all).

HYPOTHESIS BEING TESTED
------------------------
The counterfactual analysis (COVAREP removed) leaves 34 "shared" feature
columns: 26 MFCC-related (13 mean + 13 std), 5 nonlinear (PPE, RPDE, DFA,
spread1, spread2), and 3 "other" columns (zcr_mean, sc_mean, hnr) that the
ORIGINAL fix2b_shared_space.py script's own categorization logic never
assigns to any bucket (verified directly against that script earlier).

Two different normalizations are possible once COVAREP is removed:
  (a) MFCC% = MFCC_shap / (MFCC_shap + NL_shap)          <- excludes "other"
  (b) MFCC% = MFCC_shap / (MFCC_shap + NL_shap + Other_shap)  <- full 100%

This script computes the counterfactual model fresh, using the exact same
verified data-loading and merge logic as the original script (matching the
sanity-checked 0.612 dev AUROC), and reports BOTH normalizations plainly,
so the real discrepancy source can be seen directly rather than guessed.

Whichever one matches your actually-generated Figure 4B / Table 2 tells
you which convention was used where -- update the OTHER one to match,
per "do not decide based on which number looks better."
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import shap

BASE = Path.home() / 'Desktop/PULSE_IQ_AI'


def get_positive_class_shap(shap_values):
    if isinstance(shap_values, list):
        return shap_values[1]
    elif shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


def make_model(seed=42):
    return Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=500, random_state=seed,
                                        class_weight='balanced', n_jobs=-1)),
    ])


# ── Exact same loading/merge logic as fix2b_shared_space.py (verified) ─────
print("Loading data (matching fix2b_shared_space.py exactly)...")
df = pd.read_csv(BASE / 'data/features/daic_woz_covarep_allframes.csv')
train_df = pd.read_csv(BASE / 'data/raw/daic_woz/train_split_Depression_AVEC2017.csv').rename(columns={'Participant_ID': 'participant_id'})
dev_df = pd.read_csv(BASE / 'data/raw/daic_woz/dev_split_Depression_AVEC2017.csv').rename(columns={'Participant_ID': 'participant_id'})
shared_df = pd.read_csv(BASE / 'data/features/daic_woz_shared_features.csv')

cov_feats = [c for c in df.columns if c.endswith('_mean') or c.endswith('_std')]
shared_feats = [c for c in shared_df.columns if c.startswith('shared_')]

merged = df[['participant_id', 'PHQ8_Binary'] + cov_feats].merge(
    shared_df[['participant_id'] + shared_feats], on='participant_id')

train = merged[merged.participant_id.isin(train_df.participant_id)]
dev = merged[merged.participant_id.isin(dev_df.participant_id)]

y_tr = train.PHQ8_Binary.values
y_dv = dev.PHQ8_Binary.values

# ── Sanity check against the already-published, verified 0.612 dev AUROC ───
all_feats = cov_feats + shared_feats
pipe = make_model(seed=42)
pipe.fit(train[all_feats].values, y_tr)
sanity_auc = roc_auc_score(y_dv, pipe.predict_proba(dev[all_feats].values)[:, 1])
print(f"\n>>> SANITY CHECK: full-space dev AUROC = {sanity_auc:.4f} (published: 0.612) <<<")
print(">>> If this doesn't closely match, STOP before trusting results below <<<\n")

# ── Define the three sub-groups within the 34 "shared" (non-COVAREP) columns ──
mfcc_cols = [f for f in shared_feats if 'shared_mfcc' in f]
nl_cols = [f for f in shared_feats if any(f.startswith(f'shared_{k}') for k in ['ppe', 'rpde', 'dfa', 'spread'])]
other_cols = [f for f in shared_feats if f not in mfcc_cols and f not in nl_cols]
print(f"MFCC columns: {len(mfcc_cols)}, Nonlinear columns: {len(nl_cols)}, "
      f"Other (uncategorized) columns: {len(other_cols)} -> {other_cols}")

# ── Counterfactual model: shared_feats only, COVAREP removed entirely ──────
cf_model = make_model(seed=42)
cf_model.fit(train[shared_feats].values, y_tr)
cf_auc = roc_auc_score(y_dv, cf_model.predict_proba(dev[shared_feats].values)[:, 1])
print(f"\nCounterfactual (COVAREP removed) train-set AUROC check, dev AUROC = {cf_auc:.4f}")

explainer = shap.TreeExplainer(cf_model.named_steps['clf'])
X_tr_t = cf_model[:-1].transform(train[shared_feats].values)
sv = get_positive_class_shap(explainer.shap_values(X_tr_t))
mean_shap = pd.Series(np.abs(sv).mean(axis=0), index=shared_feats)

mfcc_shap = mean_shap[mfcc_cols].sum()
nl_shap = mean_shap[nl_cols].sum()
other_shap = mean_shap[other_cols].sum()

print("\n" + "=" * 70)
print("NORMALIZATION (a): MFCC + Nonlinear only (excludes 'other')")
print("=" * 70)
total_ab = mfcc_shap + nl_shap
print(f"  MFCC: {mfcc_shap/total_ab*100:.1f}%")
print(f"  Nonlinear: {nl_shap/total_ab*100:.1f}%")
print(f"  (sums to 100% by construction, ignores {other_shap:.4f} raw 'other' SHAP mass)")

print("\n" + "=" * 70)
print("NORMALIZATION (b): Full shared space (MFCC + Nonlinear + Other)")
print("=" * 70)
total_full = mfcc_shap + nl_shap + other_shap
print(f"  MFCC: {mfcc_shap/total_full*100:.1f}%")
print(f"  Nonlinear: {nl_shap/total_full*100:.1f}%")
print(f"  Other (zcr/sc/hnr): {other_shap/total_full*100:.1f}%")
print(f"  (sums to 100.0% including the 'other' category)")

print("\n\nCompare these two blocks against Table 2 (77.0/12.4) and Figure 4B")
print("(77.9/12.4/9.7) to see which normalization each one actually used.")
print("Paste the full output back.")
