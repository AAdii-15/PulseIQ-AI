"""
Held-out (cross-fitted) SHAP re-analysis for the depression attribution
model. Data loading, merge logic, and feature categorization replicate
fix2b_shared_space.py EXACTLY (verified against the real script), so
results are directly comparable to the published training-set numbers.
The only methodological change: SHAP is computed on the held-out AVEC
dev set instead of the training set used to fit the model.
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
    return Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler()),
                      ('clf', RandomForestClassifier(n_estimators=500, random_state=seed,
                                                      class_weight='balanced', n_jobs=-1))])

# ── Exact same loading/merge logic as fix2b_shared_space.py ────────────────
print("Loading data (matching fix2b_shared_space.py exactly)...")
df = pd.read_csv(BASE / 'data/features/daic_woz_covarep_allframes.csv')
train_df = pd.read_csv(BASE / 'data/raw/daic_woz/train_split_Depression_AVEC2017.csv').rename(columns={'Participant_ID': 'participant_id'})
dev_df = pd.read_csv(BASE / 'data/raw/daic_woz/dev_split_Depression_AVEC2017.csv').rename(columns={'Participant_ID': 'participant_id'})
shared_df = pd.read_csv(BASE / 'data/features/daic_woz_shared_features.csv')

cov_feats = [c for c in df.columns if c.endswith('_mean') or c.endswith('_std')]
shared_feats = [c for c in shared_df.columns if c.startswith('shared_')]

merged = df[['participant_id', 'PHQ8_Binary'] + cov_feats].merge(
    shared_df[['participant_id'] + shared_feats], on='participant_id')
print(f"Merged: {len(merged)} sessions | COVAREP: {len(cov_feats)} | Shared: {len(shared_feats)}")

train = merged[merged.participant_id.isin(train_df.participant_id)]
dev = merged[merged.participant_id.isin(dev_df.participant_id)]
all_feats = cov_feats + shared_feats

X_tr = train[all_feats].values; y_tr = train.PHQ8_Binary.values
X_dv = dev[all_feats].values; y_dv = dev.PHQ8_Binary.values
print(f"N train={len(X_tr)}, N dev={len(X_dv)} (expect 107, 35)")

def categorize(all_feats, mean_shap, cov_feats):
    """Exact same categorization as the original script -- note that
    shared_zcr_mean, shared_sc_mean, and shared_hnr are NOT assigned to
    any of the three buckets, matching the original's own totals."""
    mfcc = sum(v for f, v in zip(all_feats, mean_shap) if 'shared_mfcc' in f)
    nl = sum(v for f, v in zip(all_feats, mean_shap)
             if any(f.startswith(f'shared_{k}') for k in ['ppe', 'rpde', 'dfa', 'spread']))
    cov = sum(v for f, v in zip(all_feats, mean_shap) if f in cov_feats)
    total = mfcc + nl + cov
    return {'COVAREP': cov/total*100, 'MFCC': mfcc/total*100, 'NL': nl/total*100}

# ── 1. Sanity check: fit exactly as original, check dev AUROC ──────────────
pipe = make_model(seed=42)
pipe.fit(X_tr, y_tr)
sanity_auc = roc_auc_score(y_dv, pipe.predict_proba(X_dv)[:, 1])
print(f"\n>>> SANITY CHECK: dev AUROC = {sanity_auc:.4f} (published: 0.612) <<<")
print(">>> If this doesn't closely match, STOP before trusting results below <<<\n")

# ── 2. FULL-SPACE held-out attribution (the key methodological fix) ────────
print("="*70)
print("1. FULL-SPACE: held-out SHAP (dev set) vs. original training-set SHAP")
print("="*70)
explainer = shap.TreeExplainer(pipe.named_steps['clf'])
X_dv_t = pipe[:-1].transform(X_dv)
sv_dev = get_positive_class_shap(explainer.shap_values(X_dv_t))
mean_shap_dev = np.abs(sv_dev).mean(axis=0)
held_out_full = categorize(all_feats, mean_shap_dev, cov_feats)
print("Held-out (dev-set) attribution:")
for k, v in held_out_full.items():
    print(f"  {k}: {v:.1f}%")
print("Published training-set attribution: COVAREP 74.2%, MFCC 23.2%, NL 2.6%")

# ── 3. CARDINALITY-MATCHED, held-out (adapted logic, not from an unseen script) ──
print("\n" + "="*70)
print("2. CARDINALITY-MATCHED (5+5+5), held-out")
print("="*70)
mfcc_cols = [f for f in shared_feats if 'shared_mfcc' in f]
nl_cols = [f for f in shared_feats if any(f.startswith(f'shared_{k}') for k in ['ppe','rpde','dfa','spread'])]
rng = np.random.default_rng(42)
results = []
for r in range(200):
    mfcc_sub = list(rng.choice(mfcc_cols, size=5, replace=False))
    cov_sub = list(rng.choice(cov_feats, size=5, replace=False))
    cols = nl_cols + mfcc_sub + cov_sub
    m = make_model(seed=r)
    m.fit(train[cols].values, y_tr)
    exp_r = shap.TreeExplainer(m.named_steps['clf'])
    Xdv_t = m[:-1].transform(dev[cols].values)
    sv_r = get_positive_class_shap(exp_r.shap_values(Xdv_t))
    ma = pd.Series(np.abs(sv_r).mean(axis=0), index=cols)
    tot = ma.sum()
    results.append({'NL': ma[nl_cols].sum()/tot*100, 'MFCC': ma[mfcc_sub].sum()/tot*100, 'COVAREP': ma[cov_sub].sum()/tot*100})
rdf = pd.DataFrame(results)
print("Held-out matched attribution (mean [2.5%, 97.5%]):")
for k in rdf.columns:
    print(f"  {k}: {rdf[k].mean():.1f}% [{rdf[k].quantile(0.025):.1f}, {rdf[k].quantile(0.975):.1f}]")
print("Published training-set matched: MFCC 41.0%, COVAREP 32.5%, NL 26.5%")

# ── 4. COUNTERFACTUAL, held-out (without COVAREP) ───────────────────────────
print("\n" + "="*70)
print("3. COUNTERFACTUAL (without COVAREP), held-out")
print("="*70)
cf_cols = nl_cols + mfcc_cols
cf_model = make_model(seed=42)
cf_model.fit(train[cf_cols].values, y_tr)
cf_auc = roc_auc_score(y_dv, cf_model.predict_proba(dev[cf_cols].values)[:, 1])
exp_cf = shap.TreeExplainer(cf_model.named_steps['clf'])
Xdv_cf_t = cf_model[:-1].transform(dev[cf_cols].values)
sv_cf = get_positive_class_shap(exp_cf.shap_values(Xdv_cf_t))
ma_cf = pd.Series(np.abs(sv_cf).mean(axis=0), index=cf_cols)
tot_cf = ma_cf.sum()
print(f"  MFCC: {ma_cf[mfcc_cols].sum()/tot_cf*100:.1f}%")
print(f"  NL: {ma_cf[nl_cols].sum()/tot_cf*100:.1f}%")
print(f"  Counterfactual dev AUROC: {cf_auc:.4f}")
print("Published training-set counterfactual: MFCC 77.0%, NL 12.4%, AUROC 0.658")

# ── 5. HELD-OUT JACCARD STABILITY ───────────────────────────────────────────
print("\n" + "="*70)
print("4. HELD-OUT JACCARD STABILITY (top-5)")
print("="*70)
top_k_sets = []
for b in range(20):
    idx = rng.choice(len(X_tr), size=len(X_tr), replace=True)
    m = make_model(seed=b)
    m.fit(X_tr[idx], y_tr[idx])
    exp_b = shap.TreeExplainer(m.named_steps['clf'])
    Xdv_t_b = m[:-1].transform(X_dv)
    sv_b = get_positive_class_shap(exp_b.shap_values(Xdv_t_b))
    ma_b = pd.Series(np.abs(sv_b).mean(axis=0), index=all_feats)
    top_k_sets.append(set(ma_b.nlargest(5).index))
jaccards = []
for i in range(len(top_k_sets)):
    for j in range(i+1, len(top_k_sets)):
        inter = len(top_k_sets[i] & top_k_sets[j])
        union = len(top_k_sets[i] | top_k_sets[j])
        jaccards.append(inter/union)
print(f"Held-out top-5 Jaccard stability: {np.mean(jaccards):.4f}")
print("Published training-set value: J_5 = 0.063")

print("\n\nDone. Paste the full output back.")
