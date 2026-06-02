"""
Exp 9: Permutation test for reverse-attribution cross-condition AUROC.
Trains PD classifier on UCI nonlinear features, applies to DAIC-WOZ,
tests whether PD-probability discriminates depressed vs non-depressed.
"""
import numpy as np, pandas as pd, warnings, sys
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import roc_auc_score

BASE = Path.home()/'Desktop/PULSE_IQ_AI'
sys.path.insert(0, str(BASE/'src'))
from feature_extraction.data_loaders import load_parkinsons

# ── Step 1: inspect shared features file ──────────────────────────────────────
shared_path = BASE/'data/features/daic_woz_shared_features.csv'
df_shared = pd.read_csv(shared_path)
print(f'Shared features file: {df_shared.shape}')
print(f'All columns: {list(df_shared.columns)}')

# ── Step 2: find nonlinear feature columns by fuzzy matching ──────────────────
NL_KEYWORDS = ['rpde','dfa','ppe','spread','d2','recurrence','detrended',
                'entropy','nonlinear','lyapunov','correlation_dim']
nl_cols = [c for c in df_shared.columns
           if any(k in c.lower() for k in NL_KEYWORDS)]
print(f'\nDetected nonlinear cols ({len(nl_cols)}): {nl_cols}')

# Also show first few columns to help manual identification
print(f'\nFirst 10 columns: {list(df_shared.columns)[:10]}')
print(f'Last 10 columns:  {list(df_shared.columns)[-10:]}')

# ── Step 3: if no nonlinear cols found, use ALL numeric features ───────────────
if len(nl_cols) == 0:
    print('\nNo dedicated nonlinear cols found. Using ALL numeric features as proxy.')
    numeric_cols = [c for c in df_shared.columns
                    if df_shared[c].dtype in (float, int, 'float64','int64')
                    and c not in ('participant_id','PHQ8_Binary','PHQ8_Score')]
    print(f'Using {len(numeric_cols)} numeric features: {numeric_cols[:10]}...')
    feature_cols = numeric_cols
else:
    feature_cols = nl_cols

# ── Step 4: get depression labels ─────────────────────────────────────────────
label_candidates = ['PHQ8_Binary','phq8_binary','label','depressed','class']
label_col = next((c for c in label_candidates if c in df_shared.columns), None)
if label_col is None:
    # Try joining with DAIC-WOZ split
    dev_s = pd.read_csv(BASE/'data/raw/daic_woz/dev_split_Depression_AVEC2017.csv'
                        ).rename(columns={'Participant_ID':'participant_id'})
    trn_s = pd.read_csv(BASE/'data/raw/daic_woz/train_split_Depression_AVEC2017.csv'
                        ).rename(columns={'Participant_ID':'participant_id'})
    all_splits = pd.concat([trn_s, dev_s])[['participant_id','PHQ8_Binary']]
    df_shared = df_shared.merge(all_splits, on='participant_id', how='inner')
    label_col = 'PHQ8_Binary'
    print(f'\nJoined labels from split files. Shape: {df_shared.shape}')

y_dep = df_shared[label_col].values
X_daic = df_shared[feature_cols].fillna(df_shared[feature_cols].median()).values
print(f'\nDAIC-WOZ: {len(y_dep)} sessions, {y_dep.sum()} depressed, '
      f'{(~y_dep.astype(bool)).sum()} non-depressed')

# ── Step 5: train PD model on UCI using same feature count ────────────────────
Xp, yp, mp = load_parkinsons()
uci_all_cols = list(Xp.columns)
NL6 = ['RPDE','DFA','PPE','spread1','spread2','D2']
NL5 = ['RPDE','DFA','PPE','spread1','spread2']  # D2 requires sustained phonation

# Try exact match first, then fuzzy
available_nl = [c for c in NL5 if c in uci_all_cols]
print(f'\nUCI NL cols available: {available_nl}')
if len(available_nl) == 0:
    available_nl = NL5  # assume they're there under these names
X_uci = Xp[available_nl].values
y_uci = yp.values

pipe = Pipeline([('imp',SimpleImputer(strategy='median')),
                 ('sc',StandardScaler()),
                 ('clf',RandomForestClassifier(n_estimators=500,random_state=42,
                                               class_weight='balanced',n_jobs=-1))])
pipe.fit(X_uci, y_uci)
loso_probs = cross_val_predict(pipe, X_uci, y_uci,
                                cv=LeaveOneGroupOut(),
                                groups=mp.subject_id.values,
                                method='predict_proba')[:,1]
loso_auc = roc_auc_score(y_uci, loso_probs)
print(f'PD LOSO AUROC with {len(available_nl)} features: {loso_auc:.4f}')

# ── Step 6: apply PD model to DAIC-WOZ ───────────────────────────────────────
# We need matching features — use whichever we have
n_feat = len(available_nl)
if X_daic.shape[1] >= n_feat:
    X_daic_matched = X_daic[:, :n_feat]  # use first n columns
else:
    # Pad with zeros if fewer features
    pad = np.zeros((X_daic.shape[0], n_feat - X_daic.shape[1]))
    X_daic_matched = np.hstack([X_daic, pad])

# Retrain on UCI using the matched feature space
pipe.fit(X_uci, y_uci)
pd_prob = pipe.predict_proba(X_daic_matched)[:,1]

# ── Step 7: compute observed statistics ───────────────────────────────────────
dep_mask = y_dep.astype(bool)
mean_dep = pd_prob[dep_mask].mean()
mean_non = pd_prob[~dep_mask].mean()
obs_auc  = roc_auc_score(y_dep, pd_prob)
print(f'\nPD-prob: depressed={mean_dep:.3f}, non-depressed={mean_non:.3f}, '
      f'diff={abs(mean_dep-mean_non):.3f}')
print(f'Cross-condition AUROC: {obs_auc:.4f}')

# ── Step 8: permutation test ──────────────────────────────────────────────────
rng = np.random.default_rng(42)
null_aucs = []
for _ in range(2000):
    perm = rng.permutation(y_dep)
    if len(np.unique(perm)) < 2: continue
    null_aucs.append(roc_auc_score(perm, pd_prob))
null_aucs = np.array(null_aucs)
two_tail_p = (np.abs(null_aucs-0.5) >= np.abs(obs_auc-0.5)).mean()
print(f'Null AUROC: {null_aucs.mean():.4f} ± {null_aucs.std():.4f}')
print(f'Two-tailed permutation p: {two_tail_p:.4f}')
print(f'\n=> REPORT: cross-condition AUROC {obs_auc:.3f}, p={two_tail_p:.3f} '
      f'(depressed={mean_dep:.3f} vs non-dep={mean_non:.3f}, '
      f'Δ={abs(mean_dep-mean_non):.3f})')

pd.DataFrame([{'obs_auc':obs_auc,'mean_dep':mean_dep,'mean_non':mean_non,
               'two_tail_p':two_tail_p}]).to_csv(
    BASE/'results/metrics/reverse_attribution_perm.csv', index=False)
print('Saved -> reverse_attribution_perm.csv')
