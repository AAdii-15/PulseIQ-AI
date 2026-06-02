"""
Counterfactual Fix 2B: train depression classifier on shared features ONLY
(no COVAREP). Check if nonlinear SHAP attribution remains low.
"""
import numpy as np, pandas as pd, shap, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

BASE = Path.home()/'Desktop/PULSE_IQ_AI'
RES  = BASE/'results/metrics'

shared = pd.read_csv(BASE/'data/features/daic_woz_shared_features.csv')
labels = pd.read_csv(BASE/'data/features/daic_woz_covarep_allframes.csv')[['participant_id','PHQ8_Binary']]
trn = pd.read_csv(BASE/'data/raw/daic_woz/train_split_Depression_AVEC2017.csv').rename(columns={'Participant_ID':'participant_id'})
dev = pd.read_csv(BASE/'data/raw/daic_woz/dev_split_Depression_AVEC2017.csv').rename(columns={'Participant_ID':'participant_id'})

df = shared.merge(labels, on='participant_id')
mfcc_cols = [c for c in df.columns if 'shared_mfcc' in c]
nl_cols   = ['shared_ppe','shared_rpde','shared_dfa','shared_spread1','shared_spread2']
oth_cols  = ['shared_zcr_mean','shared_sc_mean','shared_hnr']
feats     = mfcc_cols + nl_cols + oth_cols

print(f'MFCC={len(mfcc_cols)} | Nonlinear={len(nl_cols)} | Other={len(oth_cols)} | TOTAL={len(feats)} (no COVAREP)')

train = df[df.participant_id.isin(trn.participant_id)]
devset = df[df.participant_id.isin(dev.participant_id)]
X_tr, y_tr = train[feats].values, train.PHQ8_Binary.values
X_dv, y_dv = devset[feats].values, devset.PHQ8_Binary.values

pipe = Pipeline([('imp',SimpleImputer(strategy='median')),
                 ('sc',StandardScaler()),
                 ('clf',RandomForestClassifier(n_estimators=500, random_state=42,
                                                class_weight='balanced', n_jobs=-1))])
pipe.fit(X_tr, y_tr)
auc = roc_auc_score(y_dv, pipe.predict_proba(X_dv)[:,1])
print(f'\nDepression AUROC (shared-only, no COVAREP): {auc:.4f}')

rf = pipe.named_steps['clf']
sv = shap.TreeExplainer(rf).shap_values(pipe[:-1].transform(X_tr))
sv = sv[1] if isinstance(sv,list) else sv[:,:,1]
mabs = np.abs(sv).mean(axis=0)
mfcc_i = [feats.index(c) for c in mfcc_cols]
nl_i   = [feats.index(c) for c in nl_cols]
oth_i  = [feats.index(c) for c in oth_cols]
mp, nl_v, op = mabs[mfcc_i].sum(), mabs[nl_i].sum(), mabs[oth_i].sum()
tot = mp + nl_v + op
print(f'\n=== POINT ESTIMATE ===')
print(f'MFCC:      {100*mp/tot:.2f}%')
print(f'Nonlinear: {100*nl_v/tot:.2f}%')
print(f'Other:     {100*op/tot:.2f}%')

print(f'\nRunning 200 bootstraps...')
rng = np.random.default_rng(42)
B = []
for b in range(200):
    idx = rng.integers(0, len(X_tr), len(X_tr))
    Xb, yb = X_tr[idx], y_tr[idx]
    if len(np.unique(yb)) < 2: continue
    p = Pipeline([('imp',SimpleImputer(strategy='median')),
                  ('sc',StandardScaler()),
                  ('clf',RandomForestClassifier(n_estimators=300, random_state=b,
                                                 class_weight='balanced', n_jobs=-1))])
    p.fit(Xb, yb)
    svb = shap.TreeExplainer(p.named_steps['clf']).shap_values(p[:-1].transform(Xb))
    svb = svb[1] if isinstance(svb,list) else svb[:,:,1]
    ma = np.abs(svb).mean(axis=0)
    mb, nb, ob = ma[mfcc_i].sum(), ma[nl_i].sum(), ma[oth_i].sum()
    t = mb+nb+ob
    B.append({'mfcc_pct':100*mb/t,'nl_pct':100*nb/t,'oth_pct':100*ob/t})

bdf = pd.DataFrame(B)
print(f'\n=== BOOTSTRAP CIs (200 resamples) ===')
for c in ['mfcc_pct','nl_pct','oth_pct']:
    lo,hi = np.percentile(bdf[c],[2.5,97.5])
    print(f'{c:9s}: {bdf[c].mean():.2f}% ± {bdf[c].std():.2f}  95%CI=[{lo:.2f}, {hi:.2f}]')

nlm = bdf.nl_pct.mean()
v = ("PASSES: nonlinear stays low without COVAREP. Separability claim survives." if nlm<15
     else "MIXED: reframe needed but claim partly survives." if nlm<30
     else "FAILS: original 2.6% was an artifact of training dynamics.")
print(f'\nVERDICT: {v}')

bdf.to_csv(RES/'fix2b_counterfactual_bootstrap.csv', index=False)
pd.DataFrame([{
    'auroc_shared_only':round(auc,4),
    'mfcc_pct':round(bdf.mfcc_pct.mean(),2), 'nl_pct':round(nlm,2),
    'nl_ci_lo':round(np.percentile(bdf.nl_pct,2.5),2),
    'nl_ci_hi':round(np.percentile(bdf.nl_pct,97.5),2),
    'verdict':v
}]).to_csv(RES/'fix2b_counterfactual_summary.csv', index=False)
print(f'\nSaved -> fix2b_counterfactual_summary.csv')
