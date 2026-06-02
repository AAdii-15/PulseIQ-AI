"""
Adds depression to the SHAP-Jaccard stability analysis
(currently only PD and Respiratory have this).
"""
import numpy as np, pandas as pd, shap, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

BASE = Path.home()/'Desktop/PULSE_IQ_AI'
RES  = BASE/'results/metrics'

df = pd.read_csv(BASE/'data/features/daic_woz_covarep_allframes.csv')
trn = pd.read_csv(BASE/'data/raw/daic_woz/train_split_Depression_AVEC2017.csv').rename(columns={'Participant_ID':'participant_id'})
fc = [c for c in df.columns if c.endswith('_mean') or c.endswith('_std')]
T  = df[df.participant_id.isin(trn.participant_id)]
X, y = T[fc].values, T.PHQ8_Binary.values

N=20
print(f'Running {N} bootstraps...')
rng = np.random.default_rng(42)
ranks = []
for b in range(N):
    idx = rng.integers(0, len(X), len(X))
    Xb, yb = X[idx], y[idx]
    if len(np.unique(yb))<2: continue
    p = Pipeline([('imp',SimpleImputer(strategy='median')),
                  ('sc',StandardScaler()),
                  ('clf',RandomForestClassifier(n_estimators=300, random_state=b,
                                                 class_weight='balanced', n_jobs=-1))])
    p.fit(Xb, yb)
    sv = shap.TreeExplainer(p.named_steps['clf']).shap_values(p[:-1].transform(Xb))
    sv = sv[1] if isinstance(sv,list) else sv[:,:,1]
    ranks.append(np.argsort(-np.abs(sv).mean(axis=0)))
    print(f'  {b+1}/{N}')

def jacc(a,b):
    sa,sb = set(a),set(b); return len(sa&sb)/len(sa|sb) if (sa|sb) else 0

rows=[]
for k in [3,5,10,15,20]:
    j=[jacc(ranks[i][:k],ranks[m][:k]) for i in range(len(ranks)) for m in range(i+1,len(ranks))]
    rows.append({'condition':'depression','top_k':k,
                 'jaccard_mean':round(np.mean(j),4),
                 'jaccard_std':round(np.std(j),4),
                 'jaccard_min':round(np.min(j),4)})

out = pd.DataFrame(rows)
out.to_csv(RES/'shap_jaccard_depression.csv', index=False)
print('\n=== DEPRESSION SHAP JACCARD ===')
print(out.to_string(index=False))
print(f'\nSaved -> shap_jaccard_depression.csv')
