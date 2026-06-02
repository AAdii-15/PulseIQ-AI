"""
Final polish experiments:
  A) Group ablation: remove each feature family, measure dev AUROC
  B) Cardinality control: 50 → 200 reps
  C) Bootstrap 95% CI on reverse attribution AUROC 0.486
"""
import numpy as np, pandas as pd, warnings, sys
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
try:
    import shap
except ImportError:
    import subprocess; subprocess.check_call([sys.executable,'-m','pip','install','shap','--quiet'])
    import shap

BASE = Path.home()/'Desktop/PULSE_IQ_AI'
sys.path.insert(0, str(BASE/'src'))
from feature_extraction.data_loaders import load_parkinsons

# ── Load data ─────────────────────────────────────────────────────
df_sh  = pd.read_csv(BASE/'data/features/daic_woz_shared_features.csv')
df_cov = pd.read_csv(BASE/'data/features/daic_woz_covarep_allframes.csv')
trn = pd.read_csv(BASE/'data/raw/daic_woz/train_split_Depression_AVEC2017.csv'
                  ).rename(columns={'Participant_ID':'participant_id'})
dev = pd.read_csv(BASE/'data/raw/daic_woz/dev_split_Depression_AVEC2017.csv'
                  ).rename(columns={'Participant_ID':'participant_id'})
labels = pd.concat([trn,dev])[['participant_id','PHQ8_Binary']]

for _df, _name in [(df_sh,'shared'),(df_cov,'covarep')]:
    drop = [c for c in _df.columns if 'phq' in c.lower()]
    _df.drop(columns=drop, inplace=True, errors='ignore')

df = df_sh.merge(df_cov, on='participant_id', how='inner', suffixes=('','_cov'))
df = df.merge(labels, on='participant_id', how='inner')

NL   = ['shared_ppe','shared_rpde','shared_dfa','shared_spread1','shared_spread2']
MFCC = [c for c in df.columns if 'mfcc' in c.lower() and c in df_sh.columns]
COV  = [c for c in df_cov.columns if c != 'participant_id']

df_tr = df[df.participant_id.isin(trn.participant_id)]
df_dv = df[df.participant_id.isin(dev.participant_id)]
print(f'Train:{len(df_tr)} Dev:{len(df_dv)} | NL:{len(NL)} MFCC:{len(MFCC)} COV:{len(COV)}')

def svm_auroc(cols, tr, dv):
    k = min(30, len(cols))
    pipe = Pipeline([('imp',SimpleImputer(strategy='median')),
                     ('sc', StandardScaler()),
                     ('sel',SelectKBest(f_classif,k=k)),
                     ('clf',SVC(kernel='rbf',probability=True,class_weight='balanced',random_state=42))])
    X_tr = tr[cols].fillna(tr[cols].median()).values
    y_tr = tr.PHQ8_Binary.values
    X_dv = dv[cols].fillna(dv[cols].median()).values
    y_dv = dv.PHQ8_Binary.values
    pipe.fit(X_tr,y_tr)
    p = pipe.predict_proba(X_dv)[:,1]
    return roc_auc_score(y_dv,p) if len(np.unique(y_dv))>1 else np.nan

# ════════════════════════════════════════════════════
# PART A — Group ablation
# ════════════════════════════════════════════════════
print('\n=== PART A: GROUP ABLATION (dev AUROC) ===')
configs = {
    'All features (NL+MFCC+COV)': NL+MFCC+COV,
    'No nonlinear  (MFCC+COV)  ': MFCC+COV,
    'No MFCC       (NL+COV)    ': NL+COV,
    'No COVAREP    (MFCC+NL)   ': MFCC+NL,
    'NL only                   ': NL,
    'MFCC only                 ': MFCC,
    'COVAREP only              ': COV,
}
abl = {}
for name, cols in configs.items():
    a = svm_auroc(cols, df_tr, df_dv)
    abl[name] = a
    print(f'  {name}: {a:.4f}')

# ════════════════════════════════════════════════════
# PART B — Cardinality control 200 reps
# ════════════════════════════════════════════════════
print('\n=== PART B: CARDINALITY CONTROL (200 reps, 5+5+5) ===')
rng = np.random.default_rng(42)
rows = []
for rep in range(200):
    ms = list(rng.choice(MFCC,5,replace=False))
    cs = list(rng.choice(COV,5,replace=False))
    cols = NL+ms+cs
    X = df_tr[cols].fillna(df_tr[cols].median()).values
    y = df_tr.PHQ8_Binary.values
    pipe = Pipeline([('imp',SimpleImputer()),('sc',StandardScaler()),
                     ('clf',RandomForestClassifier(200,random_state=rep,
                      class_weight='balanced',n_jobs=-1))])
    pipe.fit(X,y)
    sv = shap.TreeExplainer(pipe['clf']).shap_values(pipe[:-1].transform(X),
                                                      check_additivity=False)
    if isinstance(sv,list): sv=sv[1]
    if sv.ndim==3: sv=sv[:,:,1]
    a=np.abs(sv).mean(0)
    nl_=a[:5].sum(); mf_=a[5:10].sum(); cv_=a[10:].sum(); tot=nl_+mf_+cv_
    rows.append({'nl':100*nl_/tot,'mfcc':100*mf_/tot,'cov':100*cv_/tot})
    if (rep+1)%50==0: print(f'  {rep+1}/200')

df_r=pd.DataFrame(rows)
for col,nm in [('cov','COVAREP'),('mfcc','MFCC'),('nl','Nonlinear')]:
    v=df_r[col]
    print(f'  {nm:10s}: {v.mean():.1f}%±{v.std():.1f}% [{v.quantile(.025):.1f}–{v.quantile(.975):.1f}]')
df_r.to_csv(BASE/'results/metrics/cardinality_control_200.csv',index=False)

# ════════════════════════════════════════════════════
# PART C — Bootstrap CI for reverse attribution
# ════════════════════════════════════════════════════
print('\n=== PART C: BOOTSTRAP CI — REVERSE ATTRIBUTION ===')
Xp,yp,mp = load_parkinsons()
NL5=['RPDE','DFA','PPE','spread1','spread2']
pipe_pd=Pipeline([('imp',SimpleImputer()),('sc',StandardScaler()),
                  ('clf',RandomForestClassifier(500,random_state=42,
                   class_weight='balanced',n_jobs=-1))])
pipe_pd.fit(Xp[NL5].values,yp.values)

col_map={'RPDE':'shared_rpde','DFA':'shared_dfa','PPE':'shared_ppe',
          'spread1':'shared_spread1','spread2':'shared_spread2'}
X_daic=df[[col_map[c] for c in NL5]].fillna(df[[col_map[c] for c in NL5]].median()).values
y_dep=df.PHQ8_Binary.values
pd_prob=pipe_pd.predict_proba(X_daic)[:,1]
obs=roc_auc_score(y_dep,pd_prob)

rng2=np.random.default_rng(99)
boots=[roc_auc_score(y_dep[idx:=rng2.choice(len(y_dep),len(y_dep),replace=True)],
                      pd_prob[idx])
       for _ in range(2000)
       if len(np.unique(y_dep[idx:=rng2.choice(len(y_dep),len(y_dep),replace=True)]))>1]
boots=np.array(boots)
lo,hi=np.percentile(boots,[2.5,97.5])
print(f'  Observed AUROC: {obs:.4f}')
print(f'  Bootstrap 95% CI: [{lo:.4f}–{hi:.4f}]')
print(f'\n=> REPORT: Reverse attribution AUROC {obs:.3f} [95% CI {lo:.3f}–{hi:.3f}], p=0.738')

print('\n=== ALL DONE — paste full output ===')
