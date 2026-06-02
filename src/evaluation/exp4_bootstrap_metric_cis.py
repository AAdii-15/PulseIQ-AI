import numpy as np, pandas as pd, warnings, sys
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, brier_score_loss
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, mutual_info_classif

BASE = Path.home()/'Desktop/PULSE_IQ_AI'
sys.path.insert(0, str(BASE/'src'))
from feature_extraction.data_loaders import load_parkinsons, load_respiratory
RES = BASE/'results/metrics'
NL  = ['RPDE','DFA','PPE','spread1','spread2','D2']

def ece(y, p, nb=10):
    b = np.linspace(0,1,nb+1); e=0; n=len(y)
    for i in range(nb):
        m = (p>=b[i]) & (p<b[i+1])
        if m.sum(): e += (m.sum()/n)*abs(y[m].mean()-p[m].mean())
    return e

def metrics(y, p, t=0.5):
    pr = (p>=t).astype(int)
    tn,fp,fn,tp = confusion_matrix(y,pr).ravel()
    s = tp/(tp+fn) if (tp+fn) else 0
    sp = tn/(tn+fp) if (tn+fp) else 0
    return {'auroc':roc_auc_score(y,p),'bacc':(s+sp)/2,
            'f1':f1_score(y,pr,zero_division=0),'sens':s,'spec':sp,
            'brier':brier_score_loss(y,p),'ece':ece(y,p)}

def boot_ci(y, p, t=0.5, n=2000):
    rng = np.random.default_rng(42); rows=[]
    for _ in range(n):
        idx = rng.integers(0,len(y),len(y))
        if len(np.unique(y[idx]))<2: continue
        rows.append(metrics(y[idx],p[idx],t))
    d = pd.DataFrame(rows)
    return {f'{c}_{s}':round(v,4) for c in d.columns
            for s,v in zip(['mean','lo','hi'],
                           [d[c].mean(),np.percentile(d[c],2.5),np.percentile(d[c],97.5)])}

def rf():
    return Pipeline([('imp',SimpleImputer(strategy='median')),
                     ('sc',StandardScaler()),
                     ('clf',RandomForestClassifier(n_estimators=500,random_state=42,
                                                    class_weight='balanced',n_jobs=-1))])

# ── Parkinson's ────────────────────────────────────────────────────────────────
print('PD LOSO...')
Xp,yp,mp = load_parkinsons()
prob_p = cross_val_predict(rf(), Xp[NL].values, yp.values,
                            cv=LeaveOneGroupOut(),
                            groups=mp.subject_id.values,
                            method='predict_proba')[:,1]
ci_p = boot_ci(yp.values, prob_p)
print('  Done')

# ── Respiratory ────────────────────────────────────────────────────────────────
print('Respiratory 5-fold...')
Xr,yr,_ = load_respiratory()
prob_r = cross_val_predict(rf(), Xr.values, yr.values,
                            cv=StratifiedKFold(5,shuffle=True,random_state=42),
                            method='predict_proba')[:,1]
ci_r = boot_ci(yr.values, prob_r)
print('  Done')

# ── Depression (retrain inline — avoids sklearn pickle version mismatch) ───────
print('Depression dev set (retraining inline)...')
dd    = pd.read_csv(BASE/'data/features/daic_woz_covarep_allframes.csv')
trn_s = pd.read_csv(BASE/'data/raw/daic_woz/train_split_Depression_AVEC2017.csv').rename(columns={'Participant_ID':'participant_id'})
dev_s = pd.read_csv(BASE/'data/raw/daic_woz/dev_split_Depression_AVEC2017.csv').rename(columns={'Participant_ID':'participant_id'})
fc    = [c for c in dd.columns if c.endswith('_mean') or c.endswith('_std')]
tr    = dd[dd.participant_id.isin(trn_s.participant_id)]
dv    = dd[dd.participant_id.isin(dev_s.participant_id)]

dep_pipe = Pipeline([('imp',SimpleImputer(strategy='median')),
                     ('sc',StandardScaler()),
                     ('sel',SelectKBest(mutual_info_classif,k=30)),
                     ('svm',SVC(kernel='rbf',probability=True,random_state=42))])
dep_pipe.fit(tr[fc].values, tr.PHQ8_Binary.values)
prob_d  = dep_pipe.predict_proba(dv[fc].values)[:,1]
y_dep   = dv.PHQ8_Binary.values
ci_d    = boot_ci(y_dep, prob_d, t=0.50)
ci_dopt = boot_ci(y_dep, prob_d, t=0.2744)
print('  Done')

# ── Save ───────────────────────────────────────────────────────────────────────
rows=[]
for cond,ci in [('parkinsons',ci_p),('respiratory',ci_r),
                ('depression_t05',ci_d),('depression_topt',ci_dopt)]:
    rows.append({'condition':cond,**ci})
pd.DataFrame(rows).to_csv(RES/'bootstrap_metric_cis.csv',index=False)

print('\n=== BOOTSTRAP 95% CIs (2000 resamples) ===')
for r in rows:
    print(f'\n{r["condition"]}:')
    for m in ['auroc','bacc','f1','sens','spec','brier','ece']:
        print(f'  {m:7s}: {r[m+"_mean"]:.3f}  [{r[m+"_lo"]:.3f}, {r[m+"_hi"]:.3f}]')
print(f'\nSaved -> bootstrap_metric_cis.csv')
