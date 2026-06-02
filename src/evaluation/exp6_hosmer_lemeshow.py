import numpy as np, pandas as pd, warnings, sys
warnings.filterwarnings('ignore')
from pathlib import Path
from scipy.stats import chi2
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

def hl(y, p, nb=10):
    d = pd.DataFrame({'y':y,'p':p}).sort_values('p').reset_index(drop=True)
    d['bin'] = pd.qcut(d.p, q=nb, duplicates='drop', labels=False)
    g = d.groupby('bin').agg(obs=('y','sum'),exp=('p','sum'),n=('y','count')).reset_index()
    g['ng']=g.n-g.exp; g['on']=g.n-g.obs
    g = g[(g.exp>0)&(g.ng>0)]
    chi_ = ((g.obs-g.exp)**2/g.exp + (g.on-g.ng)**2/g.ng).sum()
    dof  = max(len(g)-2,1)
    return chi_, dof, 1-chi2.cdf(chi_,dof)

def rf():
    return Pipeline([('imp',SimpleImputer(strategy='median')),
                     ('sc',StandardScaler()),
                     ('clf',RandomForestClassifier(n_estimators=500,random_state=42,
                                                    class_weight='balanced',n_jobs=-1))])
rows=[]

Xp,yp,mp = load_parkinsons()
pr = cross_val_predict(rf(), Xp[NL].values, yp.values,
                        cv=LeaveOneGroupOut(), groups=mp.subject_id.values,
                        method='predict_proba')[:,1]
c,d,pv = hl(yp.values, pr)
print(f'PD          : chi2={c:.3f}  dof={d}  p={pv:.4f}  {"OK (calibrated)" if pv>0.05 else "MISCALIBRATED"}')
rows.append({'condition':'parkinsons','hl_chi2':round(c,3),'dof':d,'hl_p':round(pv,4)})

Xr,yr,_ = load_respiratory()
pr = cross_val_predict(rf(), Xr.values, yr.values,
                        cv=StratifiedKFold(5,shuffle=True,random_state=42),
                        method='predict_proba')[:,1]
c,d,pv = hl(yr.values, pr)
print(f'Respiratory : chi2={c:.3f}  dof={d}  p={pv:.4f}  {"OK (calibrated)" if pv>0.05 else "MISCALIBRATED"}')
rows.append({'condition':'respiratory','hl_chi2':round(c,3),'dof':d,'hl_p':round(pv,4)})

# Depression — retrain inline
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
pr = dep_pipe.predict_proba(dv[fc].values)[:,1]
c,d,pv = hl(dv.PHQ8_Binary.values, pr, nb=5)
print(f'Depression  : chi2={c:.3f}  dof={d}  p={pv:.4f}  {"OK (calibrated)" if pv>0.05 else "MISCALIBRATED"}  (N=35, interpret cautiously)')
rows.append({'condition':'depression','hl_chi2':round(c,3),'dof':d,'hl_p':round(pv,4)})

pd.DataFrame(rows).to_csv(RES/'hosmer_lemeshow.csv', index=False)
print(f'\nSaved -> hosmer_lemeshow.csv')
