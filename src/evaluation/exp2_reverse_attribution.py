"""
Reverse attribution: train 5-feature PD model on UCI, apply to DAIC.
Check if depressed subjects get high PD probability.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu

BASE = Path.home()/'Desktop/PULSE_IQ_AI'
RES  = BASE/'results/metrics'

uci = pd.read_csv(BASE/'data/raw/uci_parkinsons/parkinsons.csv')
uci['subject_id'] = uci['name'].apply(lambda x: x.split('_')[2])
nl5 = ['RPDE','DFA','PPE','spread1','spread2']

pipe_pd = Pipeline([('imp',SimpleImputer(strategy='median')),
                    ('sc',StandardScaler()),
                    ('clf',RandomForestClassifier(n_estimators=500, random_state=42,
                                                   class_weight='balanced', n_jobs=-1))])
auc_loso = roc_auc_score(uci.status.values,
    cross_val_predict(pipe_pd, uci[nl5].values, uci.status.values,
                      cv=LeaveOneGroupOut(), groups=uci.subject_id.values,
                      method='predict_proba')[:,1])
print(f'5-nonlinear PD LOSO AUROC: {auc_loso:.4f}')

pipe_pd.fit(uci[nl5].values, uci.status.values)

shared = pd.read_csv(BASE/'data/features/daic_woz_shared_features.csv')
labels = pd.read_csv(BASE/'data/features/daic_woz_covarep_allframes.csv')[['participant_id','PHQ8_Binary']]
df = shared.merge(labels, on='participant_id')
shared_nl = ['shared_rpde','shared_dfa','shared_ppe','shared_spread1','shared_spread2']
df['pd_prob'] = pipe_pd.predict_proba(df[shared_nl].values)[:,1]

dep = df[df.PHQ8_Binary==1]['pd_prob'].values
non = df[df.PHQ8_Binary==0]['pd_prob'].values
u,p = mannwhitneyu(dep, non, alternative='two-sided')
auc_d = roc_auc_score(df.PHQ8_Binary.values, df.pd_prob.values)

print(f'\nDepressed (N={len(dep)})    : mean PD-prob = {dep.mean():.3f} (std {dep.std():.3f})')
print(f'Non-depressed (N={len(non)}): mean PD-prob = {non.mean():.3f} (std {non.std():.3f})')
print(f'Mann-Whitney U={u:.1f}  p={p:.4f}')
print(f'AUROC of PD-prob predicting depression: {auc_d:.4f}  (≈0.5 means no discrimination)')

v = ("PASSES: PD features in DAIC do not separate depression." if p>0.05 and abs(auc_d-0.5)<0.10
     else "PARTIAL discrimination exists." if p>0.05
     else "CONCERN: PD-prob differs between groups.")
print(f'\nVERDICT: {v}')

pd.DataFrame([{
    'uci_pd_loso_auc_5nl':round(auc_loso,4),
    'dep_pdprob_mean':round(dep.mean(),4), 'nondep_pdprob_mean':round(non.mean(),4),
    'mw_u':round(u,2), 'mw_p':round(p,4),
    'auc_pdprob_to_dep':round(auc_d,4), 'verdict':v
}]).to_csv(RES/'reverse_attribution.csv', index=False)
print(f'\nSaved -> reverse_attribution.csv')
