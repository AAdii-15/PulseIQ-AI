"""
Properly completes the wav2vec speaker identity analysis.
Tests whether wav2vec encodes more speaker identity (gender as proxy)
than COVAREP, and whether removing top-PCA components helps depression AUROC.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

BASE = Path.home()/'Desktop/PULSE_IQ_AI'
RES  = BASE/'results/metrics'

cov = pd.read_csv(BASE/'data/features/daic_woz_covarep_allframes.csv')
w2v = pd.read_csv(BASE/'data/features/daic_woz_wav2vec_features.csv')
cov_f = [c for c in cov.columns if c.endswith('_mean') or c.endswith('_std')]
w2v_f = [c for c in w2v.columns if c.startswith('w2v_')]
m = cov[['participant_id','Gender','PHQ8_Binary']+cov_f].merge(
    w2v[['participant_id']+w2v_f], on='participant_id')
print(f'Merged: {len(m)} sessions  (gender 0/1 counts: {dict(m.Gender.value_counts())})')

skf = StratifiedKFold(5, shuffle=True, random_state=42)

def gender_auc(X, y, label):
    p = Pipeline([('imp',SimpleImputer(strategy='median')),
                  ('sc',StandardScaler()),
                  ('pca',PCA(n_components=min(30, X.shape[1]-1), random_state=42)),
                  ('clf',LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))])
    pr = cross_val_predict(p, X, y, cv=skf, method='predict_proba')[:,1]
    a = roc_auc_score(y, pr)
    print(f'  {label:24s} gender AUROC = {a:.4f}')
    return a

print('\n=== Gender prediction (proxy for speaker identity) ===')
yg = m.Gender.values
a_w = gender_auc(np.nan_to_num(m[w2v_f].values), yg, 'wav2vec 2.0 (768d)')
a_c = gender_auc(np.nan_to_num(m[cov_f].values), yg, 'COVAREP (146d)')
print(f'\nRatio (w2v/COVAREP) : {a_w/a_c:.3f}  (>1 → wav2vec carries more identity)')

print('\n=== Depression AUROC after removing top-PCA components from wav2vec ===')
trn = pd.read_csv(BASE/'data/raw/daic_woz/train_split_Depression_AVEC2017.csv').rename(columns={'Participant_ID':'participant_id'})
dev = pd.read_csv(BASE/'data/raw/daic_woz/dev_split_Depression_AVEC2017.csv').rename(columns={'Participant_ID':'participant_id'})
T = m[m.participant_id.isin(trn.participant_id)]
D = m[m.participant_id.isin(dev.participant_id)]
Xtr, ytr = np.nan_to_num(T[w2v_f].values), T.PHQ8_Binary.values
Xdv, ydv = np.nan_to_num(D[w2v_f].values), D.PHQ8_Binary.values

sc = StandardScaler().fit(Xtr)
Xtr_s, Xdv_s = sc.transform(Xtr), sc.transform(Xdv)
pca = PCA(n_components=50, random_state=42).fit(Xtr_s)
Xtr_p, Xdv_p = pca.transform(Xtr_s), pca.transform(Xdv_s)

depr_aucs={}
for skip in [0,1,2,3,5,10]:
    svm = SVC(kernel='rbf', probability=True, random_state=42).fit(Xtr_p[:,skip:], ytr)
    pr  = svm.predict_proba(Xdv_p[:,skip:])[:,1]
    a   = roc_auc_score(ydv, pr)
    depr_aucs[f'skip_{skip}']=round(a,4)
    print(f'  Skip top-{skip:2d} PCs → depression AUROC = {a:.4f}')

pd.DataFrame([{
    'w2v_gender_auc':round(a_w,4),
    'covarep_gender_auc':round(a_c,4),
    'ratio':round(a_w/a_c,4),
    **depr_aucs,
    'interpretation':('wav2vec encodes more speaker identity than COVAREP'
                      if a_w > a_c
                      else 'no clear identity-leakage difference')
}]).to_csv(RES/'wav2vec_speaker_identity_complete.csv', index=False)
print(f'\nSaved -> wav2vec_speaker_identity_complete.csv')
