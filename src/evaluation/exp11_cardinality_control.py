import numpy as np, pandas as pd, warnings, sys
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
try:
    import shap
except ImportError:
    import subprocess; subprocess.check_call([sys.executable,'-m','pip','install','shap','--quiet'])
    import shap

BASE = Path.home()/'Desktop/PULSE_IQ_AI'

df_shared  = pd.read_csv(BASE/'data/features/daic_woz_shared_features.csv')
df_covarep = pd.read_csv(BASE/'data/features/daic_woz_covarep_allframes.csv')
trn = pd.read_csv(BASE/'data/raw/daic_woz/train_split_Depression_AVEC2017.csv'
                  ).rename(columns={'Participant_ID':'participant_id'})
dev = pd.read_csv(BASE/'data/raw/daic_woz/dev_split_Depression_AVEC2017.csv'
                  ).rename(columns={'Participant_ID':'participant_id'})
labels = pd.concat([trn,dev])[['participant_id','PHQ8_Binary']]

# Strip any PHQ columns from feature files before merging to avoid collision
drop_cols = [c for c in df_covarep.columns if 'phq' in c.lower() or c=='PHQ8_Binary']
df_covarep_clean = df_covarep.drop(columns=drop_cols, errors='ignore')
drop_cols2 = [c for c in df_shared.columns if 'phq' in c.lower() or c=='PHQ8_Binary']
df_shared_clean = df_shared.drop(columns=drop_cols2, errors='ignore')

df = df_shared_clean.merge(df_covarep_clean, on='participant_id', how='inner', suffixes=('','_cov'))
df = df.merge(labels, on='participant_id', how='inner')
print(f'Merged: {df.shape}  |  PHQ8_Binary present: {"PHQ8_Binary" in df.columns}')

NL_COLS   = ['shared_ppe','shared_rpde','shared_dfa','shared_spread1','shared_spread2']
MFCC_COLS = [c for c in df.columns if 'mfcc' in c.lower() and c in df_shared_clean.columns]
COVAREP_COLS = [c for c in df_covarep_clean.columns if c != 'participant_id']
print(f'Group sizes: NL={len(NL_COLS)}, MFCC={len(MFCC_COLS)}, COVAREP={len(COVAREP_COLS)}')

train_mask = df.participant_id.isin(trn.participant_id)
df_train = df[train_mask]
print(f'Train: {len(df_train)} sessions')

TARGET = len(NL_COLS)  # 5
N_REPS = 200
rng = np.random.default_rng(42)
rows = []

print(f'\nMatched-size analysis: {N_REPS} reps × 5 NL + 5 MFCC + 5 COVAREP...')
for rep in range(N_REPS):
    mfcc_s = list(rng.choice(MFCC_COLS, TARGET, replace=False))
    cov_s  = list(rng.choice(COVAREP_COLS, TARGET, replace=False))
    cols   = NL_COLS + mfcc_s + cov_s

    X = df_train[cols].fillna(df_train[cols].median()).values
    y = df_train['PHQ8_Binary'].values

    pipe = Pipeline([('imp',SimpleImputer(strategy='median')),
                     ('sc', StandardScaler()),
                     ('clf',RandomForestClassifier(n_estimators=200,random_state=rep,
                                                    class_weight='balanced',n_jobs=-1))])
    pipe.fit(X, y)
    X_t = pipe[:-1].transform(X)

    expl = shap.TreeExplainer(pipe['clf'])
    sv   = expl.shap_values(X_t, check_additivity=False)
    if isinstance(sv, list):  sv = sv[1]
    if sv.ndim == 3:          sv = sv[:,:,1]

    a   = np.abs(sv).mean(axis=0)
    nl  = a[:TARGET].sum()
    mf  = a[TARGET:2*TARGET].sum()
    cv  = a[2*TARGET:].sum()
    tot = nl + mf + cv

    rows.append({'rep':rep,'nl_pct':100*nl/tot,'mfcc_pct':100*mf/tot,'covarep_pct':100*cv/tot})
    if (rep+1) % 10 == 0: print(f'  {rep+1}/{N_REPS}')

dfr = pd.DataFrame(rows)
print(f'\n=== (A) MATCHED-SIZE GROUP ATTRIBUTION (5+5+5, n={N_REPS} reps) ===')
for col, name in [('covarep_pct','COVAREP (clinical glottal)'),
                   ('mfcc_pct','MFCC (respiratory-type)'),
                   ('nl_pct','Nonlinear (PD-type)')]:
    v = dfr[col]
    print(f'  {name:32s}: {v.mean():5.1f}% ± {v.std():4.1f}%  [{v.quantile(0.025):4.1f}–{v.quantile(0.975):4.1f}]')

print(f'\n=== (B) PER-FEATURE NORMALISATION (original full feature space) ===')
orig = {'COVAREP':(74.2,len(COVAREP_COLS)),'MFCC':(23.2,len(MFCC_COLS)),'Nonlinear':(2.6,len(NL_COLS))}
for g,(att,n) in orig.items():
    print(f'  {g:10s}: {att:5.1f}% / {n:3d} features = {att/n:.3f}% per feature')

cov_m = dfr.covarep_pct.mean(); mfcc_m = dfr.mfcc_pct.mean(); nl_m = dfr.nl_pct.mean()
print(f'\n=== VERDICT ===')
print(f'  Matched: COVAREP={cov_m:.1f}%, MFCC={mfcc_m:.1f}%, NL={nl_m:.1f}%')
if cov_m > mfcc_m and cov_m > nl_m:
    print(f'  COVAREP leads even at matched size -> NOT a pure cardinality artefact.')
else:
    print(f'  COVAREP does not lead at matched size -> original was cardinality-confounded.')

dfr.to_csv(BASE/'results/metrics/cardinality_control.csv', index=False)
print(f'\nSaved -> cardinality_control.csv')
print(f'\n=> REPORT: COVAREP={cov_m:.1f}%±{dfr.covarep_pct.std():.1f}%, '
      f'MFCC={mfcc_m:.1f}%±{dfr.mfcc_pct.std():.1f}%, '
      f'NL={nl_m:.1f}%±{dfr.nl_pct.std():.1f}%')
