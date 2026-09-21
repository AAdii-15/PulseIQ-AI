"""
Participant-level leakage check for COVID-19 (Coswara), mirroring the
PD leakage-quantification methodology (LOSO vs fold-CV).

Built after discovering the paper's Limitations claim "Coswara subject
identifiers are absent" is factually wrong -- the actual data file
(voice_dataset_labeled_full.csv) has a real user_id column, 2,709
unique participants across 5,411 recordings (~2.0 recordings/person).
This recomputes COVID-19 AUROC under both protocols on the SAME data
and reports the real gap, exactly as already done for PD, rather than
the indirect "5-20% random removal" sensitivity-analysis proxy that
was built on the incorrect assumption that no participant ID existed.

Also checks the N=5,238 (paper) vs N=5,411 (raw file) row-count gap.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

BASE = Path.home() / 'Desktop/PULSE_IQ_AI'
df = pd.read_csv(BASE / 'data/raw/coswara/voice_dataset_labeled_full.csv')

print(f'Raw file: {len(df)} rows, {df.user_id.nunique()} unique participants')
print(f'Paper states N=5,238 -- checking what filtering explains the {len(df)-5238} row gap...')
print(f'covid_status value counts:')
print(df.covid_status.value_counts())
print(f'\\nlabel value counts:')
print(df.label.value_counts())
print(f'Rows with any missing feature values: {df.isnull().any(axis=1).sum()}')

feat_cols = ['pitch', 'spectral_centroid', 'zcr', 'jitter', 'shimmer', 'hnr'] + \
            [f'mfcc_{i}' for i in range(1, 14)]
X = df[feat_cols].values
y = df['label'].values
groups = df['user_id'].values

def make_rf(seed=42):
    return Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc', StandardScaler()),
                      ('clf', RandomForestClassifier(n_estimators=500, random_state=seed,
                                                      class_weight='balanced', n_jobs=-1))])

print('\\nRunning recording-level 5-fold CV (current paper methodology)...')
prob_recording = cross_val_predict(make_rf(), X, y,
                                    cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                    method='predict_proba')[:, 1]
auc_recording = roc_auc_score(y, prob_recording)

print('Running participant-grouped 5-fold CV (zero participant overlap)...')
prob_grouped = cross_val_predict(make_rf(), X, y,
                                  cv=StratifiedGroupKFold(5, shuffle=True, random_state=42),
                                  groups=groups, method='predict_proba')[:, 1]
auc_grouped = roc_auc_score(y, prob_grouped)

print('\\n' + '=' * 70)
print(' COVID-19 PARTICIPANT-LEVEL LEAKAGE CHECK')
print('=' * 70)
print(f'  Recording-level 5-fold CV (current paper): AUROC = {auc_recording:.4f}')
print(f'  Participant-grouped 5-fold CV (correct):    AUROC = {auc_grouped:.4f}')
print(f'  Gap (leakage inflation):                     {auc_recording - auc_grouped:+.4f}')
print('=' * 70)

pd.DataFrame([{
    'protocol': 'recording_level', 'auroc': round(auc_recording, 4)
}, {
    'protocol': 'participant_grouped', 'auroc': round(auc_grouped, 4)
}]).to_csv(BASE / 'results/metrics/covid_participant_leakage.csv', index=False)
print(f"\\nSaved -> results/metrics/covid_participant_leakage.csv")
