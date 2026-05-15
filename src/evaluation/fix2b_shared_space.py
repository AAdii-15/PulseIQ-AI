"""
Fix 2B: Shared Feature Space Orthogonality Analysis
=====================================================
Extracts BOTH MFCC (respiratory-type) AND nonlinear (PD-type) features
from the SAME DAIC-WOZ audio recordings, alongside COVAREP features.

Trains depression model on the shared feature space.
Runs SHAP → identifies which feature TYPE dominates for depression.

Comparison:
  PD model SHAP        → nonlinear features dominate
  COVID-19 model SHAP  → MFCC features dominate
  Depression model SHAP → which type? (COVAREP/MFCC/nonlinear?)

This tests orthogonality on the SAME recordings — fully defensible.
"""

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
import librosa
import parselmouth
from parselmouth.praat import call
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import shap

BASE     = Path.home() / 'Desktop/PULSE_IQ_AI'
AUDIO    = BASE / 'data/raw/daic_woz'
RESULTS  = BASE / 'results/metrics'
FIGS     = BASE / 'results/figures'

# ── DFA (pure numpy) ─────────────────────────────────────────────────────────

def _dfa(x):
    x = np.array(x, dtype=float); N = len(x)
    if N < 20: return 0.6
    y = np.cumsum(x - x.mean())
    nvals = np.unique(np.floor(
        np.logspace(np.log10(4), np.log10(max(N//4,5)), 14)).astype(int))
    F = []
    for n in nvals:
        if n < 4 or n > N//2: continue
        rms = []
        for s in range(0, N-n, n):
            seg = y[s:s+n]; xi = np.arange(n)
            p = np.polyfit(xi, seg, 1)
            rms.append(np.sqrt(np.mean((seg - np.polyval(p, xi))**2)))
        if rms: F.append(np.mean(rms))
    if len(F) < 2: return 0.6
    try:
        return float(np.clip(
            np.polyfit(np.log(nvals[:len(F)]),np.log(F),1)[0], 0, 2))
    except: return 0.6


# ── Feature extraction from audio ────────────────────────────────────────────

def extract_shared_features(wav_path, sr=16000, duration=60):
    """
    Extract both MFCC (respiratory-type) AND nonlinear (PD-type) features
    from the same audio file.
    Returns dict with 13 MFCCs + 6 nonlinear = 19 shared features.
    """
    y, _ = librosa.load(wav_path, sr=sr, mono=True, duration=duration)
    if len(y) < sr * 5:
        return None

    feats = {}

    # ── MFCC features (respiratory-type) ─────────────────────────────────────
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        feats[f'shared_mfcc_{i+1}_mean'] = float(mfccs[i].mean())
        feats[f'shared_mfcc_{i+1}_std']  = float(mfccs[i].std())

    # Spectral features
    feats['shared_zcr_mean']    = float(librosa.feature.zero_crossing_rate(y).mean())
    feats['shared_sc_mean']     = float(librosa.feature.spectral_centroid(y=y,sr=sr).mean())

    # ── Nonlinear features (PD-type) ─────────────────────────────────────────
    snd   = parselmouth.Sound(y, sampling_frequency=sr)
    try:
        pitch = call(snd, 'To Pitch', 0.01, 50, 600)
        f0    = pitch.selected_array['frequency']
        f0_v  = f0[f0 > 0]
    except:
        f0_v = np.array([])

    if len(f0_v) >= 20:
        # PPE
        periods = 1.0 / f0_v
        pdiff   = np.diff(periods)
        if len(pdiff) > 1:
            h, edges = np.histogram(pdiff, bins=20, density=True)
            h = h[h > 0]
            bw = (edges[1]-edges[0]) + 1e-10
            ppe = float(np.clip(-np.sum(h*np.log2(h+1e-10))*bw, 0, 5))
        else:
            ppe = 0.15

        # RPDE
        f0_n = (f0_v - f0_v.mean()) / (f0_v.std() + 1e-8)
        ac   = np.correlate(f0_n, f0_n, mode='full')[len(f0_n)-1:]
        ac   = ac / (ac[0] + 1e-8)
        peaks = [ac[i] for i in range(1, min(len(ac)-1,300))
                 if ac[i] > ac[i-1] and ac[i] > ac[i+1] and ac[i] > 0]
        rpde = float(np.clip(
            -np.sum(np.array(peaks[:10]+[1e-8])/sum(peaks[:10]+[1e-8]) *
                    np.log(np.array(peaks[:10]+[1e-8])/sum(peaks[:10]+[1e-8])+1e-8)), 0, 1)) \
            if len(peaks) > 2 else 0.45

        dfa     = _dfa(f0_v)
        log_f0  = np.log(f0_v + 1e-8)
        spread1 = float(np.polyfit(np.arange(len(log_f0)), log_f0, 1)[1])
        spread2 = float(np.std(log_f0 - np.mean(log_f0)))

        feats['shared_ppe']     = ppe
        feats['shared_rpde']    = rpde
        feats['shared_dfa']     = dfa
        feats['shared_spread1'] = spread1
        feats['shared_spread2'] = spread2
    else:
        for k in ['shared_ppe','shared_rpde','shared_dfa',
                  'shared_spread1','shared_spread2']:
            feats[k] = 0.0

    # HNR (shared)
    try:
        harm = call(snd, 'To Harmonicity (cc)', 0.01, 50, 0.1, 1.0)
        feats['shared_hnr'] = float(call(harm, 'Get mean', 0, 0))
    except:
        feats['shared_hnr'] = 15.0

    return feats


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Load labels
    df       = pd.read_csv(BASE/'data/features/daic_woz_covarep_allframes.csv')
    train_df = pd.read_csv(BASE/'data/raw/daic_woz/train_split_Depression_AVEC2017.csv').rename(columns={'Participant_ID':'participant_id'})
    dev_df   = pd.read_csv(BASE/'data/raw/daic_woz/dev_split_Depression_AVEC2017.csv').rename(columns={'Participant_ID':'participant_id'})

    # Load or extract shared features
    cache = BASE/'data/features/daic_woz_shared_features.csv'
    if cache.exists():
        print('Loading cached shared features...')
        shared_df = pd.read_csv(cache)
    else:
        print('Extracting shared features from DAIC-WOZ audio...')
        wav_map = {int(w.stem.split('_')[0]): w
                   for w in AUDIO.glob('*_AUDIO.wav')}
        all_pids = df.participant_id.tolist()
        records  = []
        for pid in tqdm(all_pids, desc='Extracting'):
            if pid not in wav_map: continue
            feats = extract_shared_features(wav_map[pid])
            if feats:
                feats['participant_id'] = pid
                records.append(feats)
        shared_df = pd.DataFrame(records)
        shared_df.to_csv(cache, index=False)
        print(f'Saved {len(shared_df)} sessions')

    print(f'Shared features: {len(shared_df)} sessions')

    # Merge with COVAREP features
    cov_feats = [c for c in df.columns if c.endswith('_mean') or c.endswith('_std')]
    shared_feats = [c for c in shared_df.columns if c.startswith('shared_')]

    merged = df[['participant_id','PHQ8_Binary']+cov_feats].merge(
             shared_df[['participant_id']+shared_feats], on='participant_id')
    print(f'Merged: {len(merged)} sessions | '
          f'COVAREP: {len(cov_feats)} | Shared: {len(shared_feats)}')

    # Split
    train = merged[merged.participant_id.isin(train_df.participant_id)]
    dev   = merged[merged.participant_id.isin(dev_df.participant_id)]
    all_feats = cov_feats + shared_feats

    X_tr = train[all_feats].values; y_tr = train.PHQ8_Binary.values
    X_dv = dev[all_feats].values;   y_dv = dev.PHQ8_Binary.values

    # Train RF on shared space (for SHAP)
    pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc',  StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=500, random_state=42,
                                       class_weight='balanced', n_jobs=-1))
    ])
    pipe.fit(X_tr, y_tr)
    y_prob = pipe.predict_proba(X_dv)[:,1]
    auc = roc_auc_score(y_dv, y_prob)
    print(f'\nDepression RF (shared space): AUROC={auc:.4f}')

    # SHAP analysis
    print('Running TreeSHAP on shared feature space...')
    rf       = pipe.named_steps['clf']
    X_tr_t   = pipe[:-1].transform(X_tr)
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X_tr_t)
    sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals[:,:,1]

    mean_shap = np.abs(sv).mean(axis=0)
    top_feats = sorted(zip(all_feats, mean_shap),
                       key=lambda x: x[1], reverse=True)[:20]

    print('\nTop 20 SHAP features for Depression (shared feature space):')
    print(f'{"Feature":<35} {"Mean|SHAP|":>10}  {"Type"}')
    print('-'*60)
    for feat, val in top_feats:
        if feat.startswith('shared_mfcc'):
            ftype = 'MFCC (respiratory-type)'
        elif any(feat.startswith(f'shared_{k}') for k in
                 ['ppe','rpde','dfa','spread']):
            ftype = 'NONLINEAR (PD-type)'
        elif feat.startswith('shared_'):
            ftype = 'shared (other)'
        else:
            ftype = 'COVAREP (depression-type)'
        print(f'  {feat:<33} {val:>10.5f}  {ftype}')

    # Categorise SHAP importance by feature type
    mfcc_shap    = sum(v for f,v in zip(all_feats, mean_shap)
                       if 'shared_mfcc' in f)
    nonlinear_shap = sum(v for f,v in zip(all_feats, mean_shap)
                         if any(f.startswith(f'shared_{k}')
                                for k in ['ppe','rpde','dfa','spread']))
    covarep_shap = sum(v for f,v in zip(all_feats, mean_shap)
                       if f in cov_feats)
    total = mfcc_shap + nonlinear_shap + covarep_shap

    print(f'\n=== Feature Type SHAP Attribution (Depression) ===')
    print(f'COVAREP features  : {covarep_shap:.4f} ({covarep_shap/total*100:.1f}%)')
    print(f'MFCC features     : {mfcc_shap:.4f}    ({mfcc_shap/total*100:.1f}%)')
    print(f'Nonlinear features: {nonlinear_shap:.4f}    ({nonlinear_shap/total*100:.1f}%)')
    print()
    print('Comparison (from separate-dataset analysis):')
    print('  PD model       → nonlinear dominant (PPE, RPDE, DFA = top 3)')
    print('  COVID-19 model → MFCC dominant (mfcc_10, mfcc_6, mfcc_8 = top 3)')
    print(f'  Depression     → see above')
    print()
    print('If COVAREP/MFCC > nonlinear for depression:')
    print('→ Distinct feature attribution confirmed in SAME recordings')
    print('→ Fix 2B complete: orthogonality defensible')

    # Save figure
    fig, ax = plt.subplots(figsize=(10, 7))
    feat_names = [f for f,v in top_feats]
    feat_vals  = [v for f,v in top_feats]
    colors = []
    for f in feat_names:
        if 'shared_mfcc' in f:
            colors.append('#4CAF50')
        elif any(f.startswith(f'shared_{k}')
                 for k in ['ppe','rpde','dfa','spread']):
            colors.append('#F44336')
        else:
            colors.append('#9C27B0')

    bars = ax.barh(range(len(feat_names)), feat_vals, color=colors, alpha=0.85)
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels([f.replace('shared_','[S] ') for f in feat_names], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP| Value')
    ax.set_title('Depression Model SHAP (Shared Feature Space)\n'
                 'Purple=COVAREP, Green=MFCC(resp-type), Red=Nonlinear(PD-type)',
                 fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')

    from matplotlib.patches import Patch
    legend = [Patch(color='#9C27B0', label='COVAREP (depression-specific)'),
              Patch(color='#4CAF50', label='MFCC (respiratory-type)'),
              Patch(color='#F44336', label='Nonlinear (PD-type)')]
    ax.legend(handles=legend, loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGS/'fig5_shared_shap.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved fig5_shared_shap.png')

    # Save results
    pd.DataFrame([{
        'covarep_shap': round(covarep_shap,4),
        'mfcc_shap': round(mfcc_shap,4),
        'nonlinear_shap': round(nonlinear_shap,4),
        'depression_auroc_shared': round(auc,4)
    }]).to_csv(RESULTS/'fix2b_shared_space_shap.csv', index=False)
    print(f'Saved fix2b_shared_space_shap.csv')
