"""
Extract respiratory features from DAIC-WOZ audio files.
Applies trained Coswara respiratory model to get pseudo-labels.
These become Task 2 in the multi-task learning framework.

Strategy: extract first 60 seconds of each session (sufficient for
respiratory biomarkers, avoids memory issues with 40-min files).
"""

import numpy as np
import pandas as pd
import librosa
import joblib
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from tqdm import tqdm
import parselmouth
from parselmouth.praat import call

BASE    = Path.home() / 'Desktop/PULSE_IQ_AI'
AUDIO_DIR = BASE / 'data/raw/daic_woz'
RESULTS   = BASE / 'data/features'

RESP_FEATURES = [f'mfcc_{i+1}' for i in range(13)] + \
                ['pitch','spectral_centroid','zcr','jitter','shimmer','hnr']

def extract_resp_features(wav_path, duration=60, sr=16000):
    """Extract 19 respiratory features from first `duration` seconds."""
    try:
        y, _ = librosa.load(wav_path, sr=sr, mono=True, duration=duration)
        if len(y) < sr * 5:  # need at least 5 seconds
            return None

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = mfccs.mean(axis=1)

        # Pitch
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=600, sr=sr)
        f0_clean  = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        pitch_val = float(f0_clean.mean()) if len(f0_clean) > 0 else 0.0

        # Spectral + ZCR
        sc  = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
        zcr = float(librosa.feature.zero_crossing_rate(y).mean())

        # Praat features
        snd = parselmouth.Sound(y, sampling_frequency=sr)
        try:
            pp      = call(snd, 'To PointProcess (periodic, cc)', 50, 600)
            jitter  = call(pp, 'Get jitter (local)', 0,0,0.0001,0.02,1.3)
            shimmer = call([snd,pp],'Get shimmer (local)',0,0,0.0001,0.02,1.3,1.6)
            harm    = call(snd, 'To Harmonicity (cc)', 0.01, 50, 0.1, 1.0)
            hnr     = call(harm, 'Get mean', 0, 0)
        except Exception:
            jitter, shimmer, hnr = 0.01, 0.05, 15.0

        feats = {f'mfcc_{i+1}': float(mfcc_means[i]) for i in range(13)}
        feats.update({
            'pitch': pitch_val, 'spectral_centroid': sc,
            'zcr': zcr, 'jitter': float(jitter),
            'shimmer': float(shimmer), 'hnr': float(hnr)
        })
        return feats

    except Exception as e:
        print(f'Error {wav_path.name}: {e}')
        return None


if __name__ == '__main__':
    # Load respiratory model
    resp_model = joblib.load(BASE/'models/respiratory_model.pkl')
    print(f'Respiratory model loaded.')

    # Load DAIC-WOZ label file for participant IDs
    train_df = pd.read_csv(BASE/'data/raw/daic_woz/train_split_Depression_AVEC2017.csv')
    dev_df   = pd.read_csv(BASE/'data/raw/daic_woz/dev_split_Depression_AVEC2017.csv')
    all_pids = set(train_df['Participant_ID'].tolist() +
                   dev_df['Participant_ID'].tolist())
    print(f'Target sessions: {len(all_pids)}')

    # Find all audio files
    wav_files = sorted(AUDIO_DIR.glob('*_AUDIO.wav'))
    wav_map   = {int(w.stem.split('_')[0]): w for w in wav_files}
    print(f'Audio files found: {len(wav_files)}')

    results = []
    for pid in tqdm(sorted(all_pids), desc='Extracting respiratory features'):
        if pid not in wav_map:
            continue
        wav_path = wav_map[pid]
        feats    = extract_resp_features(wav_path, duration=60)
        if feats is None:
            continue

        # Get respiratory pseudo-label probability
        X = np.array([[feats[f] for f in RESP_FEATURES]])
        resp_prob = float(resp_model.predict_proba(X)[0][1])

        results.append({
            'participant_id'   : pid,
            'resp_prob'        : round(resp_prob, 6),
            'resp_pseudo_label': int(resp_prob >= 0.5),
            **{f: feats[f] for f in RESP_FEATURES}
        })

    df = pd.DataFrame(results)
    out = RESULTS / 'daic_woz_respiratory_pseudolabels.csv'
    df.to_csv(out, index=False)

    print(f'\nDone! {len(df)} sessions processed')
    print(f'Pseudo-label distribution:')
    print(f'  Respiratory risk >= 0.5 : {df.resp_pseudo_label.sum()}')
    print(f'  Respiratory risk <  0.5 : {(df.resp_pseudo_label==0).sum()}')
    print(f'  Mean respiratory prob   : {df.resp_prob.mean():.4f}')
    print(f'Saved -> {out}')
