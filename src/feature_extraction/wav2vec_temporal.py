"""
wav2vec 2.0 Frame-Level Feature Extraction
============================================
Instead of mean-pooling, stores the full frame sequence.
Each session → variable-length sequence of 768-dim frames.
Saved as numpy arrays for efficient loading.

Key difference from wav2vec_features.py:
  - Keeps temporal structure (T × 768 per session)
  - Enables attention pooling over clinically relevant frames
  - Addresses the temporal information loss identified in analysis
"""

import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import librosa
from tqdm import tqdm

BASE      = Path.home() / 'Desktop/PULSE_IQ_AI'
AUDIO_DIR = BASE / 'data/raw/daic_woz'
OUT_DIR   = BASE / 'data/features/wav2vec_temporal'
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE    = 'mps' if torch.backends.mps.is_available() else 'cpu'
CHUNK_SEC = 30      # process in 30-sec chunks
MAX_CHUNKS = 8      # max 4 minutes per session (240 sec) — captures full interview context

print(f'Device: {DEVICE}')
print(f'Strategy: {MAX_CHUNKS} chunks × {CHUNK_SEC}s = {MAX_CHUNKS*CHUNK_SEC}s per session')

processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
model     = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h').to(DEVICE)
model.eval()
print('wav2vec 2.0 loaded.')

def extract_temporal(wav_path, sr=16000):
    """
    Extract frame-level wav2vec embeddings.
    Returns: list of chunk embeddings, each shape (T_i, 768)
    """
    y, _ = librosa.load(wav_path, sr=sr, mono=True,
                        duration=CHUNK_SEC * MAX_CHUNKS)
    chunk_size = sr * CHUNK_SEC
    chunks     = []

    for i in range(0, len(y), chunk_size):
        chunk = y[i:i+chunk_size]
        if len(chunk) < sr * 2:  # skip chunks < 2 seconds
            continue

        inputs = processor(chunk, sampling_rate=sr,
                          return_tensors='pt', padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs)

        # Frame-level: (1, T, 768) → (T, 768)
        frames = out.last_hidden_state.squeeze(0).cpu().numpy()
        chunks.append(frames)

        if len(chunks) >= MAX_CHUNKS:
            break

    return chunks

if __name__ == '__main__':
    train_df = pd.read_csv(BASE/'data/raw/daic_woz/train_split_Depression_AVEC2017.csv')
    dev_df   = pd.read_csv(BASE/'data/raw/daic_woz/dev_split_Depression_AVEC2017.csv')
    all_pids = train_df['Participant_ID'].tolist() + dev_df['Participant_ID'].tolist()
    wav_map  = {int(w.stem.split('_')[0]): w
                for w in AUDIO_DIR.glob('*_AUDIO.wav')}

    label_map = {}
    for _, row in pd.concat([train_df, dev_df]).iterrows():
        label_map[row['Participant_ID']] = int(row['PHQ8_Binary'])

    processed = []
    for pid in tqdm(sorted(all_pids), desc='Extracting temporal features'):
        out_path = OUT_DIR / f'{pid}_temporal.npy'
        if out_path.exists():  # resume
            processed.append(pid)
            continue
        if pid not in wav_map:
            continue
        try:
            chunks = extract_temporal(wav_map[pid])
            if chunks:
                # Stack all chunks: (total_frames, 768)
                stacked = np.vstack(chunks)
                np.save(out_path, stacked)
                processed.append(pid)
        except Exception as e:
            print(f'Error {pid}: {e}')

    print(f'\nExtracted: {len(processed)}/{len(all_pids)} sessions')
    print(f'Saved to: {OUT_DIR}')

    # Quick stats
    shapes = [np.load(OUT_DIR/f'{p}_temporal.npy').shape
              for p in processed[:5]]
    print(f'Sample shapes (frames × 768): {shapes}')
