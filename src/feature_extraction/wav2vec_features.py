"""
wav2vec 2.0 Embedding Extractor for DAIC-WOZ
----------------------------------------------
Extracts 768-dim session-level embeddings from raw audio.
Uses Apple Silicon MPS for acceleration.

Strategy:
  - Load raw WAV (16kHz mono)
  - Process in 30-second chunks to manage memory
  - Average pool hidden states across time and chunks
  - Result: 768-dim vector per session
  - Save incrementally (resume-safe)
"""

import torch
import soundfile as sf
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import warnings
warnings.filterwarnings("ignore")

BASE      = Path.home() / "Desktop" / "PULSE_IQ_AI"
DAIC_DIR  = BASE / "data/raw/daic_woz"
OUT_PATH  = BASE / "data/features/daic_woz_wav2vec_features.csv"
MODEL_ID  = "facebook/wav2vec2-base"
CHUNK_SEC = 30       # seconds per chunk
SR        = 16000    # wav2vec expects 16kHz
CHUNK_LEN = CHUNK_SEC * SR


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(device):
    print(f"Loading {MODEL_ID} on {device}...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model     = Wav2Vec2Model.from_pretrained(MODEL_ID)
    model     = model.to(device)
    model.eval()
    print("Model loaded.\n")
    return processor, model


def extract_embedding(wav_path, processor, model, device):
    """
    Extract 768-dim embedding from a WAV file.
    Processes in 30-sec chunks, returns mean-pooled embedding.
    """
    audio, sr = sf.read(str(wav_path), dtype="float32")

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)

    # Split into chunks
    chunks = [audio[i:i+CHUNK_LEN]
              for i in range(0, len(audio), CHUNK_LEN)
              if len(audio[i:i+CHUNK_LEN]) > SR]  # skip < 1 sec

    if not chunks:
        chunks = [audio]

    chunk_embeddings = []
    with torch.no_grad():
        for chunk in chunks:
            inputs = processor(
                chunk,
                sampling_rate=SR,
                return_tensors="pt",
                padding=True
            )
            input_values = inputs.input_values.to(device)

            # Handle MPS float32 requirement
            if device.type == "mps":
                input_values = input_values.float()

            outputs       = model(input_values)
            hidden_states = outputs.last_hidden_state  # (1, T, 768)
            embedding     = hidden_states.mean(dim=1).squeeze()  # (768,)
            chunk_embeddings.append(embedding.cpu().numpy())

    return np.mean(chunk_embeddings, axis=0)  # (768,)


def build_wav2vec_dataset():
    device = get_device()
    print(f"Device: {device}")

    # Load labels
    train_df = pd.read_csv(DAIC_DIR/"train_split_Depression_AVEC2017.csv")
    dev_df   = pd.read_csv(DAIC_DIR/"dev_split_Depression_AVEC2017.csv")
    labels   = pd.concat([train_df, dev_df]).rename(
                    columns={"Participant_ID": "participant_id"})
    labeled_pids = set(labels["participant_id"].values)

    # Resume support — skip already processed sessions
    processed = set()
    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH)
        processed = set(existing["participant_id"].values)
        print(f"Resuming — {len(processed)} sessions already done.")

    processor, model = load_model(device)

    wav_files = sorted(DAIC_DIR.glob("*_AUDIO.wav"))
    todo      = [f for f in wav_files
                 if int(f.stem.split("_")[0]) in labeled_pids
                 and int(f.stem.split("_")[0]) not in processed]

    print(f"Sessions to process: {len(todo)}\n")

    for i, wav_path in enumerate(todo):
        pid = int(wav_path.stem.split("_")[0])

        try:
            emb = extract_embedding(wav_path, processor, model, device)

            row = {f"w2v_{j}": emb[j] for j in range(len(emb))}
            row["participant_id"] = pid

            row_df = pd.DataFrame([row])
            write_header = not OUT_PATH.exists()
            row_df.to_csv(OUT_PATH, mode="a",
                          header=write_header, index=False)

            duration_min = i / max(1, len(todo))
            print(f"  [{i+1:3d}/{len(todo)}] Session {pid} — "
                  f"embedding shape: {emb.shape} | "
                  f"{(i+1)/len(todo)*100:.1f}% done")

        except Exception as e:
            print(f"  [WARN] Session {pid} failed: {e}")

    # Merge with labels
    print("\nMerging with PHQ-8 labels...")
    features_df = pd.read_csv(OUT_PATH)
    merged      = features_df.merge(labels, on="participant_id", how="inner")
    merged.to_csv(OUT_PATH, index=False)

    print(f"\n✅ Done!")
    print(f"   Sessions : {len(merged)}")
    print(f"   Features : {len([c for c in merged.columns if c.startswith('w2v_')])}")
    print(f"   Depressed: {merged['PHQ8_Binary'].sum()}")
    print(f"   Saved → {OUT_PATH}")
    return merged


if __name__ == "__main__":
    build_wav2vec_dataset()
