"""
Unified Data Loaders for PulseIQ AI
-------------------------------------
Loads all 3 datasets into a standard format:
  X : feature DataFrame
  y : binary label Series
  meta : metadata (subject_id, dataset, etc.)
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path.home() / "Desktop" / "PULSE_IQ_AI"


# ── Parkinson's Disease ───────────────────────────────────────────────────────

def load_parkinsons():
    """
    UCI Parkinson's dataset.
    22 acoustic features, binary label: status (1=Parkinson's, 0=healthy).
    Subject-level split required (31 subjects, ~6 recordings each).
    """
    path = BASE / "data/raw/uci_parkinsons/parkinsons.csv"
    df = pd.read_csv(path)

    feature_cols = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
        "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ",
        "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
        "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
        "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
    ]

    X    = df[feature_cols].copy()
    y    = df["status"].copy()
    meta = pd.DataFrame({
        "subject_id": df["name"].apply(lambda x: x.split("_")[2]),
        "recording" : df["name"],
        "dataset"   : "uci_parkinsons"
    })

    print(f"[Parkinson's]  {len(X)} samples | "
          f"{y.sum()} positive ({y.mean()*100:.1f}%) | "
          f"{meta['subject_id'].nunique()} subjects | "
          f"{X.shape[1]} features")
    return X, y, meta


# ── Respiratory Abnormality ───────────────────────────────────────────────────

def load_respiratory():
    """
    Coswara dataset — pre-extracted 19 acoustic features.
    Binary label: 1=respiratory abnormality (COVID+), 0=healthy.
    """
    path = BASE / "data/raw/coswara/voice_dataset_labeled_full.csv"
    df   = pd.read_csv(path)

    feature_cols = [
        "pitch", "spectral_centroid", "zcr",
        "jitter", "shimmer", "hnr",
        "mfcc_1","mfcc_2","mfcc_3","mfcc_4","mfcc_5","mfcc_6","mfcc_7",
        "mfcc_8","mfcc_9","mfcc_10","mfcc_11","mfcc_12","mfcc_13"
    ]

    # Drop rows with nulls in features
    df = df.dropna(subset=feature_cols)

    X    = df[feature_cols].copy()
    y    = df["label"].copy()
    meta = pd.DataFrame({
        "subject_id": df["user_id"] if "user_id" in df.columns else df.index.astype(str),
        "dataset"   : "coswara"
    })

    print(f"[Respiratory]  {len(X)} samples | "
          f"{y.sum()} positive ({y.mean()*100:.1f}%) | "
          f"{X.shape[1]} features")
    return X, y, meta


# ── Depression ────────────────────────────────────────────────────────────────

def load_depression(include_phq8_details=False):
    """
    DAIC-WOZ dataset — COVAREP statistical features (146-dim).
    Binary label: PHQ8_Binary (1=depressed, PHQ8>=10).
    Also returns PHQ8_Score for regression experiments.
    Uses official AVEC 2017 train/dev split.
    """
    path = BASE / "data/features/daic_woz_covarep_features.csv"
    df   = pd.read_csv(path)

    # 146 acoustic feature columns (73 COVAREP x mean+std)
    feature_cols = [c for c in df.columns
                    if c.endswith("_mean") or c.endswith("_std")]

    X = df[feature_cols].copy()
    y = df["PHQ8_Binary"].copy()

    meta_cols = ["participant_id", "Gender", "PHQ8_Score"]
    if include_phq8_details:
        phq8_items = ["PHQ8_NoInterest","PHQ8_Depressed","PHQ8_Sleep",
                      "PHQ8_Tired","PHQ8_Appetite","PHQ8_Failure",
                      "PHQ8_Concentrating","PHQ8_Moving"]
        meta_cols += phq8_items

    meta = df[meta_cols].copy()
    meta["dataset"] = "daic_woz"

    print(f"[Depression]   {len(X)} sessions | "
          f"{y.sum()} positive ({y.mean()*100:.1f}%) | "
          f"{X.shape[1]} features | "
          f"PHQ8 range: {df['PHQ8_Score'].min()}-{df['PHQ8_Score'].max()}")
    return X, y, meta


# ── Load All ──────────────────────────────────────────────────────────────────

def load_all():
    print("=" * 55)
    print(" PulseIQ AI — Dataset Summary")
    print("=" * 55)
    pk_X,   pk_y,   pk_meta   = load_parkinsons()
    resp_X, resp_y, resp_meta = load_respiratory()
    dep_X,  dep_y,  dep_meta  = load_depression()
    print("=" * 55)
    return {
        "parkinsons" : (pk_X,   pk_y,   pk_meta),
        "respiratory": (resp_X, resp_y, resp_meta),
        "depression" : (dep_X,  dep_y,  dep_meta),
    }


if __name__ == "__main__":
    datasets = load_all()
