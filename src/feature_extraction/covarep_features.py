"""
DAIC-WOZ COVAREP Feature Extractor
Handles both 73 and 74-column variants in DAIC-WOZ dataset.
Uses iloc indexing — avoids all column name assignment errors.
"""

import numpy as np
import pandas as pd
from pathlib import Path

COVAREP_COLS = (
    ["F0", "VUV"] +
    ["NAQ", "QOQ", "H1H2", "PSP", "MDQ", "peakSlope", "Rd", "Rd_conf",
     "creak", "MFCC0"] +
    [f"MCEP_{i}" for i in range(24)] +
    [f"HMPDM_{i}" for i in range(24)] +
    [f"HMPDD_{i}" for i in range(13)]
)  # 73 features — some DAIC files have 74 cols (extra ignored)

N_FEATURES = len(COVAREP_COLS)  # 73


def extract_session_features(covarep_path: str) -> dict | None:
    try:
        df = pd.read_csv(covarep_path, header=None)
        n_cols = df.shape[1]
        n_use  = min(n_cols, N_FEATURES)  # use at most 73 cols

        # Filter voiced frames (VUV = col index 1)
        if n_cols > 1:
            voiced = df[df.iloc[:, 1] == 1]
            if len(voiced) < 10:
                voiced = df
        else:
            voiced = df

        features = {}
        for i in range(n_use):
            name = COVAREP_COLS[i]
            vals = voiced.iloc[:, i].replace([np.inf, -np.inf], np.nan).dropna()
            features[f"{name}_mean"] = float(vals.mean()) if len(vals) > 0 else 0.0
            features[f"{name}_std"]  = float(vals.std())  if len(vals) > 1 else 0.0

        # Zero-fill if file had fewer than 73 cols
        for name in COVAREP_COLS[n_use:]:
            features[f"{name}_mean"] = 0.0
            features[f"{name}_std"]  = 0.0

        return features

    except Exception as e:
        print(f"  [WARN] Failed on {covarep_path}: {e}")
        return None


def build_daic_dataset(daic_dir, train_labels, dev_labels, output_path):
    daic_dir    = Path(daic_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels_df = pd.concat([
        pd.read_csv(train_labels),
        pd.read_csv(dev_labels)
    ], ignore_index=True).rename(columns={"Participant_ID": "participant_id"})

    print(f"Labels: {len(labels_df)} sessions — "
          f"{labels_df['PHQ8_Binary'].sum()} depressed, "
          f"{(labels_df['PHQ8_Binary']==0).sum()} non-depressed")
    print(f"Feature vector size: {N_FEATURES * 2} (mean+std of {N_FEATURES} COVAREP features)\n")

    records, skipped = [], 0
    covarep_files = sorted(daic_dir.glob("*_COVAREP.csv"))
    print(f"Processing {len(covarep_files)} COVAREP files...")

    for i, fpath in enumerate(covarep_files):
        pid = int(fpath.stem.split("_")[0])
        if pid not in labels_df["participant_id"].values:
            continue
        feats = extract_session_features(str(fpath))
        if feats is None:
            skipped += 1
            continue
        feats["participant_id"] = pid
        records.append(feats)
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(covarep_files)} done...")

    print(f"\nExtracted: {len(records)} sessions, Skipped: {skipped}")

    features_df = pd.DataFrame(records)
    merged = features_df.merge(labels_df, on="participant_id", how="inner")

    print(f"\n✅ Final dataset    : {len(merged)} sessions x {len(merged.columns)} columns")
    print(f"   Depressed        : {merged['PHQ8_Binary'].sum()}")
    print(f"   Non-depressed    : {(merged['PHQ8_Binary']==0).sum()}")
    merged.to_csv(output_path, index=False)
    print(f"   Saved → {output_path}")
    return merged


if __name__ == "__main__":
    BASE = Path.home() / "Desktop" / "PULSE_IQ_AI"
    build_daic_dataset(
        daic_dir     = BASE / "data/raw/daic_woz",
        train_labels = BASE / "data/raw/daic_woz/train_split_Depression_AVEC2017.csv",
        dev_labels   = BASE / "data/raw/daic_woz/dev_split_Depression_AVEC2017.csv",
        output_path  = BASE / "data/features/daic_woz_covarep_features.csv"
    )
