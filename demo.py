import warnings
warnings.filterwarnings("ignore")

import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import sys
import os
import joblib
import pandas as pd
import shap

sys.path.append("src")
from feature_extraction.audio_features import extract_features

# ── Configuration ────────────────────────────────────────────────────
SAMPLE_RATE = 22050
DURATION    = 10
OUTPUT_PATH = "recorded_voice.wav"
VERSION     = "1.0.0"

# ── Load Models ──────────────────────────────────────────────────────
pk_model         = joblib.load("models/parkinsons_model.pkl")
pk_imputer       = joblib.load("models/parkinsons_imputer.pkl")
resp_model       = joblib.load("models/respiratory_model.pkl")
resp_imputer     = joblib.load("models/respiratory_imputer.pkl")
stress_model     = joblib.load("models/stress_model.pkl")
stress_imputer   = joblib.load("models/stress_imputer.pkl")
depr_model       = joblib.load("models/depression_model.pkl")
depr_imputer     = joblib.load("models/depression_imputer.pkl")

# ── Feature Definitions ──────────────────────────────────────────────
PK_FEATURES = ["pitch", "hnr", "jitter", "shimmer", "nhr", "rpde", "dfa"]

GENERAL_FEATURES = [
    "pitch", "spectral_centroid", "zcr",
    "jitter", "shimmer", "hnr",
    "mfcc_1","mfcc_2","mfcc_3","mfcc_4","mfcc_5","mfcc_6","mfcc_7",
    "mfcc_8","mfcc_9","mfcc_10","mfcc_11","mfcc_12","mfcc_13"
]

RPDE_MEAN = 0.498
DFA_MEAN  = 0.718

# ── SHAP Explainers ──────────────────────────────────────────────────
df_pk = pd.read_csv("data/parkinsons_tabular/parkinsons.csv").rename(columns={
    "MDVP:Fo(Hz)": "pitch", "HNR": "hnr",
    "MDVP:Jitter(Abs)": "jitter", "MDVP:Shimmer": "shimmer",
    "NHR": "nhr", "RPDE": "rpde", "DFA": "dfa"
})
pk_explainer     = shap.TreeExplainer(pk_model)
resp_explainer   = shap.TreeExplainer(resp_model)
stress_explainer = shap.TreeExplainer(stress_model)
depr_explainer   = shap.TreeExplainer(depr_model)

# ── Helpers ──────────────────────────────────────────────────────────
def separator(char="-", width=60):
    print(char * width)

def risk_band(prob):
    if prob < 0.35:
        return "LOW     ", "No significant acoustic indicators detected"
    elif prob < 0.50:
        return "GUARDED ", "Minor acoustic indicators present"
    elif prob < 0.65:
        return "MODERATE", "Elevated acoustic indicators detected"
    else:
        return "HIGH    ", "Strong acoustic indicators detected"

def shap_bar(val, scale=50):
    filled = min(8, int(abs(val) * scale))
    return "[" + "#" * filled + " " * max(0, 8 - filled) + "]"

def print_shap(vals, feature_names, top_n=4):
    contrib = pd.Series(vals, index=feature_names)
    contrib = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
    for feat, val in contrib.head(top_n).items():
        direction = "(+) increases risk" if val > 0 else "(-) decreases risk"
        bar = shap_bar(val)
        print(f"    {feat:<24} {val:+.4f}  {bar}  {direction}")

# ── Record Voice ─────────────────────────────────────────────────────
def record_voice():
    separator("=")
    print("  PulseIQ AI  |  Voice-Based Multi-Disease Screening")
    print(f"  Version {VERSION}  |  Research Prototype")
    separator("=")
    print(f"\n  Recording duration : {DURATION} seconds")
    print("  Instruction        : Count from 1 to 30 at a natural pace")
    print("  Starting now...\n")
    separator()

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    write(OUTPUT_PATH, SAMPLE_RATE, audio)
    print("  Recording complete.")
    separator()

# ── Analyze ──────────────────────────────────────────────────────────
def analyze(audio_path):
    print(f"\n  Extracting acoustic features from: {audio_path}")
    features = extract_features(audio_path)
    print("  Feature extraction complete. Running inference...\n")

    # ── Parkinson's ──────────────────────────────────────────────────
    pk_input = pd.DataFrame([{
        "pitch":   features["pitch"],
        "hnr":     features["hnr"],
        "jitter":  features["jitter"],
        "shimmer": features["shimmer"],
        "nhr":     1 - (features["hnr"] / (features["hnr"] + 1)),
        "rpde":    RPDE_MEAN,
        "dfa":     DFA_MEAN
    }])
    pk_imp    = pk_imputer.transform(pk_input)
    pk_imp_df = pd.DataFrame(pk_imp, columns=PK_FEATURES)
    pk_prob   = pk_model.predict_proba(pk_imp_df)[0][1]
    pk_shap   = pk_explainer.shap_values(pk_imp_df)
    pk_vals   = pk_shap[0, :, 1] if pk_shap.ndim == 3 else pk_shap[0]

    # ── Respiratory ──────────────────────────────────────────────────
    resp_input  = pd.DataFrame([features])[GENERAL_FEATURES]
    resp_imp    = resp_imputer.transform(resp_input)
    resp_imp_df = pd.DataFrame(resp_imp, columns=GENERAL_FEATURES)
    resp_prob   = resp_model.predict_proba(resp_imp_df)[0][1]
    resp_shap   = resp_explainer.shap_values(resp_imp_df)
    resp_vals   = resp_shap[0, :, 1] if resp_shap.ndim == 3 else resp_shap[0]

    # ── Stress ───────────────────────────────────────────────────────
    stress_input  = pd.DataFrame([features])[GENERAL_FEATURES]
    stress_imp    = stress_imputer.transform(stress_input)
    stress_imp_df = pd.DataFrame(stress_imp, columns=GENERAL_FEATURES)
    stress_prob   = stress_model.predict_proba(stress_imp_df)[0][1]
    stress_shap   = stress_explainer.shap_values(stress_imp_df)
    stress_vals   = stress_shap[0, :, 1] if stress_shap.ndim == 3 else stress_shap[0]

    # ── Depression ───────────────────────────────────────────────────
    depr_input  = pd.DataFrame([features])[GENERAL_FEATURES]
    depr_imp    = depr_imputer.transform(depr_input)
    depr_imp_df = pd.DataFrame(depr_imp, columns=GENERAL_FEATURES)
    depr_prob   = depr_model.predict_proba(depr_imp_df)[0][1]
    depr_shap   = depr_explainer.shap_values(depr_imp_df)
    depr_vals   = depr_shap[0, :, 1] if depr_shap.ndim == 3 else depr_shap[0]

    # ── Risk Bands ────────────────────────────────────────────────────
    pk_band,     pk_note     = risk_band(pk_prob)
    resp_band,   resp_note   = risk_band(resp_prob)
    stress_band, stress_note = risk_band(stress_prob)
    depr_band,   depr_note   = risk_band(depr_prob)

    # ── Print Report ──────────────────────────────────────────────────
    separator("=")
    print("  PULSEIQ AI  —  ACOUSTIC RISK ASSESSMENT REPORT")
    separator("=")
    print(f"  {'CONDITION':<30} {'RISK BAND':<12} {'SCORE':<10} {'MODEL AUROC'}")
    separator()
    print(f"  {'Parkinson Disease':<30} {pk_band:<12} {pk_prob:.1%}     0.949")
    print(f"  {'Respiratory Abnormality':<30} {resp_band:<12} {resp_prob:.1%}     0.748")
    print(f"  {'Psychological Stress':<30} {stress_band:<12} {stress_prob:.1%}     0.918")
    print(f"  {'Depression Indicators':<30} {depr_band:<12} {depr_prob:.1%}     0.856")
    separator("=")

    # ── Acoustic Features ─────────────────────────────────────────────
    print("\n  EXTRACTED ACOUSTIC FEATURES")
    separator()
    print(f"  {'Pitch (F0)':<28} {features['pitch']:>10.2f} Hz")
    print(f"  {'Jitter':<28} {features['jitter']:>10.6f}")
    print(f"  {'Shimmer':<28} {features['shimmer']:>10.6f}")
    print(f"  {'HNR':<28} {features['hnr']:>10.4f} dB")
    print(f"  {'Spectral Centroid':<28} {features['spectral_centroid']:>10.2f} Hz")
    print(f"  {'Zero Crossing Rate':<28} {features['zcr']:>10.6f}")
    print(f"  {'MFCC-1':<28} {features['mfcc_1']:>10.4f}")
    separator()

    # ── SHAP Explanations ─────────────────────────────────────────────
    print("\n  FEATURE ATTRIBUTION (SHAP)")
    print("  Top 4 features driving each prediction:\n")

    print("  [1] Parkinson Disease")
    print(f"      {'Feature':<24} {'SHAP Value':<10} {'Contribution':<12} {'Direction'}")
    separator()
    print_shap(pk_vals, PK_FEATURES)

    print("\n  [2] Respiratory Abnormality")
    print(f"      {'Feature':<24} {'SHAP Value':<10} {'Contribution':<12} {'Direction'}")
    separator()
    print_shap(resp_vals, GENERAL_FEATURES)

    print("\n  [3] Psychological Stress")
    print(f"      {'Feature':<24} {'SHAP Value':<10} {'Contribution':<12} {'Direction'}")
    separator()
    print_shap(stress_vals, GENERAL_FEATURES)

    print("\n  [4] Depression Indicators")
    print(f"      {'Feature':<24} {'SHAP Value':<10} {'Contribution':<12} {'Direction'}")
    separator()
    print_shap(depr_vals, GENERAL_FEATURES)

    separator("=")
    print("  DISCLAIMER")
    separator()
    print("  PulseIQ AI is a research prototype for academic purposes only.")
    print("  It is not a certified medical device and must not be used")
    print("  for clinical diagnosis or treatment decisions.")
    print("  All outputs should be interpreted by a qualified clinician.")
    separator("=")
    print()


# ── Entry Point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze(sys.argv[1])
    else:
        record_voice()
        analyze(OUTPUT_PATH)