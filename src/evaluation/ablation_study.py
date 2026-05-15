"""
Ablation Study — Feature Group Importance
==========================================
Systematically removes acoustic feature groups to quantify
their individual contribution to each condition's model.

Groups tested:
  Parkinson's : Jitter | Shimmer | HNR/NHR | Nonlinear | Pitch
  Respiratory : MFCC   | Prosodic | Voice Quality
  Depression  : F0/VUV | MCEP | HMPDM | HMPDD | Clinical
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.preprocessing   import StandardScaler
from sklearn.impute          import SimpleImputer
from sklearn.decomposition   import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import (StratifiedKFold, LeaveOneGroupOut,
                                     cross_val_predict)
from sklearn.metrics         import roc_auc_score

import sys
BASE = Path.home() / "Desktop" / "PULSE_IQ_AI"
sys.path.insert(0, str(BASE/"src"))
from feature_extraction.data_loaders import load_parkinsons, load_respiratory

RESULTS = BASE / "results" / "metrics"


def auroc_cv(X, y, pipe, cv, groups=None):
    y_prob = cross_val_predict(pipe, X, y, cv=cv,
                               groups=groups, method="predict_proba")[:,1]
    return round(roc_auc_score(y, y_prob), 4)


def make_rf():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42,
                                       class_weight="balanced", n_jobs=-1))])


def make_svm_dep():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("sel", SelectKBest(mutual_info_classif, k=30)),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                    probability=True, random_state=42))])


# ── Parkinson's Ablation ──────────────────────────────────────────────────────

def ablation_parkinsons():
    print("\n[1/3] Parkinson's Ablation Study")
    print("-"*55)
    X, y, meta = load_parkinsons()
    feat_cols   = X.columns.tolist()
    groups      = meta["subject_id"].values
    loso        = LeaveOneGroupOut()

    groups_def = {
        "All features (baseline)": feat_cols,
        "w/o Jitter": [f for f in feat_cols if "Jitter" not in f and "RAP" not in f and "PPQ" not in f and "DDP" not in f],
        "w/o Shimmer": [f for f in feat_cols if "Shimmer" not in f and "APQ" not in f and "DDA" not in f],
        "w/o HNR/NHR": [f for f in feat_cols if f not in ["NHR","HNR"]],
        "w/o Nonlinear (RPDE,DFA,PPE,spread,D2)": [f for f in feat_cols if f not in ["RPDE","DFA","PPE","spread1","spread2","D2"]],
        "w/o Pitch (Fo,Fhi,Flo)": [f for f in feat_cols if "Fo" not in f and "Fhi" not in f and "Flo" not in f],
        "Jitter only": [f for f in feat_cols if "Jitter" in f or "RAP" in f or "PPQ" in f or "DDP" in f],
        "Nonlinear only": [f for f in feat_cols if f in ["RPDE","DFA","PPE","spread1","spread2","D2"]],
    }

    results = []
    baseline_auc = None
    for name, cols in groups_def.items():
        if len(cols) < 2:
            continue
        auc = auroc_cv(X[cols].values, y.values, make_rf(), loso, groups=groups)
        drop = "" if baseline_auc is None else f"  ({auc-baseline_auc:+.4f})"
        print(f"  {name:<45} AUROC={auc}{drop}")
        if baseline_auc is None:
            baseline_auc = auc
        results.append({"condition":"parkinsons","ablation":name,
                        "auroc":auc,"drop_from_baseline":round(auc-baseline_auc,4)})
    return results


# ── Respiratory Ablation ──────────────────────────────────────────────────────

def ablation_respiratory():
    print("\n[2/3] Respiratory Ablation Study")
    print("-"*55)
    X, y, _ = load_respiratory()
    feat_cols = X.columns.tolist()
    skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    mfcc_cols    = [f for f in feat_cols if f.startswith("mfcc_")]
    prosodic     = ["pitch","spectral_centroid","zcr"]
    voice_qual   = ["jitter","shimmer","hnr"]

    groups_def = {
        "All features (baseline)": feat_cols,
        "w/o MFCCs (13 coefficients)": [f for f in feat_cols if f not in mfcc_cols],
        "w/o Prosodic (pitch,SC,ZCR)": [f for f in feat_cols if f not in prosodic],
        "w/o Voice Quality (jitter,shimmer,HNR)": [f for f in feat_cols if f not in voice_qual],
        "MFCCs only": mfcc_cols,
        "Prosodic only": prosodic,
        "Voice Quality only": voice_qual,
        "Prosodic + Voice Quality (no MFCC)": prosodic + voice_qual,
    }

    results = []
    baseline_auc = None
    for name, cols in groups_def.items():
        if len(cols) < 2:
            continue
        auc = auroc_cv(X[cols].values, y.values, make_rf(), skf)
        drop = "" if baseline_auc is None else f"  ({auc-baseline_auc:+.4f})"
        print(f"  {name:<45} AUROC={auc}{drop}")
        if baseline_auc is None:
            baseline_auc = auc
        results.append({"condition":"respiratory","ablation":name,
                        "auroc":auc,"drop_from_baseline":round(auc-baseline_auc,4)})
    return results


# ── Depression Ablation ───────────────────────────────────────────────────────

def ablation_depression():
    print("\n[3/3] Depression Ablation Study (AVEC 2017 Dev Set)")
    print("-"*55)
    df       = pd.read_csv(BASE/"data/features/daic_woz_covarep_allframes.csv")
    train_df = pd.read_csv(BASE/"data/raw/daic_woz/train_split_Depression_AVEC2017.csv").rename(columns={"Participant_ID":"participant_id"})
    dev_df   = pd.read_csv(BASE/"data/raw/daic_woz/dev_split_Depression_AVEC2017.csv").rename(columns={"Participant_ID":"participant_id"})

    all_feats = [c for c in df.columns if c.endswith("_mean") or c.endswith("_std")]
    train = df[df.participant_id.isin(train_df.participant_id)]
    dev   = df[df.participant_id.isin(dev_df.participant_id)]

    f0_vuv   = [f for f in all_feats if f.startswith("F0_") or f.startswith("VUV_")]
    mcep     = [f for f in all_feats if f.startswith("MCEP_")]
    hmpdm    = [f for f in all_feats if f.startswith("HMPDM_")]
    hmpdd    = [f for f in all_feats if f.startswith("HMPDD_")]
    clinical = [f for f in all_feats if any(f.startswith(p) for p in
                ["NAQ_","QOQ_","H1H2_","PSP_","MDQ_","peakSlope_","Rd_","creak_","MFCC0_"])]

    groups_def = {
        "All features (baseline)": all_feats,
        "w/o F0/VUV (speaking rate)": [f for f in all_feats if f not in f0_vuv],
        "w/o MCEP (spectral envelope)": [f for f in all_feats if f not in mcep],
        "w/o HMPDM (phase distortion mean)": [f for f in all_feats if f not in hmpdm],
        "w/o HMPDD (phase distortion dev)": [f for f in all_feats if f not in hmpdd],
        "w/o Clinical (NAQ,QOQ,H1H2...)": [f for f in all_feats if f not in clinical],
        "F0/VUV only": f0_vuv,
        "MCEP only": mcep,
    }

    results = []
    baseline_auc = None
    for name, cols in groups_def.items():
        if len(cols) < 5:
            continue
        X_tr = train[cols].values
        X_dv = dev[cols].values
        y_tr = train["PHQ8_Binary"].values
        y_dv = dev["PHQ8_Binary"].values

        k = min(30, len(cols))
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
            ("sel", SelectKBest(mutual_info_classif, k=k)),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                        probability=True, random_state=42))
        ])
        pipe.fit(X_tr, y_tr)
        y_prob = pipe.predict_proba(X_dv)[:,1]
        auc = round(roc_auc_score(y_dv, y_prob), 4)

        drop = "" if baseline_auc is None else f"  ({auc-baseline_auc:+.4f})"
        print(f"  {name:<45} AUROC={auc}{drop}")
        if baseline_auc is None:
            baseline_auc = auc
        results.append({"condition":"depression","ablation":name,
                        "auroc":auc,"drop_from_baseline":round(auc-baseline_auc,4)})
    return results


if __name__ == "__main__":
    print("="*60)
    print(" ABLATION STUDY — Feature Group Importance")
    print("="*60)

    all_results = []
    all_results.extend(ablation_parkinsons())
    all_results.extend(ablation_respiratory())
    all_results.extend(ablation_depression())

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS/"ablation_study.csv", index=False)
    print(f"\nSaved -> {RESULTS/'ablation_study.csv'}")

    print("\n" + "="*60)
    print(" KEY ABLATION FINDINGS")
    print("="*60)
    for cond in ["parkinsons","respiratory","depression"]:
        sub = df[df.condition==cond].copy()
        sub = sub[sub.ablation != sub[sub.drop_from_baseline==0].ablation.values[0]]
        worst = sub.loc[sub.drop_from_baseline.idxmin()]
        print(f"  {cond}: biggest drop when removing '{worst.ablation}'"
              f" (AUROC drop: {worst.drop_from_baseline:+.4f})")
