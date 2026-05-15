"""
PulseIQ AI — Model Training Pipeline v2
Fixes depression model: PCA + SMOTE + no nested calibration in CV
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    StratifiedKFold, LeaveOneGroupOut, cross_val_predict
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

BASE    = Path.home() / "Desktop" / "PULSE_IQ_AI"
MODELS  = BASE / "models"
RESULTS = BASE / "results" / "metrics"
MODELS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)


def make_rf_pipeline(class_weight="balanced", n_estimators=300,
                     use_pca=False, pca_components=40,
                     use_smote=False, calibrate=True):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    model = CalibratedClassifierCV(rf, cv=3, method="sigmoid") if calibrate else rf

    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ]
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_components, random_state=42)))

    if use_smote:
        steps.append(("smote", SMOTE(random_state=42, k_neighbors=3)))
        steps.append(("model", model))
        return ImbPipeline(steps)
    else:
        steps.append(("model", model))
        return Pipeline(steps)


def evaluate_cv(X, y, pipeline, cv, groups=None, label=""):
    y_prob = cross_val_predict(
        pipeline, X, y,
        cv=cv, groups=groups,
        method="predict_proba"
    )[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auroc    = roc_auc_score(y, y_prob)
    accuracy = accuracy_score(y, y_pred)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"\n  {label}")
    print(f"  Accuracy    : {accuracy:.4f}")
    print(f"  AUROC       : {auroc:.4f}")
    print(f"  Sensitivity : {sensitivity:.4f}")
    print(f"  Specificity : {specificity:.4f}")

    return {
        "label": label, "accuracy": round(accuracy,4),
        "auroc": round(auroc,4), "sensitivity": round(sensitivity,4),
        "specificity": round(specificity,4),
        "n_samples": len(y), "n_positive": int(y.sum())
    }


def train_and_save(X, y, name, **pipeline_kwargs):
    pipe = make_rf_pipeline(**pipeline_kwargs)
    pipe.fit(X, y)
    path = MODELS / f"{name}_model.pkl"
    joblib.dump(pipe, path)
    print(f"  Model saved → {path}")
    return pipe


# ── Parkinson's ───────────────────────────────────────────────────────────────

def run_parkinsons(X, y, meta):
    print("\n" + "="*55)
    print(" PARKINSON'S DISEASE")
    print("="*55)
    results = []

    subjects = meta["subject_id"].values
    r_loso = evaluate_cv(X, y,
                         make_rf_pipeline(calibrate=False),
                         LeaveOneGroupOut(), groups=subjects,
                         label="LOSO (primary — subject-independent)")
    r_loso["protocol"] = "LOSO"
    results.append(r_loso)

    r_cv = evaluate_cv(X, y,
                       make_rf_pipeline(calibrate=True),
                       StratifiedKFold(5, shuffle=True, random_state=42),
                       label="5-Fold CV (secondary)")
    r_cv["protocol"] = "5-Fold CV"
    results.append(r_cv)

    pipe = train_and_save(X, y, "parkinsons", calibrate=True)
    return results, pipe


# ── Respiratory ───────────────────────────────────────────────────────────────

def run_respiratory(X, y, meta):
    print("\n" + "="*55)
    print(" RESPIRATORY ABNORMALITY")
    print("="*55)

    r_cv = evaluate_cv(X, y,
                       make_rf_pipeline(calibrate=True),
                       StratifiedKFold(5, shuffle=True, random_state=42),
                       label="5-Fold CV")
    r_cv["protocol"] = "5-Fold CV"

    pipe = train_and_save(X, y, "respiratory", calibrate=True)
    return [r_cv], pipe


# ── Depression ────────────────────────────────────────────────────────────────

def run_depression(X, y, meta):
    print("\n" + "="*55)
    print(" DEPRESSION (DAIC-WOZ / PHQ-8)")
    print("="*55)
    results = []

    # LOSO — no calibration/SMOTE (too small per fold)
    subjects = meta["participant_id"].values
    r_loso = evaluate_cv(X, y,
                         make_rf_pipeline(use_pca=True, pca_components=40,
                                          calibrate=False),
                         LeaveOneGroupOut(), groups=subjects,
                         label="LOSO (primary — subject-independent)")
    r_loso["protocol"] = "LOSO"
    results.append(r_loso)

    # 5-fold with SMOTE — more data per fold, SMOTE works
    r_cv = evaluate_cv(X, y,
                       make_rf_pipeline(use_pca=True, pca_components=40,
                                        use_smote=True, calibrate=False),
                       StratifiedKFold(5, shuffle=True, random_state=42),
                       label="5-Fold CV + SMOTE (secondary)")
    r_cv["protocol"] = "5-Fold CV+SMOTE"
    results.append(r_cv)

    pipe = train_and_save(X, y, "depression",
                          use_pca=True, pca_components=40,
                          use_smote=True, calibrate=True)
    return results, pipe


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(BASE / "src"))
    from feature_extraction.data_loaders import load_all

    datasets    = load_all()
    all_results = []

    pk_r,   pk_pipe   = run_parkinsons(  *datasets["parkinsons"])
    resp_r, resp_pipe = run_respiratory( *datasets["respiratory"])
    dep_r,  dep_pipe  = run_depression(  *datasets["depression"])

    for condition, results in [
        ("parkinsons", pk_r), ("respiratory", resp_r), ("depression", dep_r)
    ]:
        for r in results:
            r["condition"] = condition
            all_results.append(r)

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS / "model_performance.csv", index=False)

    print("\n" + "="*55)
    print(" FINAL RESULTS SUMMARY")
    print("="*55)
    print(df[["condition","protocol","accuracy","auroc",
              "sensitivity","specificity"]].to_string(index=False))
    print(f"\nSaved → {RESULTS / 'model_performance.csv'}")
