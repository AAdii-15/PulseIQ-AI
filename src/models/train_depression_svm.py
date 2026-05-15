"""
Depression Model — SVM with Official AVEC 2017 Train/Dev Split
----------------------------------------------------------------
Key fixes over RF approach:
  1. Official train/dev split (not LOSO on combined data)
  2. SVM RBF kernel (better for small high-dimensional datasets)
  3. All-frame statistics (captures silence/pause patterns)
  4. GridSearchCV for C and gamma tuning on training set
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    confusion_matrix, classification_report
)

BASE    = Path.home() / "Desktop" / "PULSE_IQ_AI"
MODELS  = BASE / "models"
RESULTS = BASE / "results" / "metrics"


def load_depression_official_split():
    """Load COVAREP features with official AVEC 2017 train/dev split."""
    features_path = BASE / "data/features/daic_woz_covarep_features.csv"
    train_labels  = BASE / "data/raw/daic_woz/train_split_Depression_AVEC2017.csv"
    dev_labels    = BASE / "data/raw/daic_woz/dev_split_Depression_AVEC2017.csv"

    df       = pd.read_csv(features_path)
    train_df = pd.read_csv(train_labels).rename(
                    columns={"Participant_ID": "participant_id"})
    dev_df   = pd.read_csv(dev_labels).rename(
                    columns={"Participant_ID": "participant_id"})

    feat_cols = [c for c in df.columns
                 if c.endswith("_mean") or c.endswith("_std")]

    train = df[df["participant_id"].isin(train_df["participant_id"])]
    dev   = df[df["participant_id"].isin(dev_df["participant_id"])]

    X_train = train[feat_cols].values
    y_train = train["PHQ8_Binary"].values
    X_dev   = dev[feat_cols].values
    y_dev   = dev["PHQ8_Binary"].values

    print(f"Train: {len(X_train)} sessions | "
          f"{y_train.sum()} depressed ({y_train.mean()*100:.1f}%)")
    print(f"Dev  : {len(X_dev)} sessions | "
          f"{y_dev.sum()} depressed ({y_dev.mean()*100:.1f}%)")
    return X_train, y_train, X_dev, y_dev, feat_cols


def train_svm_depression():
    print("="*55)
    print(" DEPRESSION — SVM (Official AVEC 2017 Split)")
    print("="*55)

    X_train, y_train, X_dev, y_dev, feat_cols = \
        load_depression_official_split()

    # Pipeline: impute → scale → PCA → SVM
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("pca",     PCA(n_components=40, random_state=42)),
        ("svm",     SVC(kernel="rbf", class_weight="balanced",
                        probability=True, random_state=42))
    ])

    # Tune C and gamma on training set only
    param_grid = {
        "svm__C"    : [0.01, 0.1, 1.0, 10.0],
        "svm__gamma": ["scale", "auto", 0.001, 0.01]
    }

    print("\nRunning GridSearchCV on training set...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid, cv=cv,
                      scoring="roc_auc", n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)

    print(f"Best params : {gs.best_params_}")
    print(f"CV AUROC    : {gs.best_score_:.4f}")

    # Evaluate on held-out dev set
    best_pipe = gs.best_estimator_
    y_prob    = best_pipe.predict_proba(X_dev)[:, 1]
    y_pred    = best_pipe.predict(X_dev)

    auroc       = roc_auc_score(y_dev, y_prob)
    accuracy    = accuracy_score(y_dev, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_dev, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"\n--- Dev Set Results (Official AVEC 2017) ---")
    print(f"Accuracy    : {accuracy:.4f}")
    print(f"AUROC       : {auroc:.4f}")
    print(f"Sensitivity : {sensitivity:.4f}")
    print(f"Specificity : {specificity:.4f}")
    print(f"\n{classification_report(y_dev, y_pred, target_names=['Non-dep','Depressed'])}")

    # Save model and results
    joblib.dump(best_pipe, MODELS / "depression_svm_model.pkl")
    print(f"Model saved → {MODELS / 'depression_svm_model.pkl'}")

    results = {
        "condition"  : "depression",
        "model"      : "SVM-RBF",
        "protocol"   : "AVEC2017 Train/Dev",
        "accuracy"   : round(accuracy, 4),
        "auroc"      : round(auroc, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "n_train"    : len(X_train),
        "n_dev"      : len(X_dev),
        "best_C"     : gs.best_params_["svm__C"],
        "best_gamma" : str(gs.best_params_["svm__gamma"])
    }

    out = RESULTS / "depression_svm_results.csv"
    pd.DataFrame([results]).to_csv(out, index=False)
    print(f"Results saved → {out}")
    return results, best_pipe


if __name__ == "__main__":
    results, pipe = train_svm_depression()
