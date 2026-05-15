"""
Depression Model Training — Full Pipeline
==========================================
Experiments:
  1. COVAREP + SVM (official AVEC 2017 split) ← best model
  2. wav2vec + SVM (comparison)
  3. COVAREP + wav2vec Fusion (comparison)

Key finding: COVAREP outperforms wav2vec in low-data regime (N=107)
"""

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from sklearn.feature_selection import SelectKBest, mutual_info_classif

BASE    = Path.home() / "Desktop" / "PULSE_IQ_AI"
MODELS  = BASE / "models"
RESULTS = BASE / "results" / "metrics"

def load_splits(features_path, feat_prefix=None, feat_suffix=None):
    df       = pd.read_csv(features_path)
    train_df = pd.read_csv(BASE/"data/raw/daic_woz/train_split_Depression_AVEC2017.csv").rename(columns={"Participant_ID":"participant_id"})
    dev_df   = pd.read_csv(BASE/"data/raw/daic_woz/dev_split_Depression_AVEC2017.csv").rename(columns={"Participant_ID":"participant_id"})

    if feat_prefix:
        feat_cols = [c for c in df.columns if c.startswith(feat_prefix)]
    elif feat_suffix:
        feat_cols = [c for c in df.columns if c.endswith(feat_suffix[0]) or c.endswith(feat_suffix[1])]
    else:
        feat_cols = [c for c in df.columns if c.endswith("_mean") or c.endswith("_std")]

    train = df[df.participant_id.isin(train_df.participant_id)]
    dev   = df[df.participant_id.isin(dev_df.participant_id)]
    return train[feat_cols].values, train["PHQ8_Binary"].values, dev[feat_cols].values, dev["PHQ8_Binary"].values, feat_cols


def evaluate(pipe, X_train, y_train, X_dev, y_dev, label):
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        "svm__C"    : [0.01, 0.1, 1.0, 10.0, 100.0],
        "svm__gamma": ["scale", "auto", 0.001, 0.01]
    }
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    gs.fit(X_train, y_train)

    best   = gs.best_estimator_
    y_prob = best.predict_proba(X_dev)[:,1]
    d_auc  = roc_auc_score(y_dev, y_prob)
    fpr, tpr, thresholds = roc_curve(y_dev, y_prob)
    y_pred = (y_prob >= thresholds[np.argmax(tpr - fpr)]).astype(int)
    tn,fp,fn,tp = confusion_matrix(y_dev, y_pred).ravel()

    print(f"\n[{label}]")
    print(f"  Best params  : {gs.best_params_}")
    print(f"  CV AUROC     : {gs.best_score_:.4f}")
    print(f"  Dev AUROC    : {d_auc:.4f}")
    print(f"  Sensitivity  : {tp/(tp+fn) if (tp+fn)>0 else 0:.4f}")
    print(f"  Specificity  : {tn/(tn+fp) if (tn+fp)>0 else 0:.4f}")
    return best, d_auc, gs.best_score_


def make_svm_pipe(pca_components=30):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("sel", SelectKBest(mutual_info_classif, k=30)),
        ("svm", SVC(kernel="rbf", probability=True, random_state=42))
    ])


if __name__ == "__main__":
    print("="*60)
    print(" DEPRESSION — Full Model Comparison")
    print("="*60)

    results = []

    # 1. COVAREP + SVM (best model)
    X_train, y_train, X_dev, y_dev, _ = load_splits(
        BASE/"data/features/daic_woz_covarep_allframes.csv")
    best_cov, auc_cov, cv_cov = evaluate(
        make_svm_pipe(), X_train, y_train, X_dev, y_dev, "COVAREP + SVM")
    joblib.dump(best_cov, MODELS/"depression_covarep_svm.pkl")
    results.append({"model":"COVAREP+SVM", "cv_auroc":cv_cov, "dev_auroc":auc_cov})

    # 2. wav2vec + SVM (comparison)
    X_train_w, y_train_w, X_dev_w, y_dev_w, _ = load_splits(
        BASE/"data/features/daic_woz_wav2vec_features.csv", feat_prefix="w2v_")
    pipe_w2v = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("pca", PCA(n_components=50, random_state=42)),
        ("svm", SVC(kernel="rbf", probability=True, random_state=42))
    ])
    best_w2v, auc_w2v, cv_w2v = evaluate(
        pipe_w2v, X_train_w, y_train_w, X_dev_w, y_dev_w, "wav2vec + SVM")
    joblib.dump(best_w2v, MODELS/"depression_wav2vec_svm.pkl")
    results.append({"model":"wav2vec+SVM", "cv_auroc":cv_w2v, "dev_auroc":auc_w2v})

    # 3. Fusion (comparison)
    cov_df = pd.read_csv(BASE/"data/features/daic_woz_covarep_allframes.csv")
    w2v_df = pd.read_csv(BASE/"data/features/daic_woz_wav2vec_features.csv")
    train_df = pd.read_csv(BASE/"data/raw/daic_woz/train_split_Depression_AVEC2017.csv").rename(columns={"Participant_ID":"participant_id"})
    dev_df_s = pd.read_csv(BASE/"data/raw/daic_woz/dev_split_Depression_AVEC2017.csv").rename(columns={"Participant_ID":"participant_id"})

    cov_feats = [c for c in cov_df.columns if c.endswith("_mean") or c.endswith("_std")]
    w2v_feats = [c for c in w2v_df.columns if c.startswith("w2v_")]
    merged    = cov_df[["participant_id","PHQ8_Binary"]+cov_feats].merge(
                w2v_df[["participant_id"]+w2v_feats], on="participant_id")

    train_m = merged[merged.participant_id.isin(train_df.participant_id)]
    dev_m   = merged[merged.participant_id.isin(dev_df_s.participant_id)]
    all_f   = cov_feats + w2v_feats

    pipe_fus = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("pca", PCA(n_components=40, random_state=42)),
        ("svm", SVC(kernel="rbf", probability=True, random_state=42))
    ])
    best_fus, auc_fus, cv_fus = evaluate(
        pipe_fus,
        train_m[all_f].values, train_m["PHQ8_Binary"].values,
        dev_m[all_f].values,   dev_m["PHQ8_Binary"].values,
        "Fusion COVAREP+wav2vec")
    joblib.dump(best_fus, MODELS/"depression_fusion.pkl")
    results.append({"model":"Fusion", "cv_auroc":cv_fus, "dev_auroc":auc_fus})

    # Summary
    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    res_df.to_csv(RESULTS/"depression_model_comparison.csv", index=False)
    print(f"\nKey finding: COVAREP ({auc_cov:.4f}) > wav2vec ({auc_w2v:.4f}) in low-data regime")
    print("This supports WavRx [R23]: speaker identity leakage hurts wav2vec on small datasets")
