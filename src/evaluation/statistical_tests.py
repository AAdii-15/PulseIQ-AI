"""
Statistical Significance Tests
================================
McNemar's test: RF vs best competing model per condition
Bootstrap confidence intervals: already in baseline_comparison.py
Permutation test: overall model significance

Reports in paper as: p-values with significance stars
  * p < 0.05, ** p < 0.01, *** p < 0.001
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from scipy.stats import wilcoxon, permutation_test
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.model_selection import (StratifiedKFold, LeaveOneGroupOut,
                                     cross_val_predict)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score

import sys
BASE = Path.home() / "Desktop" / "PULSE_IQ_AI"
sys.path.insert(0, str(BASE/"src"))
from feature_extraction.data_loaders import load_parkinsons, load_respiratory

RESULTS = BASE / "results" / "metrics"


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    b = np.sum((y_pred_a == y_true) & (y_pred_b != y_true))
    c = np.sum((y_pred_a != y_true) & (y_pred_b == y_true))
    if b + c == 0:
        return 1.0
    table  = [[0, b], [c, 0]]
    result = mcnemar(table, exact=True)
    return result.pvalue


def permutation_auroc_test(y_true, y_prob, n_perm=1000):
    """Test if AUROC is significantly better than chance (0.5)."""
    obs_auc = roc_auc_score(y_true, y_prob)
    rng = np.random.default_rng(42)
    null_aucs = []
    for _ in range(n_perm):
        shuffled = rng.permutation(y_true)
        null_aucs.append(roc_auc_score(shuffled, y_prob))
    p_val = np.mean(np.array(null_aucs) >= obs_auc)
    return obs_auc, p_val


def make_pipe(model):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("clf", model)
    ])


if __name__ == "__main__":
    results = []
    print("="*65)
    print(" STATISTICAL SIGNIFICANCE TESTS")
    print("="*65)

    # ── Parkinson's ───────────────────────────────────────────────────────────
    print("\n[1] Parkinson's Disease — LOSO")
    print("-"*65)
    X_pk, y_pk, meta = load_parkinsons()
    loso   = LeaveOneGroupOut()
    groups = meta["subject_id"].values

    models_pk = {
        "RF"                 : RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "SVM"                : SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"),
    }
    probs_pk = {}
    for name, clf in models_pk.items():
        pipe    = make_pipe(clf)
        y_prob  = cross_val_predict(pipe, X_pk.values, y_pk.values,
                                    cv=loso, groups=groups, method="predict_proba")[:,1]
        y_pred  = (y_prob >= 0.5).astype(int)
        auc     = roc_auc_score(y_pk, y_prob)
        obs_auc, p_perm = permutation_auroc_test(y_pk.values, y_prob)
        probs_pk[name] = (y_prob, y_pred)
        print(f"  {name:<22} AUROC={auc:.4f}  vs-chance: p={p_perm:.4f} {sig_stars(p_perm)}")

    # McNemar: RF vs LR and RF vs SVM
    for name in ["Logistic Regression", "SVM"]:
        rf_pred   = probs_pk["RF"][1]
        other_pred= probs_pk[name][1]
        p = mcnemar_test(y_pk.values, rf_pred, other_pred)
        print(f"  McNemar RF vs {name:<20} p={p:.4f} {sig_stars(p)}")
        results.append({"condition":"parkinsons","comparison":f"RF vs {name}",
                        "mcnemar_p":round(p,4),"sig":sig_stars(p)})

    # ── Respiratory ───────────────────────────────────────────────────────────
    print("\n[2] Respiratory — 5-Fold CV")
    print("-"*65)
    X_resp, y_resp, _ = load_respiratory()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models_resp = {
        "RF"                 : RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "SVM"                : SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"),
    }
    probs_resp = {}
    for name, clf in models_resp.items():
        pipe   = make_pipe(clf)
        y_prob = cross_val_predict(pipe, X_resp.values, y_resp.values,
                                   cv=skf, method="predict_proba")[:,1]
        y_pred = (y_prob >= 0.5).astype(int)
        auc    = roc_auc_score(y_resp, y_prob)
        obs_auc, p_perm = permutation_auroc_test(y_resp.values, y_prob, n_perm=500)
        probs_resp[name] = (y_prob, y_pred)
        print(f"  {name:<22} AUROC={auc:.4f}  vs-chance: p={p_perm:.4f} {sig_stars(p_perm)}")

    for name in ["Logistic Regression", "SVM"]:
        rf_pred    = probs_resp["RF"][1]
        other_pred = probs_resp[name][1]
        p = mcnemar_test(y_resp.values, rf_pred, other_pred)
        print(f"  McNemar RF vs {name:<20} p={p:.4f} {sig_stars(p)}")
        results.append({"condition":"respiratory","comparison":f"RF vs {name}",
                        "mcnemar_p":round(p,4),"sig":sig_stars(p)})

    # ── Depression ────────────────────────────────────────────────────────────
    print("\n[3] Depression — AVEC 2017 Dev Set")
    print("-"*65)
    import joblib
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    from sklearn.decomposition import PCA

    df_dep   = pd.read_csv(BASE/"data/features/daic_woz_covarep_allframes.csv")
    train_df = pd.read_csv(BASE/"data/raw/daic_woz/train_split_Depression_AVEC2017.csv").rename(columns={"Participant_ID":"participant_id"})
    dev_df   = pd.read_csv(BASE/"data/raw/daic_woz/dev_split_Depression_AVEC2017.csv").rename(columns={"Participant_ID":"participant_id"})
    feat_cols = [c for c in df_dep.columns if c.endswith("_mean") or c.endswith("_std")]
    train    = df_dep[df_dep.participant_id.isin(train_df.participant_id)]
    dev      = df_dep[df_dep.participant_id.isin(dev_df.participant_id)]
    X_tr, y_tr = train[feat_cols].values, train["PHQ8_Binary"].values
    X_dv, y_dv = dev[feat_cols].values,   dev["PHQ8_Binary"].values

    # Use saved best model (no refit)
    best_dep = joblib.load(BASE/"models/depression_svm_allframes.pkl")
    y_prob_best = best_dep.predict_proba(X_dv)[:,1]
    y_pred_best = (y_prob_best >= 0.5).astype(int)
    auc_best    = roc_auc_score(y_dv, y_prob_best)
    obs, p_perm = permutation_auroc_test(y_dv, y_prob_best, n_perm=1000)
    print(f"  SVM (COVAREP)       AUROC={auc_best:.4f}  vs-chance: p={p_perm:.4f} {sig_stars(p_perm)}")

    # LR baseline
    pipe_lr = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("pca", PCA(n_components=30, random_state=42)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
    ])
    pipe_lr.fit(X_tr, y_tr)
    y_prob_lr = pipe_lr.predict_proba(X_dv)[:,1]
    y_pred_lr = (y_prob_lr >= 0.5).astype(int)
    auc_lr    = roc_auc_score(y_dv, y_prob_lr)
    print(f"  LR (PCA)            AUROC={auc_lr:.4f}")

    p_mc = mcnemar_test(y_dv, y_pred_best, y_pred_lr)
    print(f"  McNemar SVM vs LR            p={p_mc:.4f} {sig_stars(p_mc)}")
    results.append({"condition":"depression","comparison":"SVM vs LR",
                    "mcnemar_p":round(p_mc,4),"sig":sig_stars(p_mc)})

    print(f"\n  Best depression model: AUROC {auc_best:.4f} vs chance: {sig_stars(p_perm)}")

    # Save
    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS/"statistical_tests.csv", index=False)

    print("\n" + "="*65)
    print(" STATISTICAL TESTS SUMMARY")
    print("="*65)
    print(res_df.to_string(index=False))
    print(f"\nSaved -> {RESULTS/'statistical_tests.csv'}")
