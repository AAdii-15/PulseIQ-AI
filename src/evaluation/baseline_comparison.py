"""
Baseline Model Comparison
==========================
Compares RF against LR, DT, SVM, KNN on all 3 conditions
using condition-appropriate evaluation protocols.
Includes McNemar's test and bootstrap CI on AUROC.
"""

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from scipy.stats import bootstrap
from statsmodels.stats.contingency_tables import mcnemar

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.impute          import SimpleImputer
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import (StratifiedKFold, LeaveOneGroupOut,
                                     cross_val_predict)
from sklearn.metrics         import roc_auc_score, accuracy_score, confusion_matrix

import sys
BASE = Path.home() / "Desktop" / "PULSE_IQ_AI"
sys.path.insert(0, str(BASE / "src"))
from feature_extraction.data_loaders import load_parkinsons, load_respiratory

RESULTS = BASE / "results" / "metrics"
RESULTS.mkdir(parents=True, exist_ok=True)


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_auroc_ci(y_true, y_prob, n_boot=1000, ci=0.95):
    """Bootstrap 95% CI for AUROC."""
    aucs = []
    rng  = np.random.default_rng(42)
    n    = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    lo = np.percentile(aucs, (1-ci)/2*100)
    hi = np.percentile(aucs, (1+ci)/2*100)
    return round(np.mean(aucs),4), round(lo,4), round(hi,4)


# ── McNemar's Test ────────────────────────────────────────────────────────────

def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """McNemar's test between model A and model B predictions."""
    b = np.sum((y_pred_a == y_true) & (y_pred_b != y_true))
    c = np.sum((y_pred_a != y_true) & (y_pred_b == y_true))
    table = [[0, b], [c, 0]]
    result = mcnemar(table, exact=True)
    return round(result.pvalue, 4)


# ── Model Definitions ─────────────────────────────────────────────────────────

def get_models():
    return {
        "Logistic Regression": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42,
                                       class_weight="balanced"))]),
        "Decision Tree": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
            ("clf", DecisionTreeClassifier(max_depth=5, random_state=42,
                                           class_weight="balanced"))]),
        "SVM (RBF)": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=42,
                       class_weight="balanced"))]),
        "KNN": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))]),
        "Random Forest": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=42,
                                           class_weight="balanced", n_jobs=-1))])
    }


# ── Evaluation Function ───────────────────────────────────────────────────────

def evaluate_models(X, y, cv, groups=None, condition="", protocol=""):
    models  = get_models()
    results = []
    all_probs = {}

    for name, pipe in models.items():
        y_prob = cross_val_predict(pipe, X, y, cv=cv,
                                   groups=groups, method="predict_proba")[:,1]
        y_pred = (y_prob >= 0.5).astype(int)

        auroc = roc_auc_score(y, y_prob)
        acc   = accuracy_score(y, y_pred)
        tn,fp,fn,tp = confusion_matrix(y, y_pred).ravel()
        sens = tp/(tp+fn) if (tp+fn)>0 else 0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0

        auc_mean, auc_lo, auc_hi = bootstrap_auroc_ci(y, y_prob)

        results.append({
            "condition": condition, "protocol": protocol,
            "model": name,
            "accuracy": round(acc,4), "auroc": round(auroc,4),
            "auroc_ci_lo": auc_lo, "auroc_ci_hi": auc_hi,
            "sensitivity": round(sens,4), "specificity": round(spec,4)
        })
        all_probs[name] = (y_prob, y_pred)
        print(f"  {name:<22} AUROC={auroc:.4f} [{auc_lo:.3f}-{auc_hi:.3f}]"
              f"  Sens={sens:.3f}  Spec={spec:.3f}")

    # McNemar's vs Random Forest
    rf_prob, rf_pred = all_probs["Random Forest"]
    for name in models:
        if name == "Random Forest":
            continue
        _, other_pred = all_probs[name]
        p = mcnemar_test(y, rf_pred, other_pred)
        for r in results:
            if r["model"] == name and r["condition"] == condition:
                r["mcnemar_p_vs_rf"] = p

    return results, all_probs


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_results = []

    # 1. Parkinson's — LOSO (primary)
    print("\n" + "="*60)
    print(" PARKINSON'S — LOSO (Subject-Independent)")
    print("="*60)
    X_pk, y_pk, meta_pk = load_parkinsons()
    loso = LeaveOneGroupOut()
    pk_results, pk_probs = evaluate_models(
        X_pk.values, y_pk.values, loso,
        groups=meta_pk["subject_id"].values,
        condition="parkinsons", protocol="LOSO")
    all_results.extend(pk_results)

    # 2. Respiratory — 5-Fold CV
    print("\n" + "="*60)
    print(" RESPIRATORY — 5-Fold Stratified CV")
    print("="*60)
    X_resp, y_resp, _ = load_respiratory()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    resp_results, resp_probs = evaluate_models(
        X_resp.values, y_resp.values, skf,
        condition="respiratory", protocol="5-Fold CV")
    all_results.extend(resp_results)

    # 3. Depression — AVEC 2017 official split
    print("\n" + "="*60)
    print(" DEPRESSION — Official AVEC 2017 Split")
    print("="*60)
    df       = pd.read_csv(BASE/"data/features/daic_woz_covarep_allframes.csv")
    train_df = pd.read_csv(BASE/"data/raw/daic_woz/train_split_Depression_AVEC2017.csv").rename(columns={"Participant_ID":"participant_id"})
    dev_df   = pd.read_csv(BASE/"data/raw/daic_woz/dev_split_Depression_AVEC2017.csv").rename(columns={"Participant_ID":"participant_id"})
    feat_cols = [c for c in df.columns if c.endswith("_mean") or c.endswith("_std")]
    train = df[df.participant_id.isin(train_df.participant_id)]
    dev   = df[df.participant_id.isin(dev_df.participant_id)]
    X_tr, y_tr = train[feat_cols].values, train["PHQ8_Binary"].values
    X_dv, y_dv = dev[feat_cols].values,   dev["PHQ8_Binary"].values

    models = get_models()
    dep_results = []
    dep_probs   = {}
    for name, pipe in models.items():
        from sklearn.decomposition import PCA
        # Add PCA for high-dim depression features
        pipe.steps.insert(-1, ("pca", PCA(n_components=30, random_state=42)))
        pipe.fit(X_tr, y_tr)
        y_prob = pipe.predict_proba(X_dv)[:,1]
        y_pred = (y_prob >= 0.5).astype(int)
        auroc  = roc_auc_score(y_dv, y_prob)
        acc    = accuracy_score(y_dv, y_pred)
        tn,fp,fn,tp = confusion_matrix(y_dv, y_pred).ravel()
        sens = tp/(tp+fn) if (tp+fn)>0 else 0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0
        auc_mean, auc_lo, auc_hi = bootstrap_auroc_ci(y_dv, y_prob, n_boot=500)
        print(f"  {name:<22} AUROC={auroc:.4f} [{auc_lo:.3f}-{auc_hi:.3f}]"
              f"  Sens={sens:.3f}  Spec={spec:.3f}")
        dep_results.append({
            "condition":"depression", "protocol":"AVEC2017 Dev",
            "model":name, "accuracy":round(acc,4), "auroc":round(auroc,4),
            "auroc_ci_lo":auc_lo, "auroc_ci_hi":auc_hi,
            "sensitivity":round(sens,4), "specificity":round(spec,4)
        })
        dep_probs[name] = (y_prob, y_pred)

    # McNemar for depression
    rf_pred_dep = dep_probs["Random Forest"][1]
    for name in models:
        if name == "Random Forest": continue
        p = mcnemar_test(y_dv, rf_pred_dep, dep_probs[name][1])
        for r in dep_results:
            if r["model"] == name:
                r["mcnemar_p_vs_rf"] = p
    all_results.extend(dep_results)

    # Save
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(RESULTS/"baseline_comparison.csv", index=False)

    print("\n" + "="*60)
    print(" SUMMARY TABLE")
    print("="*60)
    print(res_df[["condition","model","auroc","auroc_ci_lo",
                  "auroc_ci_hi","sensitivity","specificity"]].to_string(index=False))
    print(f"\nSaved -> {RESULTS/'baseline_comparison.csv'}")
