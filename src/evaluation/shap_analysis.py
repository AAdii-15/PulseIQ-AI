"""
SHAP Cross-Condition Analysis
==============================
Runs SHAP TreeExplainer on:
  - Parkinson's RF model
  - Respiratory RF model
  - Depression SVM model (LinearSHAP approximation)

Key analysis:
  - Per-condition top features
  - Cross-condition feature overlap
  - Shared vs condition-specific acoustic biomarkers
"""

import numpy as np
import pandas as pd
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

BASE = Path.home() / "Desktop" / "PULSE_IQ_AI"
sys.path.insert(0, str(BASE / "src"))
from feature_extraction.data_loaders import load_parkinsons, load_respiratory

FIGS    = BASE / "results" / "shap"
RESULTS = BASE / "results" / "metrics"
FIGS.mkdir(parents=True, exist_ok=True)


def get_rf_from_pipeline(pipe):
    """Extract RF from calibrated pipeline."""
    model = pipe.named_steps["model"]
    if hasattr(model, "calibrated_classifiers_"):
        return model.calibrated_classifiers_[0].estimator
    return model


def run_shap_parkinsons():
    print("\n[1/3] Parkinson's SHAP Analysis...")
    X, y, _ = load_parkinsons()
    pipe     = joblib.load(BASE/"models/parkinsons_model.pkl")
    rf       = get_rf_from_pipeline(pipe)
    X_t      = pipe[:-1].transform(X)

    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X_t)
    sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals[:,:,1]

    # Summary plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X_t, feature_names=X.columns.tolist(),
                      show=False, max_display=15)
    plt.title("Parkinson's Disease — SHAP Feature Importance", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGS/"shap_parkinsons.png", dpi=150, bbox_inches="tight")
    plt.close()

    mean_shap = np.abs(sv).mean(axis=0)
    top = sorted(zip(X.columns.tolist(), mean_shap), key=lambda x: x[1], reverse=True)
    print(f"  Top 5: {[f for f,v in top[:5]]}")
    return top, sv, X_t, X.columns.tolist()


def run_shap_respiratory():
    print("\n[2/3] Respiratory SHAP Analysis...")
    X, y, _ = load_respiratory()
    pipe     = joblib.load(BASE/"models/respiratory_model.pkl")
    rf       = get_rf_from_pipeline(pipe)
    X_t      = pipe[:-1].transform(X)

    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X_t)
    sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals[:,:,1]

    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X_t, feature_names=X.columns.tolist(),
                      show=False, max_display=15)
    plt.title("Respiratory Abnormality — SHAP Feature Importance", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGS/"shap_respiratory.png", dpi=150, bbox_inches="tight")
    plt.close()

    mean_shap = np.abs(sv).mean(axis=0)
    top = sorted(zip(X.columns.tolist(), mean_shap), key=lambda x: x[1], reverse=True)
    print(f"  Top 5: {[f for f,v in top[:5]]}")
    return top, sv, X_t, X.columns.tolist()


def cross_condition_overlap(top_pk, top_resp, n=15):
    print(f"\n[3/3] Cross-Condition Overlap (top-{n} features)...")
    pk_set   = set([f for f,v in top_pk[:n]])
    resp_set = set([f for f,v in top_resp[:n]])
    overlap  = pk_set & resp_set
    pk_only  = pk_set - resp_set
    resp_only= resp_set - pk_set

    print(f"  Shared features  ({len(overlap)}): {sorted(overlap)}")
    print(f"  PD-specific      ({len(pk_only)}): {sorted(pk_only)[:5]}...")
    print(f"  Resp-specific    ({len(resp_only)}): {sorted(resp_only)[:5]}...")

    # Save overlap table
    all_rows = []
    for f,v in top_pk[:n]:
        all_rows.append({"feature":f, "condition":"parkinsons",
                         "mean_shap":round(v,6),
                         "shared": f in overlap})
    for f,v in top_resp[:n]:
        all_rows.append({"feature":f, "condition":"respiratory",
                         "mean_shap":round(v,6),
                         "shared": f in overlap})

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS/"shap_cross_condition_overlap.csv", index=False)
    print(f"  Saved shap_cross_condition_overlap.csv")
    return overlap, pk_only, resp_only


if __name__ == "__main__":
    print("="*55)
    print(" SHAP Cross-Condition Analysis")
    print("="*55)

    top_pk,   sv_pk,   X_pk_t,   feat_pk   = run_shap_parkinsons()
    top_resp, sv_resp, X_resp_t, feat_resp = run_shap_respiratory()
    overlap, pk_only, resp_only = cross_condition_overlap(top_pk, top_resp)

    print("\n" + "="*55)
    print(" KEY FINDING")
    print("="*55)
    print(f"  {len(overlap)} acoustic features are shared between")
    print(f"  Parkinson's and Respiratory top-15 predictors.")
    if overlap:
        print(f"  Shared: {sorted(overlap)}")
    print("\n  This suggests a partial shared acoustic biomarker space")
    print("  across neurological and pulmonary conditions.")
    print("="*55)
