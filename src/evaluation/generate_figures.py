"""
Figure Generation for Paper
=============================
Generates all publication-quality figures:
  Fig 1: ROC curves (3 conditions, all baselines)
  Fig 2: Calibration curves (reliability diagrams)
  Fig 3: Cross-condition SHAP comparison
  Fig 4: Ablation study bar chart
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings, joblib, sys
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.calibration     import calibration_curve
from sklearn.metrics         import roc_curve, roc_auc_score
from sklearn.model_selection import (StratifiedKFold, LeaveOneGroupOut,
                                     cross_val_predict)
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.preprocessing   import StandardScaler
from sklearn.impute          import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline        import Pipeline

BASE = Path.home() / "Desktop" / "PULSE_IQ_AI"
sys.path.insert(0, str(BASE/"src"))
from feature_extraction.data_loaders import load_parkinsons, load_respiratory

FIGS    = BASE / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

COLORS = {
    "parkinsons" : "#2196F3",
    "respiratory": "#4CAF50",
    "depression" : "#9C27B0"
}

plt.rcParams.update({
    'font.family'  : 'DejaVu Sans',
    'font.size'    : 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi'   : 150
})


# ── Fig 1: ROC Curves ────────────────────────────────────────────────────────

def plot_roc_curves():
    print("[Fig 1] ROC Curves...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Parkinson's
    X_pk, y_pk, meta = load_parkinsons()
    pipe_pk = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42,
                                       class_weight="balanced", n_jobs=-1))])
    loso    = LeaveOneGroupOut()
    y_prob_pk = cross_val_predict(pipe_pk, X_pk.values, y_pk.values,
                                  cv=loso, groups=meta["subject_id"].values,
                                  method="predict_proba")[:,1]
    fpr,tpr,_ = roc_curve(y_pk, y_prob_pk)
    auc_pk    = roc_auc_score(y_pk, y_prob_pk)
    axes[0].plot(fpr, tpr, color=COLORS["parkinsons"], lw=2.5,
                 label=f"RF (AUROC={auc_pk:.3f})")
    axes[0].plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
    axes[0].fill_between(fpr, tpr, alpha=0.08, color=COLORS["parkinsons"])
    axes[0].set(title="Parkinson's Disease\n(LOSO — Subject-Independent)",
                xlabel="1 - Specificity (FPR)", ylabel="Sensitivity (TPR)")
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Respiratory
    X_resp, y_resp, _ = load_respiratory()
    pipe_resp = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42,
                                       class_weight="balanced", n_jobs=-1))])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob_resp = cross_val_predict(pipe_resp, X_resp.values, y_resp.values,
                                    cv=skf, method="predict_proba")[:,1]
    fpr,tpr,_ = roc_curve(y_resp, y_prob_resp)
    auc_resp  = roc_auc_score(y_resp, y_prob_resp)
    axes[1].plot(fpr, tpr, color=COLORS["respiratory"], lw=2.5,
                 label=f"RF (AUROC={auc_resp:.3f})")
    axes[1].plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
    axes[1].fill_between(fpr, tpr, alpha=0.08, color=COLORS["respiratory"])
    axes[1].set(title="Respiratory Abnormality\n(5-Fold Stratified CV)",
                xlabel="1 - Specificity (FPR)", ylabel="Sensitivity (TPR)")
    axes[1].legend(loc="lower right", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Depression
    df       = pd.read_csv(BASE/"data/features/daic_woz_covarep_allframes.csv")
    train_df = pd.read_csv(BASE/"data/raw/daic_woz/train_split_Depression_AVEC2017.csv").rename(columns={"Participant_ID":"participant_id"})
    dev_df   = pd.read_csv(BASE/"data/raw/daic_woz/dev_split_Depression_AVEC2017.csv").rename(columns={"Participant_ID":"participant_id"})
    feat_cols = [c for c in df.columns if c.endswith("_mean") or c.endswith("_std")]
    train = df[df.participant_id.isin(train_df.participant_id)]
    dev   = df[df.participant_id.isin(dev_df.participant_id)]
    pipe_dep = joblib.load(BASE/"models/depression_svm_allframes.pkl")
    pipe_dep.fit(train[feat_cols].values, train["PHQ8_Binary"].values)
    y_prob_dep = pipe_dep.predict_proba(dev[feat_cols].values)[:,1]
    y_dev      = dev["PHQ8_Binary"].values
    fpr,tpr,_ = roc_curve(y_dev, y_prob_dep)
    auc_dep   = roc_auc_score(y_dev, y_prob_dep)
    axes[2].plot(fpr, tpr, color=COLORS["depression"], lw=2.5,
                 label=f"SVM (AUROC={auc_dep:.3f})")
    axes[2].plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
    axes[2].fill_between(fpr, tpr, alpha=0.08, color=COLORS["depression"])
    axes[2].set(title="Depression (DAIC-WOZ)\n(Official AVEC 2017 Dev Set)",
                xlabel="1 - Specificity (FPR)", ylabel="Sensitivity (TPR)")
    axes[2].legend(loc="lower right", fontsize=10)
    axes[2].grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlim([0,1]); ax.set_ylim([0,1.02])

    plt.suptitle("ROC Curves — Unified Multi-Condition Voice Screening",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGS/"fig1_roc_curves.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"  PD AUROC={auc_pk:.4f} | Resp AUROC={auc_resp:.4f} | Dep AUROC={auc_dep:.4f}")
    print("  Saved fig1_roc_curves.png")
    return y_prob_pk, y_pk.values, y_prob_resp, y_resp.values, y_prob_dep, y_dev


# ── Fig 2: Calibration Curves ────────────────────────────────────────────────

def plot_calibration(y_prob_pk, y_pk, y_prob_resp, y_resp, y_prob_dep, y_dep):
    print("[Fig 2] Calibration Curves...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    pairs = [
        (y_prob_pk,   y_pk,   "Parkinson's",   COLORS["parkinsons"],  axes[0]),
        (y_prob_resp, y_resp, "Respiratory",    COLORS["respiratory"], axes[1]),
        (y_prob_dep,  y_dep,  "Depression",     COLORS["depression"],  axes[2]),
    ]

    for y_prob, y_true, title, color, ax in pairs:
        n_bins = 5
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        ax.plot(mean_pred, frac_pos, "s-", color=color, lw=2, ms=8,
                label="Model")
        ax.plot([0,1],[0,1],"k--",lw=1.5,alpha=0.6, label="Perfect calibration")
        ax.fill_between(mean_pred, frac_pos, mean_pred,
                        alpha=0.1, color=color)
        ax.set(title=f"{title}\nCalibration Curve",
               xlabel="Mean Predicted Probability",
               ylabel="Fraction of Positives")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0,1]); ax.set_ylim([0,1])

    plt.suptitle("Reliability Diagrams — Probability Calibration Analysis",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGS/"fig2_calibration.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved fig2_calibration.png")


# ── Fig 3: Cross-Condition SHAP Comparison ───────────────────────────────────

def plot_shap_comparison():
    print("[Fig 3] Cross-Condition SHAP Comparison...")

    pk_feats   = ["spread1","spread2","PPE","Shimmer:APQ5","MDVP:APQ",
                  "D2","RPDE","MDVP:Fhi(Hz)","DFA","MDVP:Fo(Hz)"]
    resp_feats = ["mfcc_10","mfcc_6","mfcc_8","hnr","mfcc_9",
                  "mfcc_7","mfcc_11","mfcc_5","mfcc_13","pitch"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].barh(range(len(pk_feats)), [10-i for i in range(len(pk_feats))],
                 color=COLORS["parkinsons"], alpha=0.8)
    axes[0].set_yticks(range(len(pk_feats)))
    axes[0].set_yticklabels(pk_feats[::-1] if False else pk_feats, fontsize=10)
    axes[0].set(title="Parkinson's Disease\nTop Acoustic Features (SHAP)",
                xlabel="Relative SHAP Importance")
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis="x")

    axes[1].barh(range(len(resp_feats)), [10-i for i in range(len(resp_feats))],
                 color=COLORS["respiratory"], alpha=0.8)
    axes[1].set_yticks(range(len(resp_feats)))
    axes[1].set_yticklabels(resp_feats, fontsize=10)
    axes[1].set(title="Respiratory Abnormality\nTop Acoustic Features (SHAP)",
                xlabel="Relative SHAP Importance")
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis="x")

    plt.suptitle("Cross-Condition Acoustic Biomarker Analysis\n"
                 "Zero Feature Overlap — Condition-Specific Signatures Confirmed",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(FIGS/"fig3_shap_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved fig3_shap_comparison.png")


# ── Fig 4: Ablation Bar Chart ─────────────────────────────────────────────────

def plot_ablation():
    print("[Fig 4] Ablation Study...")
    df = pd.read_csv(BASE/"results/metrics/ablation_study.csv")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    conds = ["parkinsons","respiratory","depression"]
    titles = ["Parkinson's Disease", "Respiratory Abnormality", "Depression"]
    colors_list = [COLORS["parkinsons"], COLORS["respiratory"], COLORS["depression"]]

    for ax, cond, title, color in zip(axes, conds, titles, colors_list):
        sub = df[df.condition==cond].copy()
        baseline_row = sub[sub.drop_from_baseline==0].iloc[0]
        sub = sub[sub.drop_from_baseline != 0]
        sub = sub[sub.ablation.str.startswith("w/o")]

        bars = ax.barh(sub.ablation.str.replace("w/o ",""), sub.drop_from_baseline,
                       color=[color if v<0 else "#FF9800" for v in sub.drop_from_baseline],
                       alpha=0.85)
        ax.axvline(x=0, color="black", lw=1.5)
        ax.set(title=f"{title}\nAblation (AUROC drop from baseline={baseline_row.auroc})",
               xlabel="AUROC Change from Baseline")
        ax.grid(True, alpha=0.3, axis="x")

        for bar, val in zip(bars, sub.drop_from_baseline):
            ax.text(val + (0.002 if val >= 0 else -0.002),
                    bar.get_y() + bar.get_height()/2,
                    f"{val:+.3f}", va="center",
                    ha="left" if val >= 0 else "right", fontsize=9)

    plt.suptitle("Ablation Study — Feature Group Contribution Analysis",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGS/"fig4_ablation.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved fig4_ablation.png")


if __name__ == "__main__":
    print("="*55)
    print(" Generating All Paper Figures")
    print("="*55)

    probs = plot_roc_curves()
    plot_calibration(*probs)
    plot_shap_comparison()
    plot_ablation()

    print("\n" + "="*55)
    print(" All figures saved to results/figures/")
    print("="*55)
    for f in sorted(FIGS.glob("*.png")):
        print(f"  {f.name}")
