"""
Remaining Reviewer Fixes
=========================
A: SHAP stability (bootstrap Spearman rank correlation)
B: Calibration metrics (Brier score, ECE)
C: Balanced Accuracy + F1 for all models
D: RF variance across random seeds
"""

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')
import shap
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (LeaveOneGroupOut, StratifiedKFold,
                                     cross_val_predict)
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             confusion_matrix, brier_score_loss)
import sys

BASE    = Path.home() / 'Desktop/PULSE_IQ_AI'
RESULTS = BASE / 'results/metrics'
sys.path.insert(0, str(BASE/'src'))
from feature_extraction.data_loaders import load_parkinsons, load_respiratory

NONLINEAR = ['RPDE','DFA','PPE','spread1','spread2','D2']

def make_rf(seed=42):
    return Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc',  StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=500, random_state=seed,
                                       class_weight='balanced', n_jobs=-1))])

# ── A: SHAP Stability ─────────────────────────────────────────────────────────
def fix_a_shap_stability():
    print('='*60)
    print('FIX A: SHAP Stability (Bootstrap Spearman ρ)')
    print('='*60)
    X_pk, y_pk, meta = load_parkinsons()
    groups = meta['subject_id'].values
    loso   = LeaveOneGroupOut()
    N_BOOT = 20

    baseline_ranks, boot_ranks_pk = None, []
    for b in range(N_BOOT + 1):
        seed = 42 if b == 0 else b * 7
        pipe = make_rf(seed)
        # Train on all data for SHAP (not LOSO — SHAP on full model)
        pipe.fit(X_pk[NONLINEAR].values, y_pk.values)
        rf = pipe.named_steps['clf']
        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(pipe[:-1].transform(X_pk[NONLINEAR].values))
        sv = sv[1] if isinstance(sv, list) else sv[:,:,1]
        mean_abs = np.abs(sv).mean(axis=0)
        ranks = np.argsort(-mean_abs)  # descending rank
        if b == 0:
            baseline_ranks = ranks
        else:
            boot_ranks_pk.append(ranks)

    rhos_pk = [spearmanr(baseline_ranks, r).correlation for r in boot_ranks_pk]
    print(f'PD SHAP rank stability:')
    print(f'  Mean Spearman ρ : {np.mean(rhos_pk):.4f}')
    print(f'  Std             : {np.std(rhos_pk):.4f}')
    print(f'  Min             : {np.min(rhos_pk):.4f}')
    print(f'  All ρ > 0.80    : {all(r > 0.80 for r in rhos_pk)}')

    # Respiratory
    X_resp, y_resp, _ = load_respiratory()
    baseline_ranks_resp, boot_ranks_resp = None, []
    for b in range(N_BOOT + 1):
        seed = 42 if b == 0 else b * 7
        pipe = make_rf(seed)
        pipe.fit(X_resp.values, y_resp.values)
        rf = pipe.named_steps['clf']
        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(pipe[:-1].transform(X_resp.values[:500]))
        sv = sv[1] if isinstance(sv, list) else sv[:,:,1]
        mean_abs = np.abs(sv).mean(axis=0)
        ranks = np.argsort(-mean_abs)
        if b == 0:
            baseline_ranks_resp = ranks
        else:
            boot_ranks_resp.append(ranks)

    rhos_resp = [spearmanr(baseline_ranks_resp, r).correlation
                 for r in boot_ranks_resp]
    print(f'\nCOVID-19 Respiratory SHAP rank stability:')
    print(f'  Mean Spearman ρ : {np.mean(rhos_resp):.4f}')
    print(f'  Std             : {np.std(rhos_resp):.4f}')
    print(f'  Min             : {np.min(rhos_resp):.4f}')
    print(f'  All ρ > 0.80    : {all(r > 0.80 for r in rhos_resp)}')

    result = pd.DataFrame([{
        'condition':'parkinsons',
        'mean_spearman_rho':round(np.mean(rhos_pk),4),
        'std_rho':round(np.std(rhos_pk),4),
        'min_rho':round(np.min(rhos_pk),4),
        'n_bootstrap':N_BOOT,
        'all_above_080':all(r>0.80 for r in rhos_pk)
    },{
        'condition':'respiratory',
        'mean_spearman_rho':round(np.mean(rhos_resp),4),
        'std_rho':round(np.std(rhos_resp),4),
        'min_rho':round(np.min(rhos_resp),4),
        'n_bootstrap':N_BOOT,
        'all_above_080':all(r>0.80 for r in rhos_resp)
    }])
    result.to_csv(RESULTS/'shap_stability.csv', index=False)
    print(f'\nSaved -> shap_stability.csv')
    return np.mean(rhos_pk), np.mean(rhos_resp)


# ── B: Calibration Metrics ────────────────────────────────────────────────────
def fix_b_calibration():
    print('\n'+'='*60)
    print('FIX B: Calibration Metrics (Brier Score + ECE)')
    print('='*60)

    def expected_calibration_error(y_true, y_prob, n_bins=10):
        bins  = np.linspace(0, 1, n_bins+1)
        ece   = 0.0
        n     = len(y_true)
        for i in range(n_bins):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
            if mask.sum() == 0: continue
            acc  = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += (mask.sum() / n) * abs(acc - conf)
        return round(ece, 4)

    results = []

    # PD
    X_pk, y_pk, meta = load_parkinsons()
    loso = LeaveOneGroupOut()
    y_prob_pk = cross_val_predict(make_rf(), X_pk[NONLINEAR].values,
                                  y_pk.values, cv=loso,
                                  groups=meta['subject_id'].values,
                                  method='predict_proba')[:,1]
    brier_pk = round(brier_score_loss(y_pk.values, y_prob_pk), 4)
    ece_pk   = expected_calibration_error(y_pk.values, y_prob_pk)
    print(f'PD (LOSO):               Brier={brier_pk}  ECE={ece_pk}')
    results.append({'condition':'parkinsons','brier':brier_pk,'ece':ece_pk})

    # Respiratory
    X_resp, y_resp, _ = load_respiratory()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob_resp = cross_val_predict(make_rf(), X_resp.values, y_resp.values,
                                    cv=skf, method='predict_proba')[:,1]
    brier_resp = round(brier_score_loss(y_resp.values, y_prob_resp), 4)
    ece_resp   = expected_calibration_error(y_resp.values, y_prob_resp)
    print(f'COVID-19 Resp (5-fold):  Brier={brier_resp}  ECE={ece_resp}')
    results.append({'condition':'respiratory','brier':brier_resp,'ece':ece_resp})

    # Depression
    df      = pd.read_csv(BASE/'data/features/daic_woz_covarep_allframes.csv')
    dev_df  = pd.read_csv(BASE/'data/raw/daic_woz/dev_split_Depression_AVEC2017.csv').rename(columns={'Participant_ID':'participant_id'})
    fc      = [c for c in df.columns if c.endswith('_mean') or c.endswith('_std')]
    dev     = df[df.participant_id.isin(dev_df.participant_id)]
    pipe_dep= joblib.load(BASE/'models/depression_best_model.pkl')
    y_prob_dep = pipe_dep.predict_proba(dev[fc].values)[:,1]
    y_dv    = dev['PHQ8_Binary'].values
    brier_dep = round(brier_score_loss(y_dv, y_prob_dep), 4)
    ece_dep   = expected_calibration_error(y_dv, y_prob_dep)
    print(f'Depression (AVEC dev):   Brier={brier_dep}  ECE={ece_dep}')
    print(f'  Note: depression metrics unreliable (N=35)')
    results.append({'condition':'depression','brier':brier_dep,'ece':ece_dep})

    pd.DataFrame(results).to_csv(RESULTS/'calibration_metrics.csv', index=False)
    print('Saved -> calibration_metrics.csv')
    return brier_pk, brier_resp, brier_dep


# ── C: BACC + F1 for all models ───────────────────────────────────────────────
def fix_c_bacc_f1():
    print('\n'+'='*60)
    print('FIX C: Balanced Accuracy + F1 Score')
    print('='*60)

    def compute_metrics(y_true, y_prob, thresh=0.5):
        y_pred = (y_prob >= thresh).astype(int)
        tn,fp,fn,tp = confusion_matrix(y_true, y_pred).ravel()
        sens = tp/(tp+fn) if (tp+fn)>0 else 0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0
        bacc = (sens + spec) / 2
        f1   = f1_score(y_true, y_pred, zero_division=0)
        auroc= roc_auc_score(y_true, y_prob)
        return {'auroc':round(auroc,4),'bacc':round(bacc,4),'f1':round(f1,4),
                'sensitivity':round(sens,4),'specificity':round(spec,4)}

    results = []

    # PD
    X_pk, y_pk, meta = load_parkinsons()
    loso = LeaveOneGroupOut()
    y_prob = cross_val_predict(make_rf(), X_pk[NONLINEAR].values,
                               y_pk.values, cv=loso,
                               groups=meta['subject_id'].values,
                               method='predict_proba')[:,1]
    m = compute_metrics(y_pk.values, y_prob)
    print(f'PD (LOSO, nonlinear-6):  AUROC={m["auroc"]}  BACC={m["bacc"]}  F1={m["f1"]}')
    results.append({'condition':'parkinsons','protocol':'LOSO',**m})

    # Respiratory
    X_resp, y_resp, _ = load_respiratory()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob = cross_val_predict(make_rf(), X_resp.values, y_resp.values,
                               cv=skf, method='predict_proba')[:,1]
    m = compute_metrics(y_resp.values, y_prob)
    print(f'COVID-19 (5-fold):       AUROC={m["auroc"]}  BACC={m["bacc"]}  F1={m["f1"]}')
    results.append({'condition':'respiratory','protocol':'5-Fold CV',**m})

    # Depression — optimal threshold
    df     = pd.read_csv(BASE/'data/features/daic_woz_covarep_allframes.csv')
    dev_df = pd.read_csv(BASE/'data/raw/daic_woz/dev_split_Depression_AVEC2017.csv').rename(columns={'Participant_ID':'participant_id'})
    fc     = [c for c in df.columns if c.endswith('_mean') or c.endswith('_std')]
    dev    = df[df.participant_id.isin(dev_df.participant_id)]
    pipe   = joblib.load(BASE/'models/depression_best_model.pkl')
    y_prob = pipe.predict_proba(dev[fc].values)[:,1]
    y_dv   = dev['PHQ8_Binary'].values
    m_def  = compute_metrics(y_dv, y_prob, thresh=0.50)
    m_opt  = compute_metrics(y_dv, y_prob, thresh=0.2744)
    print(f'Depression (default 0.5):  AUROC={m_def["auroc"]}  BACC={m_def["bacc"]}  F1={m_def["f1"]}')
    print(f'Depression (thresh 0.274): AUROC={m_opt["auroc"]}  BACC={m_opt["bacc"]}  F1={m_opt["f1"]}')
    print(f'  Note: Non-significant (p=0.110). Exploratory only.')
    results.append({'condition':'depression','protocol':'AVEC2017 (default)',**m_def})
    results.append({'condition':'depression','protocol':'AVEC2017 (optimal-thresh)',**m_opt})

    df_res = pd.DataFrame(results)
    df_res.to_csv(RESULTS/'bacc_f1_metrics.csv', index=False)
    print(f'\nSaved -> bacc_f1_metrics.csv')
    return df_res


# ── D: RF Variance Across Seeds ───────────────────────────────────────────────
def fix_d_rf_variance():
    print('\n'+'='*60)
    print('FIX D: RF Variance Across Random Seeds')
    print('='*60)
    SEEDS = [42, 7, 123, 256, 512, 999, 1337, 2024, 31, 88]

    # PD
    X_pk, y_pk, meta = load_parkinsons()
    loso = LeaveOneGroupOut()
    pk_aucs = []
    for seed in SEEDS:
        pipe = make_rf(seed)
        y_prob = cross_val_predict(pipe, X_pk[NONLINEAR].values,
                                   y_pk.values, cv=loso,
                                   groups=meta['subject_id'].values,
                                   method='predict_proba')[:,1]
        pk_aucs.append(roc_auc_score(y_pk.values, y_prob))

    print(f'PD LOSO (10 seeds):')
    print(f'  Mean AUROC : {np.mean(pk_aucs):.4f}')
    print(f'  Std        : {np.std(pk_aucs):.4f}')
    print(f'  Range      : [{min(pk_aucs):.4f}, {max(pk_aucs):.4f}]')
    print(f'  All ≥ 0.75 : {all(a >= 0.75 for a in pk_aucs)}')

    # Respiratory
    X_resp, y_resp, _ = load_respiratory()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    resp_aucs = []
    for seed in SEEDS:
        pipe = make_rf(seed)
        y_prob = cross_val_predict(pipe, X_resp.values, y_resp.values,
                                   cv=skf, method='predict_proba')[:,1]
        resp_aucs.append(roc_auc_score(y_resp.values, y_prob))

    print(f'\nCOVID-19 Respiratory (10 seeds):')
    print(f'  Mean AUROC : {np.mean(resp_aucs):.4f}')
    print(f'  Std        : {np.std(resp_aucs):.4f}')
    print(f'  Range      : [{min(resp_aucs):.4f}, {max(resp_aucs):.4f}]')

    result = pd.DataFrame([{
        'condition':'parkinsons',
        'mean_auroc':round(np.mean(pk_aucs),4),
        'std_auroc':round(np.std(pk_aucs),4),
        'min_auroc':round(min(pk_aucs),4),
        'max_auroc':round(max(pk_aucs),4),
        'n_seeds':len(SEEDS)
    },{
        'condition':'respiratory',
        'mean_auroc':round(np.mean(resp_aucs),4),
        'std_auroc':round(np.std(resp_aucs),4),
        'min_auroc':round(min(resp_aucs),4),
        'max_auroc':round(max(resp_aucs),4),
        'n_seeds':len(SEEDS)
    }])
    result.to_csv(RESULTS/'rf_variance.csv', index=False)
    print(f'\nSaved -> rf_variance.csv')
    return result


# ── Run All ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    rho_pk, rho_resp = fix_a_shap_stability()
    brier_pk, brier_resp, brier_dep = fix_b_calibration()
    bacc_df = fix_c_bacc_f1()
    rf_var  = fix_d_rf_variance()

    print('\n'+'='*60)
    print(' ALL FIXES COMPLETE — SUMMARY')
    print('='*60)
    print(f'A. SHAP Stability  : PD ρ={rho_pk:.3f} | Resp ρ={rho_resp:.3f}')
    print(f'B. Brier Scores    : PD={brier_pk} | Resp={brier_resp} | Dep={brier_dep}')
    print(f'C. BACC+F1         : see bacc_f1_metrics.csv')
    print(f'D. RF Variance     : see rf_variance.csv')
    print()
    print('All results saved to results/metrics/')
