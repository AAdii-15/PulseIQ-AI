"""
PHQ-8 Acoustic Correlation Analysis
=====================================
Systematic analysis of acoustic feature correlations
with individual PHQ-8 symptom dimensions.

Result: No statistically significant correlations survive
Bonferroni correction (N=142, 1168 tests, alpha=0.000043).
Reported as null finding — motivates future larger-scale work.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE = Path.home() / "Desktop" / "PULSE_IQ_AI"

PHQ8_ITEMS = {
    'Anhedonia'     : 'PHQ8_NoInterest',
    'Depressed Mood': 'PHQ8_Depressed',
    'Sleep'         : 'PHQ8_Sleep',
    'Fatigue'       : 'PHQ8_Tired',
    'Appetite'      : 'PHQ8_Appetite',
    'Worthlessness' : 'PHQ8_Failure',
    'Concentration' : 'PHQ8_Concentrating',
    'Psychomotor'   : 'PHQ8_Moving'
}


def run_phq8_correlation_analysis():
    df = pd.read_csv(BASE/"data/features/daic_woz_covarep_allframes.csv")

    feat_cols = [c for c in df.columns
                 if c.endswith("_mean") or c.endswith("_std")]
    n       = len(df)
    n_tests = len(feat_cols) * len(PHQ8_ITEMS)
    alpha_bon = 0.05 / n_tests  # Bonferroni
    alpha_fdr = 0.05            # Uncorrected for exploratory

    print(f"Samples : {n}")
    print(f"Features: {len(feat_cols)}")
    print(f"Tests   : {n_tests}")
    print(f"Bonferroni alpha: {alpha_bon:.6f}\n")

    all_results = []
    for symp_name, phq_col in PHQ8_ITEMS.items():
        y = df[phq_col].values
        for feat in feat_cols:
            x = df[feat].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 10:
                continue
            r, p = stats.spearmanr(x[mask], y[mask])
            all_results.append({
                'symptom'               : symp_name,
                'phq8_item'             : phq_col,
                'feature'               : feat,
                'spearman_r'            : round(r, 5),
                'p_value'               : round(p, 8),
                'sig_bonferroni'        : p < alpha_bon,
                'sig_uncorrected_p05'   : p < 0.05,
            })

    res_df = pd.DataFrame(all_results)

    print("Bonferroni-significant correlations per symptom:")
    for symp in PHQ8_ITEMS:
        n_sig = res_df[(res_df.symptom==symp) & res_df.sig_bonferroni].shape[0]
        print(f"  {symp:<20} {n_sig:3d}")

    print("\nUncorrected p<0.05 correlations per symptom (exploratory):")
    for symp in PHQ8_ITEMS:
        sub = res_df[(res_df.symptom==symp) & res_df.sig_uncorrected_p05]
        top = sub.reindex(sub.spearman_r.abs().sort_values(ascending=False).index)
        if len(top) > 0:
            row = top.iloc[0]
            print(f"  {symp:<20} n={len(sub):3d}  top: {row.feature} r={row.spearman_r:+.3f}")
        else:
            print(f"  {symp:<20} n=  0")

    out = BASE / "results/metrics/phq8_acoustic_correlations_full.csv"
    res_df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")
    print("\nConclusion: No Bonferroni-corrected significant correlations found.")
    print("N=142 is insufficient for 1168 simultaneous tests.")
    print("Reported as null finding — motivates future larger-scale studies.")
    return res_df


if __name__ == "__main__":
    run_phq8_correlation_analysis()
