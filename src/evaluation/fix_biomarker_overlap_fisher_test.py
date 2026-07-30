"""
Corrected regeneration of results/metrics/biomarker_overlap_fisher_test.csv

Fixes the bug identified and verified in verify_fisher_test.py: the
previous version of this file was generated with the wrong `alternative`
direction in scipy.stats.fisher_exact (giving fisher_p=1.0 for every row,
regardless of true significance). This version uses the correct one-sided
test (alternative='less', testing for lower-than-chance overlap) and
derives the interpretation string from the actual computed p-value
instead of hardcoding it.

Schema is unchanged from the original file for drop-in compatibility:
    top_n, pk_features, resp_features, overlap, odds_ratio, fisher_p, interpretation

Universe feature lists are the real column names from
src/feature_extraction/data_loaders.py (load_parkinsons / load_respiratory),
copied here rather than re-imported to keep this script dependency-free
(no need to load the raw UCI/Coswara CSVs just to read 41 column names).

Top-k SHAP rankings are read directly from the already-computed and
already-verified results/metrics/shap_cross_condition_overlap.csv.
"""
import sys
import shutil
import pandas as pd
from pathlib import Path
from scipy.stats import fisher_exact

BASE = Path.home() / 'Desktop/PULSE_IQ_AI'
RESULTS = BASE / 'results/metrics'
SRC_CSV = RESULTS / 'shap_cross_condition_overlap.csv'
OUT_CSV = RESULTS / 'biomarker_overlap_fisher_test.csv'
BACKUP_CSV = RESULTS / 'biomarker_overlap_fisher_test_OLD.csv'

# ── Feature universes (from src/feature_extraction/data_loaders.py) ───────────
PD_UNIVERSE = {
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ",
    "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
    "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
}
COVID_UNIVERSE = {
    "pitch", "spectral_centroid", "zcr", "jitter", "shimmer", "hnr",
    "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5", "mfcc_6", "mfcc_7",
    "mfcc_8", "mfcc_9", "mfcc_10", "mfcc_11", "mfcc_12", "mfcc_13"
}
assert len(PD_UNIVERSE) == 22 and len(COVID_UNIVERSE) == 19
N_UNIVERSE = len(PD_UNIVERSE | COVID_UNIVERSE)  # 41, strict case-sensitive union

if __name__ == '__main__':
    if not SRC_CSV.exists():
        print(f'ERROR: {SRC_CSV} not found. Run src/evaluation/shap_analysis.py first.')
        sys.exit(1)

    if OUT_CSV.exists():
        shutil.copy(OUT_CSV, BACKUP_CSV)
        print(f'Backed up existing (incorrect) file -> {BACKUP_CSV}')

    df = pd.read_csv(SRC_CSV)
    pk_ranked = df[df.condition == 'parkinsons'].sort_values(
        'mean_shap', ascending=False)['feature'].tolist()
    resp_ranked = df[df.condition == 'respiratory'].sort_values(
        'mean_shap', ascending=False)['feature'].tolist()
    print(f'Loaded {len(pk_ranked)} ranked PD features, '
          f'{len(resp_ranked)} ranked COVID features from {SRC_CSV.name}')

    rows = []
    print(f'\nUniverse N = {N_UNIVERSE} (PD 22 + COVID-19 19, case-sensitive union)\n')

    for top_n in [5, 10, 15]:
        pk_set = set(pk_ranked[:top_n])
        resp_set = set(resp_ranked[:top_n])
        overlap = len(pk_set & resp_set)

        a = overlap
        b = top_n - overlap
        c = top_n - overlap
        d = N_UNIVERSE - (2 * top_n - overlap)
        table = [[a, b], [c, d]]

        odds_ratio, fisher_p = fisher_exact(table, alternative='less')

        if fisher_p < 0.001:
            interpretation = f'Significantly disjoint (p<0.001)'
        elif fisher_p < 0.05:
            interpretation = f'Significantly disjoint (p<0.05)'
        else:
            interpretation = 'Not significant (chance-level overlap)'

        rows.append({
            'top_n': top_n,
            'pk_features': top_n,
            'resp_features': top_n,
            'overlap': overlap,
            'odds_ratio': round(odds_ratio, 4),
            'fisher_p': round(fisher_p, 6),
            'interpretation': interpretation
        })
        print(f'  top_n={top_n:2d}  overlap={overlap}  odds_ratio={odds_ratio:.4f}  '
              f'fisher_p={fisher_p:.6g}  -> {interpretation}')

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f'\nSaved corrected file -> {OUT_CSV}')
    print('\nNote: previous file always used alternative=\'greater\' (or the scipy')
    print('default), which is trivially non-significant whenever overlap=0. This')
    print('version uses alternative=\'less\', the direction that actually tests')
    print('the "significantly disjoint" claim made in the paper.')
