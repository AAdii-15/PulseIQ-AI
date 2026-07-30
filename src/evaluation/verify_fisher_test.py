"""
Verification: PD vs COVID-19 top-k SHAP feature overlap significance test.

Uses the ACTUAL saved top-15 SHAP rankings from
results/metrics/shap_cross_condition_overlap.csv and the full feature
universes from src/feature_extraction/data_loaders.py -- nothing here is
re-simulated or approximated.

Question: is the observed zero overlap between PD's and COVID-19's top-k
SHAP features more disjoint than chance would produce, given both are
drawn from a combined universe of N distinct named features?

This tests BOTH the correct one-sided direction (overlap SMALLER than
chance -> evidence of disjointness) and the reversed direction, to show
directly whether a wrong `alternative=` argument in scipy could explain
the fisher_p=1.0 currently in biomarker_overlap_fisher_test.csv.
"""
import numpy as np
from scipy.stats import fisher_exact, hypergeom

# ── Real data: full feature universes (from data_loaders.py) ──────────────────
PD_UNIVERSE = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ",
    "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
    "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
]
COVID_UNIVERSE = [
    "pitch", "spectral_centroid", "zcr", "jitter", "shimmer", "hnr",
    "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5", "mfcc_6", "mfcc_7",
    "mfcc_8", "mfcc_9", "mfcc_10", "mfcc_11", "mfcc_12", "mfcc_13"
]
assert len(PD_UNIVERSE) == 22, "PD universe should be 22 features"
assert len(COVID_UNIVERSE) == 19, "COVID universe should be 19 features"

# ── Real data: top-15 SHAP rankings, descending mean|SHAP|
#    (from results/metrics/shap_cross_condition_overlap.csv) ──────────────────
PD_TOP15 = [
    "spread1", "spread2", "PPE", "Shimmer:APQ5", "MDVP:APQ",
    "MDVP:Fhi(Hz)", "DFA", "D2", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
    "Shimmer:APQ3", "MDVP:Fo(Hz)", "Shimmer:DDA", "Jitter:DDP", "MDVP:RAP"
]
COVID_TOP15 = [
    "mfcc_10", "mfcc_6", "mfcc_8", "hnr", "mfcc_9", "shimmer",
    "spectral_centroid", "zcr", "jitter", "mfcc_12", "mfcc_5",
    "mfcc_11", "mfcc_4", "pitch", "mfcc_7"
]
assert len(PD_TOP15) == 15 and len(COVID_TOP15) == 15

# ── Universe size: two definitions, to sensitivity-check the result ───────────
N_STRICT = len(set(PD_UNIVERSE) | set(COVID_UNIVERSE))              # exact string match
N_CASEFOLD = len(set(f.lower() for f in PD_UNIVERSE) |
                 set(f.lower() for f in COVID_UNIVERSE))             # HNR == hnr merged

print("=" * 72)
print("STEP 1 — Universe size sanity check")
print("=" * 72)
print(f"PD universe:    {len(PD_UNIVERSE)} features")
print(f"COVID universe: {len(COVID_UNIVERSE)} features")
print(f"Union (strict, case-sensitive):    N = {N_STRICT}")
print(f"Union (case-folded, HNR=hnr):       N = {N_CASEFOLD}")
print()

for k in [5, 10, 15]:
    pd_set = set(PD_TOP15[:k])
    cov_set = set(COVID_TOP15[:k])
    overlap = len(pd_set & cov_set)

    print("=" * 72)
    print(f"STEP 2 — top-{k} overlap and significance test")
    print("=" * 72)
    print(f"PD top-{k}:    {sorted(pd_set)}")
    print(f"COVID top-{k}: {sorted(cov_set)}")
    print(f"Observed overlap: {overlap}")

    for N_label, N in [("strict", N_STRICT), ("case-folded", N_CASEFOLD)]:
        # Hypergeometric: draw k items from N total, K=k are "marked" (COVID's set),
        # what's P(X <= observed overlap) under COVID's set being fixed and PD's
        # set being an independent draw of size k from the same N-item universe?
        p_less_hyper = hypergeom.cdf(overlap, N, k, k)

        # 2x2 contingency table version (equivalent test, standard presentation):
        #                  in COVID top-k   not in COVID top-k
        # in PD top-k            a                  b
        # not in PD top-k        c                  d
        a = overlap
        b = k - overlap
        c = k - overlap
        d = N - (2 * k - overlap)
        table = [[a, b], [c, d]]

        _, p_less = fisher_exact(table, alternative='less')
        _, p_greater = fisher_exact(table, alternative='greater')
        _, p_two = fisher_exact(table, alternative='two-sided')

        print(f"\n  Universe N={N} ({N_label}), contingency table {table}:")
        print(f"    fisher_exact(alternative='less')      p = {p_less:.6g}"
              f"   <- correct direction for 'significantly disjoint'")
        print(f"    fisher_exact(alternative='greater')   p = {p_greater:.6g}"
              f"   <- wrong direction; trivially ~1.0 when overlap=0")
        print(f"    fisher_exact(alternative='two-sided') p = {p_two:.6g}")
        print(f"    hypergeom.cdf cross-check             p = {p_less_hyper:.6g}"
              f"   (should match 'less' above)")
    print()

print("=" * 72)
print("DIAGNOSIS")
print("=" * 72)
print("""If 'less' gives a small p-value (<0.05) while 'greater' gives ~1.0,
that is strong evidence the saved biomarker_overlap_fisher_test.csv
(fisher_p=1.0 for all top_n) was generated with the wrong `alternative`
argument -- 'greater' or the scipy default instead of 'less'. The 'less'
p-value above is the statistically correct one for the claim actually
being made in the paper ("significantly disjoint" / lower overlap than
chance). Use that number, not 0.0001 and not 1.0, unless it also comes
out non-significant -- in which case the honest move is to drop the
Fisher's-exact framing entirely and keep only the qualitative zero-overlap
observation, which is what the current manuscript text already does.""")
