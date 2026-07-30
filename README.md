# Same Voice, Different Signals

**Subject-Independent Evaluation and Within-Recording Attribution Analysis for Voice-Based Health Screening**

Aditya Raj¹\* and Prashant Kumar²

¹ Department of Computer Science (AI/ML), Bennett University, Greater Noida, India
² Independent Researcher

\*Corresponding author: adityarajdhuria@gmail.com

This repository accompanies the manuscript of the same title (currently under peer review). It is not a medical device and is not intended for clinical use.

---

## Key Findings

- **Evaluation-protocol leakage (PD):** fold-based cross-validation inflates AUROC by 0.13 relative to subject-independent Leave-One-Subject-Out (LOSO) evaluation on identical data (0.932 fold-CV vs. 0.802 LOSO) — a gap consistent with subject-level leakage documented elsewhere in the PD voice literature.
- **Parkinson's Disease Detection:** AUROC 0.802 [0.728–0.870], *p*<0.001, under subject-independent LOSO (UCI, *N*=195).
- **COVID-19 Respiratory Screening:** AUROC 0.758 [0.744–0.771], *p*<0.001, under 5-fold CV (Coswara, *N*=5,238).
- **Depression (case study, DAIC-WOZ, *N*=142):** AUROC 0.621 [0.416–0.816], not significant against chance (*p*=0.126, permutation test), reported as the mean across 10 random seeds because this pipeline's feature selection is measurably seed-sensitive at *N*<sub>train</sub>=107.
- **Within-recording attribution:** PD-type nonlinear features receive only 2.6% of SHAP attribution for the depression classifier when extracted from the same audio as COVAREP and MFCC features — consistently the lowest-attributed group across the full feature space, a cardinality-matched analysis, and a COVAREP-removed counterfactual.
- **Cross-condition transfer:** a PD classifier applied to the same DAIC-WOZ audio shows no discriminative transfer to depression (AUROC 0.434, *p*>0.05).
- Random Forest is benchmarked against XGBoost and LightGBM: statistically indistinguishable for PD, significantly outperforms both for COVID-19 (DeLong *p*<0.001).
- A formal Hanley–McNeil power analysis establishes *N*≥196 as the minimum sample size to confirm the observed depression effect at 80% power.

---

## Overview

### Why this work matters

Most voice-biomarker studies evaluate a single health condition in isolation, and evaluation practices vary widely across that literature: many use subject-dependent cross-validation, few report calibration, and almost none test whether classifiers for different conditions rely on the same acoustic features. This repository addresses both gaps.

**First**, we provide a rigorously evaluated, calibrated screening pipeline for Parkinson's disease (UCI Parkinson's) and COVID-19 respiratory screening (Coswara), and we directly quantify how much a common evaluation shortcut — fold-based cross-validation without subject-level holdout — inflates reported performance on the same PD data.

**Second**, we introduce a within-recording SHAP attribution framework: PD-type nonlinear, MFCC, and COVAREP features are extracted from the *same* audio recordings, and TreeSHAP is used to measure which feature family a classifier actually relies on. This is demonstrated as a case study on DAIC-WOZ depression audio (AVEC 2017 split).

This is a **parallel evaluation** under one shared methodology across three independent datasets and subject populations — not simultaneous multi-condition screening from a single individual. Conclusions concern model-level evaluation methodology and attribution separability; they do not establish biological biomarker independence, which requires co-labelled, multi-condition, prospective data.

---

## Results

### Primary screening performance

| Condition | Protocol | *N* | AUROC [95% CI] | BACC | *F*₁ | Sens. | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| Parkinson's Disease (6 nonlinear features) | LOSO | 195 | 0.802 [0.728–0.870]\*\*\* | 0.671 | 0.882 | 0.946 | 0.144 | 0.108 |
| COVID-19 Respiratory | 5-fold CV | 5,238 | 0.758 [0.744–0.771]\*\*\* | 0.697 | 0.667 | 0.631 | 0.202 | 0.066 |
| Depression (case study)† | AVEC 2017 dev | 142 | 0.621 [0.416–0.816] ns | 0.692‡ | 0.620‡ | 0.850‡ | 0.224 | 0.094 |

\*\*\* *p*<0.001; ns not significant (*p*=0.126, permutation test). † Case-study dataset; no clinical claim; all depression values are the mean across 10 random seeds (see Limitations). ‡ At each seed's own ROC-optimal (Youden *J*) threshold, then averaged.

### Random Forest vs. modern tree ensembles

RF is not simply assumed superior — it is benchmarked directly against XGBoost and LightGBM under each condition's primary protocol:

| Condition | Model | AUROC [95% CI] | *p* vs. RF |
|---|---|---|---|
| PD (LOSO) | RF | 0.802 [0.728–0.870] | — |
| | XGBoost | 0.815 [0.748–0.878] | 0.780 |
| | LightGBM | 0.793 [0.718–0.860] | 0.836 |
| COVID-19 (5-fold CV) | RF | 0.760 [0.746–0.773] | — |
| | XGBoost | 0.733 [0.719–0.747] | <0.001 |
| | LightGBM | 0.731 [0.716–0.744] | <0.001 |

RF matches both alternatives for PD and significantly outperforms both for COVID-19, in addition to enabling exact TreeSHAP attribution.

### Within-recording attribution separability

Three feature families extracted from the *same* DAIC-WOZ audio, evaluated by a Random Forest trained on the shared 178-dimensional feature space:

| Feature type | Full space | Counterfactual (no COVAREP) | Matched (5+5+5) |
|---|---|---|---|
| COVAREP (phonatory) | 74.2% | — | 32.5% ± 4.8% |
| MFCC (spectral) | 23.2% | 77.0% | 41.0% ± 4.0% |
| Nonlinear (PD-type) | 2.6% | 12.4% | 26.5% ± 2.5% |

Nonlinear (PD-type) features consistently receive the lowest attribution across every configuration tested. Group ablation corroborates this: removing nonlinear features changes dev AUROC by only −0.018, removing MFCC by −0.007, while removing COVAREP *increases* AUROC by +0.182 — revealing that COVAREP's high SHAP share partly reflects overfitting at *N*<sub>train</sub>=107 rather than generalisable importance.

### Additional findings

- **Cross-condition transfer (null result):** the trained PD classifier applied to DAIC-WOZ nonlinear features shows no discriminative transfer to depression (AUROC 0.434 [0.344–0.527], Mann–Whitney *U*=1824.5, *p*=0.139).
- **SHAP stability:** top-5 feature rankings are perfectly stable (Jaccard=1.000) across 20 bootstrap replicates for PD and COVID-19; the depression model operates at the noise floor (Jaccard=0.063), so depression attribution is reported only at the feature-group level, never as individual feature rankings.
- **Statistical power:** a Hanley–McNeil analysis shows the minimum detectable AUROC at *N*=35 (80% power) is 0.757; confirming the observed depression effect (0.621) requires *N*≥196.
- **Cross-dataset generalisation:** a model trained on shared traditional features performs near chance transferring from UCI Parkinson's to Sakar 2013, consistent with dataset-specific rather than generalisable signal.
- **Cross-model SHAP overlap:** separately trained PD and COVID-19 models share zero features in their top-15 SHAP rankings (Fisher's exact, one-sided, *p*=1.2×10⁻⁴ at top-15; not significant at top-5/10, reflecting the smaller sample size at those cutoffs).
- **Null results, reported transparently:** zero significant PHQ-8 symptom correlations across 1,168 tests under Bonferroni and Benjamini–Hochberg correction.

All within-dataset results are same-distribution upper bounds, not generalisation estimates.

---

## Installation

```bash
git clone https://github.com/AAdii-15/PulseIQ-AI.git
cd PulseIQ-AI
conda env create -f environment.yml
conda activate pulseiq
```

A pip-based setup is also available through `requirements.txt`.

## Datasets

Raw data is not redistributed. Download each dataset from its original source; see `data/README_DATA.md` for details. Pre-extracted COVAREP features for DAIC-WOZ are included under `data/features/` for reproducibility. **Raw DAIC-WOZ audio is deliberately excluded from this repository**, consistent with its USC ICT data use agreement.

| Dataset | Condition | *N* | License | Source |
|---|---|---|---|---|
| UCI Parkinson's (Little et al., 2009) | Parkinson's | 195 recordings, 32 subjects | CC BY 4.0 | https://archive.ics.uci.edu/dataset/174/parkinsons |
| Coswara (Sharma et al., 2020) | COVID-19 | 5,238 recordings | CC BY 4.0 | https://github.com/iiscleap/Coswara-Data |
| DAIC-WOZ (Gratch et al., 2014) | Depression | 142 sessions | USC ICT agreement | https://dcapswoz.ict.usc.edu/ |

## Reproducing the Results

| Step | Script | Produces |
|---|---|---|
| 1 | `python src/models/train.py` | PD and COVID-19 Random Forest models |
| 2 | `python src/models/train_depression_svm.py` | Depression SVM model (primary screening result) |
| 3 | `python src/evaluation/tree_ensemble_comparison.py` | RF vs. XGBoost vs. LightGBM comparison |
| 4 | `python src/evaluation/ablation_study.py` | Feature group ablation |
| 5 | `python src/evaluation/fix2b_shared_space.py` | Within-recording SHAP attribution (full space) |
| 6 | `python src/evaluation/shap_analysis.py` | Cross-model SHAP rankings and Jaccard stability |
| 7 | `python src/evaluation/depression_seed_robustness.py` | Depression seed-robustness check (10 seeds) |
| 8 | `python src/evaluation/depression_final_v2.py` | Final depression AUROC/BACC/F1/Sens with combined seed+bootstrap CI |
| 9 | `python src/evaluation/depression_permutation_test.py` | Depression permutation test (*p*=0.126) |
| 10 | `python src/evaluation/verify_attribution_rf_auroc.py` | Attribution RF AUROC verification |
| 11 | `python src/evaluation/fix_biomarker_overlap_fisher_test.py` | Corrected cross-model SHAP overlap test |
| 12 | `python src/evaluation/statistical_tests.py` | DeLong tests, bootstrap CIs |
| 13 | `python src/evaluation/phq8_analysis.py` | PHQ-8 symptom null analysis |
| 14 | `python src/evaluation/gen_fig_calibration.py` | Calibration reliability diagrams |
| 15 | `python src/evaluation/gen_fig1_iconstyle.py`, `gen_fig_fix2b_combined_BSPC.py`, `gen_fig_stability_power_BSPC.py` | Main text figures |

Every reported number traces to a CSV in `results/metrics/` or the console output of one of the scripts above.

## Demo

`demo.py` records microphone audio and outputs probability scores and top SHAP drivers. It requires trained model files, which are not tracked in version control due to size — regenerate them via steps 1–2 above.

```bash
python demo.py
```

## Repository Structure

```
PulseIQ-AI/
├── src/
│ ├── feature_extraction/ # dataset loaders, COVAREP/MFCC/nonlinear extraction
│ ├── models/ # PD/COVID-19 RF, depression SVM
│ └── evaluation/ # SHAP analysis, statistical tests, seed robustness,
│ # tree-ensemble comparison, calibration, figures
├── results/
│ ├── metrics/ # one CSV per reported result
│ └── figures/ # paper figures (600dpi PNGs)
├── data/
│ ├── features/ # pre-extracted COVAREP features (DAIC-WOZ)
│ └── README_DATA.md # dataset download instructions
├── demo.py
├── environment.yml
├── requirements.txt
└── LICENSE
```

Trained model files (`.pkl`, `.pt`) and raw audio are not tracked in version control (size and data-use-agreement reasons respectively). Regenerate models by running the training scripts above, or contact the authors.

## Limitations

The manuscript reports a full limitations discussion in its Discussion and Limitations sections. The most important points:

- There is no co-labelled multi-condition dataset, so within-recording attribution separability does not establish biological biomarker independence.
- The depression analysis is a case study, not a validated screening result: the AVEC 2017 development set (*N*=35) is fundamentally underpowered, and this pipeline's feature selection is measurably seed-sensitive (AUROC range 0.605–0.638 across 10 seeds at *N*<sub>train</sub>=107) — all reported depression values are seed-averaged for this reason.
- Cross-dataset generalisation was not achieved; within-dataset results are same-distribution upper bounds only.
- Coswara lacks subject identifiers, so 5-fold CV cannot guarantee subject-independent splits (sensitivity analysis: max Δ≤0.022 AUROC under 5–20% random removal).
- All speech is in English; cross-lingual generalisation is untested.

## Citation

If you use this code, please cite the manuscript associated with this repository (citation details to be updated upon publication).

## Acknowledgements

The authors thank the creators of the UCI Parkinson's, Coswara, and DAIC-WOZ datasets for making their data publicly available.

## License

Released under the MIT License (see `LICENSE`). The datasets remain subject to their own licenses.
