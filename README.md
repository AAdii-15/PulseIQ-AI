# PulseIQ AI — Voice Biomarker Screening for Parkinson's, COVID-19, and Depression

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PD AUROC](https://img.shields.io/badge/PD%20AUROC-0.802%20LOSO-red.svg)]()
[![COVID AUROC](https://img.shields.io/badge/COVID--19%20AUROC-0.758-orange.svg)]()

**Bennett University · Aditya Raj · Dr. Prashant Kumar**

---

## What This Is

A rigorous parallel evaluation pipeline for voice-based clinical screening across three independent tasks.
This is a **research contribution** — not a medical device, not a clinical tool.

| Task | Dataset | N | Protocol | AUROC | Status |
|------|---------|---|----------|-------|--------|
| Parkinson's Disease | UCI (Little 2009) | 195 (32 subjects) | **LOSO** | **0.802** [0.728–0.870] | p < 0.001 |
| COVID-19 Respiratory | Coswara (Sharma 2020) | 5,238 | 5-Fold CV | **0.758** [0.744–0.771] | p < 0.001 |
| Depression | DAIC-WOZ AVEC 2017 | 142 (dev=35) | Official Split | 0.634 [0.300–0.705] | p=0.110, exploratory |

> **Critical caveat:** Cross-dataset PD validation (UCI→Sakar) = AUROC 0.535 (chance level).
> All within-dataset LOSO results are idealised upper bounds, not clinical performance estimates.

---

## Primary Scientific Finding

When PD-type nonlinear features, COVID-19-type MFCCs, and depression-specific COVAREP features
are simultaneously extracted from the **same DAIC-WOZ audio recordings**, the depression classifier
assigns only **2.6% SHAP attribution** to nonlinear features — the dominant predictor for PD.

Bootstrap-validated over 200 independent resamples: **NL% = 2.51 ± 1.15, 95% CI [0.89, 5.25]**

*Interpretation: the depression classifier does not rely on PD-discriminative features —
consistent with, but not proof of, independent acoustic biomarker channels.*

---

## Key Results

**Parkinson's Disease** (LOSO, 6 nonlinear features)
- AUROC: 0.802 · BACC: 0.660 · F1: 0.880 · Brier: 0.144 · ECE: 0.102
- DeLong (nonlinear-6 vs all-22): z = 2.067, *p* = 0.039 \*
- RF variance (10 seeds): 0.799 ± 0.007

**COVID-19 Respiratory** (5-fold CV)
- AUROC: 0.758 · BACC: 0.695 · F1: 0.667 · Brier: 0.202 · ECE: 0.063
- DeLong (RF vs LR): z = 12.96, *p* < 0.001 \*\*\*
- Duplication inflation ceiling: Δ ≤ 0.009 AUROC

**Depression** — *Exploratory Only*
- AUROC: 0.634, *p* = 0.110 (non-significant)
- AVEC 2017 official baseline: 0.630
- Bootstrap NL%: 2.51 ± 1.15, 95% CI [0.89, 5.25] — stable

**SHAP Stability**
- PD top-5 Jaccard = 1.000 (20 bootstraps)
- COVID-19 top-5 Jaccard = 1.000 (20 bootstraps)

**External Validation**
- UCI → Sakar: AUROC = 0.535 ≈ chance
---

## Repository Structure

```
PulseIQ-AI/
├── src/
│   ├── feature_extraction/
│   │   ├── data_loaders.py              # UCI, Coswara, DAIC-WOZ loaders
│   │   ├── covarep_features.py          # COVAREP feature extraction
│   │   ├── wav2vec_features.py          # wav2vec 2.0 embeddings
│   │   ├── wav2vec_temporal.py          # BiLSTM temporal attention
│   │   └── daic_respiratory_features.py # MFCC + nonlinear from DAIC-WOZ
│   ├── models/
│   │   ├── train.py                     # PD + COVID-19 RF (LOSO / 5-fold)
│   │   ├── train_depression.py          # Depression pipeline
│   │   ├── train_depression_svm.py      # SVM + SelectKBest
│   │   └── temporal_attention_model.py  # BiLSTM attention model
│   └── evaluation/
│       ├── shap_analysis.py             # TreeSHAP + Jaccard stability
│       ├── statistical_tests.py         # DeLong, bootstrap CIs, permutation
│       ├── ablation_study.py            # Feature group ablation
│       ├── baseline_comparison.py       # RF vs LR/SVM/KNN/DT
│       ├── fix2b_shared_space.py        # Within-recording SHAP analysis
│       ├── remaining_fixes.py           # Calibration, BACC, F1, RF variance
│       ├── phq8_analysis.py             # PHQ-8 BH-FDR null finding
│       └── generate_figures.py          # All paper figures
├── models/                              # 15 trained models (.pkl, .pt)
│   ├── parkinsons_nonlinear_model.pkl   # PRIMARY PD model (6 features)
│   ├── parkinsons_model.pkl             # All-22 features (comparison)
│   ├── respiratory_model.pkl            # COVID-19 RF
│   ├── depression_best_model.pkl        # COVAREP + SVM (AUROC 0.634)
│   ├── temporal_attention_model.pt      # BiLSTM attention
│   └── phq8_*_regressor.pkl             # PHQ-8 symptom regressors (8 items)
├── results/
│   ├── metrics/                         # 31 result CSVs (fully reproducible)
│   └── figures/                         # All paper figures (PNG)
├── data/
│   └── README_DATA.md                   # Dataset download instructions
├── demo.py                              # Real-time 15-second voice screening
├── environment.yml                      # Conda environment (Python 3.10)
├── requirements.txt                     # pip requirements
├── CITATION.cff                         # Citation metadata
├── LICENSE                              # MIT
└── README.md                            # This file
```

## Reproducibility

Every number in the paper traces to a CSV in `results/metrics/`.

| Component | Reproducible | File |
|-----------|-------------|------|
| PD LOSO AUROC 0.802 | ✅ | `results/metrics/final_primary_results.csv` |
| DeLong test results | ✅ | `results/metrics/delong_tests.csv` |
| SHAP Jaccard stability | ✅ | `results/metrics/shap_stability_jaccard.csv` |
| Bootstrap NL% (200 runs) | ✅ | `results/metrics/shap_bootstrap_stability_depression.csv` |
| Calibration (Brier, ECE) | ✅ | `results/metrics/calibration_bacc_f1.csv` |
| Coswara sensitivity | ✅ | `results/metrics/coswara_sensitivity_analysis.csv` |
| RF variance (10 seeds) | ✅ | `results/metrics/rf_variance.csv` |
| PHQ-8 BH-FDR null | ✅ | `results/metrics/phq8_multiple_comparison.csv` |
| External validation | ✅ | `results/metrics/external_validation_sakar.csv` |

---

## Quick Start

### Setup

```bash
git clone https://github.com/AAdii-15/PulseIQ-AI.git
cd PulseIQ-AI
conda env create -f environment.yml
conda activate pulseiq
```

### Download Datasets

```bash
# See data/README_DATA.md for full instructions
# UCI Parkinson's  → https://archive.ics.uci.edu/dataset/174/parkinsons
# Coswara          → https://github.com/iiscleap/Coswara-Data
# DAIC-WOZ         → https://dcapswoz.ict.usc.edu/ (requires agreement)
```

### Reproduce All Results

```bash
python src/models/train.py                    # Train PD + COVID-19 models
python src/models/train_depression_svm.py     # Train depression model
python src/evaluation/fix2b_shared_space.py   # Within-recording SHAP analysis
python src/evaluation/statistical_tests.py    # DeLong + bootstrap
python src/evaluation/shap_analysis.py        # SHAP + Jaccard stability
python src/evaluation/remaining_fixes.py      # Calibration, BACC, F1
python src/evaluation/generate_figures.py     # Regenerate all figures
```

### Real-Time Demo

```bash
python demo.py
# Records 15 seconds via microphone
# Outputs probability scores for all three conditions
# Shows top SHAP feature drivers
```

---

## Datasets

| Dataset | Condition | N | License | Link |
|---------|-----------|---|---------|------|
| UCI Parkinson's (Little et al., 2009) | PD | 195 recordings | CC BY 4.0 | [UCI ML](https://archive.ics.uci.edu/dataset/174/parkinsons) |
| Coswara (Sharma et al., 2020) | COVID-19 | 5,238 recordings | CC BY 4.0 | [GitHub](https://github.com/iiscleap/Coswara-Data) |
| DAIC-WOZ (Gratch et al., 2014) | Depression | 142 sessions | USC ICT agreement | [USC ICT](https://dcapswoz.ict.usc.edu/) |

Raw data is not included. Pre-extracted COVAREP features for DAIC-WOZ
(`data/features/daic_woz_covarep_allframes.csv`) are included for reproducibility.

---

## Citation

```bibtex
@article{raj2025pulseiq,
  title   = {Condition-Specific Acoustic Feature Attribution Across
             Parallel Voice Screening Tasks for {Parkinson's Disease},
             {COVID-19} Respiratory Screening, and Depression},
  author  = {Raj, Aditya and Kumar, Prashant},
  year    = {2025},
  url     = {https://github.com/AAdii-15/PulseIQ-AI}
}
```

---

## Known Limitations

1. **No co-labelled multi-condition data** — attribution separability ≠ biomarker independence
2. **External validation fails** (AUROC 0.535 = chance) — models learn dataset, not disease
3. **Depression non-significant** (p=0.110) — exploratory only, not clinical evidence
4. **No cross-site robustness** — single recording site per condition
5. **No human clinical baseline** — AUROC 0.802 cannot be contextualised against clinical practice
6. **English speech only** — no cross-lingual validation
7. **Classical ML stack** — biomedical AI contribution, not novel ML architecture

Full 13-point limitations list in paper Section 6.4.

---

## License

MIT License — see [LICENSE](LICENSE).
Datasets are governed by their own licences (see above).

---

*Bennett University, Greater Noida, India · 2025*
