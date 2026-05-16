# PulseIQ AI: Voice Biomarker Screening for Parkinson's Disease, COVID-19, and Depression

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code for the paper *Condition-Specific Acoustic Feature Attribution Across Parallel Voice Screening Tasks for Parkinson's Disease, COVID-19 Respiratory Screening, and Depression*. Developed at Bennett University by Aditya Raj and Dr. Prashant Kumar.

## Overview

PulseIQ AI is a parallel evaluation pipeline for voice-based clinical screening across three independent tasks: Parkinson's disease, COVID-19 respiratory screening, and depression. Each task uses a separate public dataset and an established evaluation protocol. This is a research contribution; it is not a medical device or clinical tool.

The main finding concerns feature attribution. When PD-type nonlinear features, COVID-19-type MFCCs, and depression-specific COVAREP features are extracted from the same DAIC-WOZ recordings, the depression classifier assigns only 2.6% of its SHAP attribution to nonlinear features, which are the dominant predictor for Parkinson's disease. The result is stable across 200 bootstrap resamples (NL% = 2.51 ± 1.15, 95% CI [0.89, 5.25]). It is consistent with independent acoustic biomarker channels but does not by itself establish biomarker independence.

## Results

Primary within-dataset results:

| Task | Dataset | N | Protocol | AUROC [95% CI] | BACC | F1 | Brier | ECE |
|------|---------|---|----------|----------------|------|-----|-------|-----|
| Parkinson's Disease | UCI (Little 2009) | 195 / 32 subjects | LOSO | 0.802 [0.728–0.870] | 0.660 | 0.880 | 0.144 | 0.102 |
| COVID-19 Respiratory | Coswara (Sharma 2020) | 5,238 | 5-fold CV | 0.758 [0.744–0.771] | 0.695 | 0.667 | 0.202 | 0.063 |
| Depression (exploratory) | DAIC-WOZ AVEC 2017 | 142 / dev 35 | Official split | 0.634 [0.300–0.705] | — | — | — | — |

Statistical tests:

| Test | Result |
|------|--------|
| PD: nonlinear-6 vs all-22 features (DeLong) | z = 2.067, p = 0.039 |
| COVID-19: RF vs LR (DeLong) | z = 12.96, p < 0.001 |
| PD random forest variance (10 seeds) | 0.799 ± 0.007 |
| Depression vs AVEC 2017 baseline (0.630) | p = 0.110, not significant |
| SHAP top-5 stability, PD and COVID-19 (20 bootstraps) | Jaccard = 1.000 |

External validation (UCI to Sakar) yields AUROC 0.535, which is at chance level. Within-dataset LOSO results are idealised upper bounds and should not be read as estimates of clinical performance.

## Installation

```bash
git clone https://github.com/AAdii-15/PulseIQ-AI.git
cd PulseIQ-AI
conda env create -f environment.yml
conda activate pulseiq
```

A pip-based setup is also available through `requirements.txt`.

## Datasets

Raw data is not redistributed. Download each dataset from its original source; see `data/README_DATA.md` for details.

| Dataset | Condition | N | License | Source |
|---------|-----------|---|---------|--------|
| UCI Parkinson's (Little et al., 2009) | Parkinson's | 195 recordings | CC BY 4.0 | https://archive.ics.uci.edu/dataset/174/parkinsons |
| Coswara (Sharma et al., 2020) | COVID-19 | 5,238 recordings | CC BY 4.0 | https://github.com/iiscleap/Coswara-Data |
| DAIC-WOZ (Gratch et al., 2014) | Depression | 142 sessions | USC ICT agreement | https://dcapswoz.ict.usc.edu/ |

Pre-extracted COVAREP features for DAIC-WOZ are included under `data/features/` for reproducibility.

## Reproducing the Results

Every reported metric corresponds to a CSV file in `results/metrics/`. Run the scripts in the following order:

```bash
python src/models/train.py                  # PD and COVID-19 models
python src/models/train_depression_svm.py   # depression model
python src/evaluation/fix2b_shared_space.py  # within-recording SHAP analysis
python src/evaluation/statistical_tests.py   # DeLong tests and bootstrap CIs
python src/evaluation/shap_analysis.py       # SHAP and Jaccard stability
python src/evaluation/remaining_fixes.py     # calibration, BACC, F1
python src/evaluation/generate_figures.py    # paper figures
```

The real-time demo records 15 seconds of microphone audio and outputs probability scores and top SHAP drivers for all three conditions. It requires the trained model files (see Repository Structure note):

```bash
python demo.py
```

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
│   │   ├── train_depression.py          # depression pipeline
│   │   ├── train_depression_svm.py      # SVM + SelectKBest
│   │   └── temporal_attention_model.py  # BiLSTM attention model
│   └── evaluation/
│       ├── shap_analysis.py             # TreeSHAP + Jaccard stability
│       ├── statistical_tests.py         # DeLong, bootstrap CIs, permutation
│       ├── ablation_study.py            # feature group ablation
│       ├── baseline_comparison.py       # RF vs LR/SVM/KNN/DT
│       ├── fix2b_shared_space.py        # within-recording SHAP analysis
│       ├── remaining_fixes.py           # calibration, BACC, F1, RF variance
│       ├── phq8_analysis.py             # PHQ-8 BH-FDR analysis
│       └── generate_figures.py          # paper figures
├── results/
│   ├── metrics/                         # result CSVs (one per reported metric)
│   └── figures/                         # paper figures (PNG)
├── data/
│   ├── features/                        # pre-extracted COVAREP features
│   └── README_DATA.md                   # dataset download instructions
├── demo.py                              # real-time voice screening demo
├── environment.yml                      # conda environment (Python 3.10)
├── requirements.txt                     # pip requirements
├── CITATION.cff                         # citation metadata
└── LICENSE                              # MIT
```

Trained model files (`.pkl`, `.pt`) are not tracked in version control because of their size. Regenerate them by running the training scripts above, or contact the authors to obtain them.

## Citation

If you use this code, please cite:

```bibtex
@misc{raj2025pulseiq,
  title  = {Condition-Specific Acoustic Feature Attribution Across Parallel
            Voice Screening Tasks for Parkinson's Disease, COVID-19 Respiratory
            Screening, and Depression},
  author = {Raj, Aditya and Kumar, Prashant},
  year   = {2025},
  url    = {https://github.com/AAdii-15/PulseIQ-AI}
}
```

## Limitations

1. There is no co-labelled multi-condition dataset, so attribution separability does not establish biomarker independence.
2. External validation (UCI to Sakar) performs at chance, indicating the models capture dataset-specific characteristics rather than disease signal.
3. The depression result is not statistically significant (p = 0.110) and is reported as exploratory only.
4. Each condition uses a single recording site, so cross-site robustness is untested.
5. No human clinical baseline is available, so the LOSO AUROC cannot be compared against clinical practice.
6. All speech is in English; cross-lingual generalisation is untested.
7. The pipeline uses classical machine learning; the contribution is in biomedical evaluation, not in novel model architecture.

A full limitations discussion is in Section 6.4 of the paper.

## License

Released under the MIT License (see `LICENSE`). Datasets remain subject to their own licenses.

---

Bennett University, Greater Noida, India, 2025
