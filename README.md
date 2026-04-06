# PulseIQ AI
## Voice-Based Multi-Disease Early Detection System

> A unified acoustic framework for simultaneous screening of neurological, respiratory, and psychological conditions from a single 10-second voice recording.

---

## Overview

PulseIQ AI is a research prototype that demonstrates the feasibility of multi-domain disease screening using a shared acoustic feature space. A single voice recording is analyzed to produce calibrated risk scores for four clinically distinct conditions, each accompanied by SHAP-based feature attribution explaining the model's decision.

This work addresses a critical gap in existing literature: while voice-based disease detection has been studied extensively for individual conditions, no open-source system simultaneously screens for neurological, respiratory, and psychological disorders from a single unstructured voice sample with transparent, explainable predictions.

---

## Screened Conditions

| Condition | Domain | Dataset | Samples |
|---|---|---|---|
| Parkinson's Disease | Neurological | UCI ML Repository | 195 |
| Respiratory Abnormality | Pulmonary | Coswara (IISc) | 5,411 |
| Psychological Stress | Psychological | RAVDESS | 1,060 |
| Depression Indicators | Psychiatric | RAVDESS | 1,060 |

---

## System Architecture
Voice Recording (10 seconds)
|
v
Acoustic Feature Extraction
Praat/Parselmouth  ->  Jitter, Shimmer, HNR
Librosa            ->  Pitch, Spectral Centroid, ZCR
Librosa            ->  MFCCs (13 coefficients)
|
v
Multi-Disease Inference Pipeline
|- Parkinson's Model    (Random Forest, 7 features)
|- Respiratory Model    (Random Forest, 19 features)
|- Stress Model         (Random Forest, 19 features)
|- Depression Model     (Random Forest, 19 features)
|
v
SHAP Explainability Layer
|
v
Risk Report with Calibrated Scores and Feature Attribution
---

## Results

### Model Performance (5-Fold Stratified Cross-Validation)

| Condition | Accuracy | +/- | AUROC | +/- |
|---|---|---|---|---|
| Parkinson's Disease | 88.72% | 3.84% | 0.9494 | 0.0197 |
| Respiratory Abnormality | 68.08% | 1.08% | 0.7483 | 0.0126 |
| Psychological Stress | 83.62% | 2.62% | 0.9184 | 0.0173 |
| Depression Indicators | 76.33% | 2.95% | 0.8555 | 0.0320 |

### Baseline Comparison (5-Fold CV AUROC)

| Model | Parkinson's | Respiratory | Stress | Depression |
|---|---|---|---|---|
| Logistic Regression | 0.8255 | 0.6865 | 0.8594 | 0.7587 |
| Decision Tree | 0.7920 | 0.5887 | 0.7738 | 0.6872 |
| SVM (RBF) | 0.8939 | 0.7534 | 0.9110 | 0.8493 |
| KNN | 0.9668 | 0.7311 | 0.9141 | 0.8554 |
| **Random Forest (Ours)** | **0.9494** | **0.7555** | **0.9184** | **0.8555** |

Random Forest is selected as the primary model architecture because it achieves competitive or superior AUROC across all four conditions while natively supporting SHAP TreeExplainer — enabling clinically interpretable feature attribution without post-hoc approximation.

### Statistical Significance (McNemar's Test)

Random Forest significantly outperforms Logistic Regression on all four conditions (p < 0.05). Performance parity with SVM is observed; however, SVM does not support equivalent explainability.

| Condition | vs Logistic Regression | vs SVM |
|---|---|---|
| Parkinson's Disease | p = 0.0455 * | p = 0.823 |
| Respiratory Abnormality | p < 0.0001 *** | p = 1.000 |
| Psychological Stress | p = 0.0021 ** | p = 0.556 |
| Depression Indicators | p < 0.0001 *** | p = 0.127 |

### Ablation Study

MFCCs are the most critical feature group across all conditions. Removing MFCCs causes the largest accuracy drops: Stress -12.0%, Depression -11.7%, Respiratory -7.5%. Praat-based clinical features (Jitter, Shimmer, HNR) provide consistent additive value across all models. No single feature group alone achieves competitive performance, confirming that multi-feature fusion is necessary.

### Subject-Independent Evaluation (LOSO)

| Evaluation Protocol | Accuracy | AUROC |
|---|---|---|
| 5-Fold Stratified CV | 88.72% | 0.9494 |
| Leave-One-Subject-Out | 71.28% | 0.6050 |

The performance gap between stratified CV and LOSO reflects subject-specific vocal patterns in the UCI dataset (195 recordings from 31 subjects). This highlights the importance of evaluation methodology in clinical voice analysis and motivates future work on larger, more diverse cohorts.

---

## Feature Extraction

19 acoustic features are extracted per recording:

| Feature | Tool | Clinical Relevance |
|---|---|---|
| Jitter | Praat | Cycle-to-cycle pitch frequency variation — Parkinson's marker |
| Shimmer | Praat | Cycle-to-cycle amplitude variation — vocal instability |
| HNR | Praat | Harmonics-to-noise ratio — voice clarity |
| MFCC 1-13 | Librosa | Spectral envelope — respiratory and phonatory patterns |
| Pitch (F0) | Librosa | Fundamental frequency |
| Spectral Centroid | Librosa | Frequency mass center — voice brightness |
| Zero Crossing Rate | Librosa | Speech rate and voicing indicator |

---

## Explainability

Every prediction is accompanied by SHAP (SHapley Additive exPlanations) feature attribution using TreeExplainer. This provides both global feature importance across the training population and per-prediction attribution for individual recordings.

SHAP summary plots for all four models are available in `results/`.

---

## Installation
```bash
git clone https://github.com/YOUR_USERNAME/PulseIQ-AI.git
cd PulseIQ-AI
pip install -r requirements.txt
```

---

## Usage

**Live voice recording:**
```bash
python demo.py
```

**Analyze an existing wav file:**
```bash
python src/predict.py path/to/audio.wav
```

**Example output:**
============================================================
PULSEIQ AI  --  ACOUSTIC RISK ASSESSMENT REPORT
CONDITION                      RISK BAND    SCORE    AUROC
Parkinson Disease              MODERATE     60.5%    0.949
Respiratory Abnormality        LOW          21.8%    0.748
Psychological Stress           GUARDED      41.5%    0.918
Depression Indicators          LOW          33.0%    0.856
FEATURE ATTRIBUTION (SHAP)
[1] Parkinson Disease
pitch                    -0.3444  [########]  (-) decreases risk
shimmer                  +0.0618  [###     ]  (+) increases risk
---

## Project Structure
PulseIQ-AI/
|-- demo.py                          <- Live demo: record and analyze
|-- requirements.txt
|-- src/
|   |-- feature_extraction/
|   |   -- audio_features.py        <- Core 19-feature extractor |   -- predict.py                   <- File-based prediction
|-- notebooks/
|   |-- voice_dataset_builder.ipynb  <- Coswara pipeline
|   |-- voice_model_training.ipynb   <- Respiratory model
|   |-- parkinsons_audio_model.ipynb <- Parkinson's model
|   |-- stress_model.ipynb           <- Stress model
|   |-- depression_model.ipynb       <- Depression model
|   |-- shap_explainability.ipynb    <- SHAP analysis
|   |-- baseline_comparison.ipynb    <- Comparative evaluation
|   |-- ablation_study.ipynb         <- Feature ablation
|   |-- statistical_tests.ipynb      <- McNemar's tests
|   -- cross_dataset_validation.ipynb <- LOSO validation |-- models/ |   |-- parkinsons_model.pkl |   |-- respiratory_model.pkl |   |-- stress_model.pkl |   -- depression_model.pkl
|-- data/
|   |-- parkinsons_tabular/          <- UCI Parkinson's dataset
|   -- voice_features/              <- Extracted feature CSVs -- results/
|-- baseline_comparison.csv
|-- ablation_study.csv
|-- statistical_tests.csv
|-- loso_validation.csv
|-- shap_parkinsons.png
|-- shap_respiratory.png
|-- shap_stress.png
`-- shap_depression.png
---

## Datasets

| Dataset | Source | License |
|---|---|---|
| UCI Parkinson's | Little et al. 2007, Oxford University | CC BY 4.0 |
| Coswara | IISc Bangalore, Sharma et al. 2020 | CC BY 4.0 |
| RAVDESS | Livingstone & Russo 2018, Ryerson University | CC BY-NC-SA 4.0 |

Coswara and RAVDESS raw audio are not included in this repository due to size. Feature CSVs derived from these datasets are provided in `data/voice_features/`.

---

## Limitations

- Parkinson's model trained on pre-extracted tabular features; RPDE and DFA are approximated with dataset means for live inference
- LOSO evaluation reveals subject-independent Parkinson's accuracy of 71.3%, indicating sensitivity to subject-specific vocal characteristics
- Models not validated on clinical populations
- Depression and stress labels are derived from acted emotional speech (RAVDESS), not clinical assessments
- Language and accent generalizability not evaluated

---

## Roadmap

- Deep acoustic feature extraction using wav2vec 2.0
- Mild Cognitive Impairment detection (pending DementiaBank access)
- Clinical validation study
- Multilingual evaluation
- Mobile application

---

## References

1. Little, M. et al. (2007). Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection. BioMedical Engineering OnLine.
2. Sharma, N. et al. (2020). Coswara: A Database of Breathing, Cough, and Voice Sounds for COVID-19 Diagnosis. INTERSPEECH 2020.
3. Livingstone, S. & Russo, F. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). PLoS ONE.
4. Lundberg, S. & Lee, S. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS 2017.
5. Fagherazzi, G. et al. (2021). Voice for Health: The Use of Vocal Biomarkers from Research to Clinical Practice. Digital Biomarkers.

---

## Disclaimer

PulseIQ AI is a research prototype for academic purposes only. It is not a certified medical device and must not be used for clinical diagnosis or treatment decisions. All outputs should be interpreted by a qualified clinician.

---

*PulseIQ AI — Aditya | B.Tech CSE (AI/ML) | 2026*
