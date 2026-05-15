# Dataset Download Instructions

Raw data is NOT included in this repo. All datasets are publicly available.

## 1. UCI Parkinson's (Little et al., 2009)
  wget https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data \
       -O data/raw/uci_parkinsons/parkinsons.csv

## 2. Coswara COVID-19 (Sharma et al., 2020)
  git clone https://github.com/iiscleap/Coswara-Data data/raw/coswara_raw

## 3. DAIC-WOZ Depression (requires agreement)
  1. Visit https://dcapswoz.ict.usc.edu/
  2. Sign the EULA
  3. Place audio files in data/raw/daic_woz/{id}_AUDIO.wav

Pre-extracted COVAREP features (data/features/daic_woz_covarep_allframes.csv)
are included — all depression results reproduce without raw audio.

## Verify after download
  python src/feature_extraction/data_loaders.py --verify
