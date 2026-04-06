import librosa
import numpy as np
import parselmouth


def extract_features(audio_path):

    # Load audio
    y, sr = librosa.load(audio_path, sr=None)

    # -----------------------------
    # Praat Voice Biomarkers
    # -----------------------------
    sound = parselmouth.Sound(audio_path)

    point_process = parselmouth.praat.call(
        sound, "To PointProcess (periodic, cc)", 75, 500
    )

    jitter = parselmouth.praat.call(
        point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
    )

    shimmer = parselmouth.praat.call(
        [sound, point_process],
        "Get shimmer (local)",
        0,
        0,
        0.0001,
        0.02,
        1.3,
        1.6,
    )

    harmonicity = parselmouth.praat.call(
        sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0
    )

    hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)

    # -----------------------------
    # Librosa Features
    # -----------------------------

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Pitch
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0

    # Spectral centroid
    spectral_centroid = np.mean(
        librosa.feature.spectral_centroid(y=y, sr=sr)
    )

    # Zero crossing rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # -----------------------------
    # Feature Dictionary
    # -----------------------------

    features = {
        "pitch": pitch,
        "spectral_centroid": spectral_centroid,
        "zcr": zcr,
        "jitter": jitter,
        "shimmer": shimmer,
        "hnr": hnr,
    }

    # Add MFCCs
    for i, mf in enumerate(mfcc_mean):
        features[f"mfcc_{i+1}"] = mf

    return features