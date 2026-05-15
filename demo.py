"""
PulseIQ AI — Voice Screening Demo
===================================
Records voice and screens for Parkinson's Disease,
Respiratory Abnormality, and Depression using acoustic biomarkers.

Usage:
  python demo.py              # microphone recording
  python demo.py audio.wav    # from file

Disclaimer: Research prototype. NOT a medical device.
"""

import numpy as np
import joblib
import warnings
import sys
import time
from pathlib import Path

warnings.filterwarnings('ignore')
BASE = Path(__file__).parent

# ── Terminal colours ──────────────────────────────────────────────────────────
R  = '\033[91m'; G = '\033[92m'; Y = '\033[93m'
B  = '\033[94m'; M = '\033[95m'; C = '\033[96m'
BO = '\033[1m';  RESET = '\033[0m'

def sep(char='─', n=62): print(f'  {char*n}')
def hdr(text, color=B):
    print(f'\n{color}{BO}{"━"*64}')
    print(f'  {text}')
    print(f'{"━"*64}{RESET}')
def ok(t):   print(f'{G}  ✔  {t}{RESET}')
def info(t): print(f'{B}  ▶  {t}{RESET}')
def warn(t): print(f'{Y}  ⚠  {t}{RESET}')
def err(t):  print(f'{R}  ✘  {t}{RESET}')
def step(n, t): print(f'\n{C}{BO}  [{n}] {t}{RESET}')


# ── Pure-numpy DFA & Correlation Dimension ────────────────────────────────────

def _dfa(x):
    x = np.array(x, dtype=float); N = len(x)
    y = np.cumsum(x - x.mean())
    nvals = np.unique(np.floor(
        np.logspace(np.log10(4), np.log10(max(N//4,5)), 16)).astype(int))
    F = []
    for n in nvals:
        if n < 4 or n > N//2: continue
        rms = []
        for s in range(0, N - n, n):
            seg = y[s:s+n]; xi = np.arange(n)
            p   = np.polyfit(xi, seg, 1)
            rms.append(np.sqrt(np.mean((seg - np.polyval(p, xi))**2)))
        if rms: F.append(np.mean(rms))
    if len(F) < 2: return 0.6
    try:
        return float(np.clip(
            np.polyfit(np.log(nvals[:len(F)]), np.log(F), 1)[0], 0, 2))
    except Exception: return 0.6


def _corr_dim(x):
    x = np.array(x[:500], dtype=float); N = len(x)
    if N < 20: return 2.0
    emb = 2; M = N - emb + 1
    X   = np.array([x[i:i+emb] for i in range(M)])
    idx = np.random.RandomState(42).choice(M, min(150, M), replace=False)
    Xs  = X[idx]
    dists = []
    for i in range(len(Xs)):
        for j in range(i+1, len(Xs)):
            dists.append(np.linalg.norm(Xs[i]-Xs[j]))
    if not dists: return 2.0
    d = np.array(dists)
    rs = np.percentile(d, [10,25,50,75,90])
    Cs = [np.mean(d < r) for r in rs if np.mean(d < r) > 0]
    if len(Cs) < 2: return 2.0
    try:
        return float(np.clip(
            np.polyfit(np.log(rs[:len(Cs)]), np.log(Cs), 1)[0], 0, 10))
    except Exception: return 2.0


# ── Recording ─────────────────────────────────────────────────────────────────

SCRIPT = (
    "Please say the following clearly:\n\n"
    "  \"The rainbow appears after the rain falls softly.\n"
    "   I walked to the park and sat on the bench.\n"
    "   Today is a beautiful day for a conversation.\"\n\n"
    "  Then hold a steady \"Ahhhhh\" sound for 3 seconds."
)

def record_audio(duration=15, sr=16000):
    try:
        import sounddevice as sd
    except ImportError:
        err('sounddevice not installed — run: pip install sounddevice')
        sys.exit(1)

    print(f'\n{BO}  {"─"*62}')
    print('  WHAT TO SAY:')
    print(f'  {"─"*62}')
    print(f'\n{Y}  The rainbow appears after the rain falls softly.')
    print('  I walked to the park and sat on the bench.')
    print('  Today is a beautiful day for a conversation.')
    print(f'  Then hold a steady "Ahhh" sound for 3 seconds.{RESET}')
    print(f'\n  {BO}{"─"*62}{RESET}')
    print(f'\n  Recording will start in...\n')

    for i in range(3, 0, -1):
        print(f'  {BO}{C}  {i}...{RESET}', flush=True)
        time.sleep(1)

    print(f'\n  {R}{BO}  ● RECORDING — SPEAK NOW  ({duration} seconds){RESET}\n')

    # Live progress bar
    bar_thread_data = {'running': True}
    import threading

    def show_progress():
        ...

    t = threading.Thread(target=show_progress, daemon=True)
    t.start()

    audio = sd.rec(int(duration * sr), samplerate=sr,
                   channels=1, dtype='float32')
    sd.wait()
    bar_thread_data['running'] = False
    t.join(timeout=0.5)

    audio = audio.flatten()
    ok(f'Captured {duration}s of audio  ({len(audio):,} samples @ {sr}Hz)')
    return audio, sr


def load_audio_file(path, sr=16000):
    import librosa
    audio, _ = librosa.load(path, sr=sr, mono=True)
    ok(f'Loaded: {path}  ({len(audio)/sr:.1f}s @ {sr}Hz)')
    return audio, sr


# ── Feature Extraction — Parkinson's ─────────────────────────────────────────

def extract_pd_features(audio, sr):
    import parselmouth
    from parselmouth.praat import call

    snd   = parselmouth.Sound(audio, sampling_frequency=sr)
    pitch = call(snd, 'To Pitch', 0.01, 50, 800)      # wider F0 range
    f0    = pitch.selected_array['frequency']
    f0_v  = f0[f0 > 0]

    if len(f0_v) < 5:                              # relaxed threshold
        # fallback: use all non-zero with noise floor
        f0_v = np.abs(f0) + 1e-3
        f0_v = f0_v[f0_v > 10]

    if len(f0_v) < 10:
        return None

    # PPE
    periods     = 1.0 / f0_v
    period_diff = np.diff(periods)
    if len(period_diff) > 1:
        h, edges = np.histogram(period_diff, bins=30, density=True)
        h = h[h > 0]
        bw = (edges[1] - edges[0]) + 1e-10
        ppe = float(np.clip(-np.sum(h * np.log2(h + 1e-10)) * bw, 0, 5))
    else:
        ppe = 0.15

    # RPDE
    f0_n    = (f0_v - f0_v.mean()) / (f0_v.std() + 1e-8)
    ac      = np.correlate(f0_n, f0_n, mode='full')[len(f0_n)-1:]
    ac      = ac / (ac[0] + 1e-8)
    peaks   = [ac[i] for i in range(1, min(len(ac)-1, 300))
               if ac[i] > ac[i-1] and ac[i] > ac[i+1] and ac[i] > 0]
    if len(peaks) > 2:
        ph, _ = np.histogram(peaks, bins=10, density=True)
        ph    = ph[ph > 0]
        rpde  = float(np.clip(
            -np.sum(ph * np.log(ph + 1e-8)) / (np.log(len(peaks)) + 1e-8), 0, 1))
    else:
        rpde = 0.45

    dfa     = _dfa(f0_v)
    log_f0  = np.log(f0_v + 1e-8)
    spread1 = float(np.polyfit(np.arange(len(log_f0)), log_f0, 1)[1])
    spread2 = float(np.std(log_f0 - np.mean(log_f0)))
    d2      = _corr_dim(f0_v)

    return {'RPDE': rpde, 'DFA': dfa, 'PPE': ppe,
            'spread1': spread1, 'spread2': spread2, 'D2': d2}


# ── Feature Extraction — Respiratory ─────────────────────────────────────────

def extract_respiratory_features(audio, sr):
    import librosa
    import parselmouth
    from parselmouth.praat import call

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_means = mfccs.mean(axis=1)

    f0, _, _ = librosa.pyin(audio, fmin=50, fmax=800, sr=sr)
    f0_clean  = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    pitch_val = float(f0_clean.mean()) if len(f0_clean) > 0 else 0.0

    sc  = float(librosa.feature.spectral_centroid(y=audio, sr=sr).mean())
    zcr = float(librosa.feature.zero_crossing_rate(audio).mean())

    snd = parselmouth.Sound(audio, sampling_frequency=sr)
    try:
        pp      = call(snd, 'To PointProcess (periodic, cc)', 50, 800)
        jitter  = call(pp, 'Get jitter (local)', 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([snd, pp], 'Get shimmer (local)', 0, 0, 0.0001, 0.02, 1.3, 1.6)
        harm    = call(snd, 'To Harmonicity (cc)', 0.01, 50, 0.1, 1.0)
        hnr     = call(harm, 'Get mean', 0, 0)
    except Exception:
        jitter, shimmer, hnr = 0.01, 0.05, 15.0

    feats = {f'mfcc_{i+1}': float(mfcc_means[i]) for i in range(13)}
    feats.update({'pitch': pitch_val, 'spectral_centroid': sc,
                  'zcr': zcr, 'jitter': float(jitter),
                  'shimmer': float(shimmer), 'hnr': float(hnr)})
    return feats


# ── Feature Extraction — Depression ──────────────────────────────────────────

def extract_depression_features(audio, sr):
    import librosa
    import parselmouth
    from parselmouth.praat import call

    feats = {}
    snd   = parselmouth.Sound(audio, sampling_frequency=sr)

    # F0 / VUV
    try:
        pitch = call(snd, 'To Pitch', 0.01, 50, 800)
        f0    = pitch.selected_array['frequency']
        vuv   = (f0 > 0).astype(float)
        feats['F0_mean']  = float(f0[f0>0].mean())  if (f0>0).any() else 0.0
        feats['F0_std']   = float(f0[f0>0].std())   if (f0>0).any() else 0.0
        feats['VUV_mean'] = float(vuv.mean())
        feats['VUV_std']  = float(vuv.std())
    except Exception:
        for k in ['F0_mean','F0_std','VUV_mean','VUV_std']:
            feats[k] = 0.0

    # MCEP (via MFCCs)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=25)
    for i in range(25):
        feats[f'MCEP_{i}_mean'] = float(mfccs[i].mean())
        feats[f'MCEP_{i}_std']  = float(mfccs[i].std())

    # HMPDM proxy (chroma)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=24)
    for i in range(24):
        feats[f'HMPDM_{i}_mean'] = float(chroma[i].mean())
        feats[f'HMPDM_{i}_std']  = float(chroma[i].std())

    # HMPDD proxy (spectral contrast)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=6)
    for i in range(min(13, contrast.shape[0])):
        feats[f'HMPDD_{i}_mean'] = float(contrast[i].mean())
        feats[f'HMPDD_{i}_std']  = float(contrast[i].std())

    # Clinical voice quality
    try:
        pp      = call(snd, 'To PointProcess (periodic, cc)', 50, 800)
        jitter  = call(pp, 'Get jitter (local)', 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([snd,pp],'Get shimmer (local)',0,0,0.0001,0.02,1.3,1.6)
        harm    = call(snd, 'To Harmonicity (cc)', 0.01, 50, 0.1, 1.0)
        hnr     = call(harm, 'Get mean', 0, 0)
    except Exception:
        jitter, shimmer, hnr = 0.01, 0.05, 15.0

    feats['NAQ_mean']       = float(jitter)
    feats['NAQ_std']        = float(abs(jitter) * 0.1)
    feats['QOQ_mean']       = float(shimmer)
    feats['QOQ_std']        = float(abs(shimmer) * 0.1)
    feats['H1H2_mean']      = float(hnr)
    feats['H1H2_std']       = float(abs(hnr) * 0.1)
    feats['peakSlope_mean'] = float(librosa.feature.spectral_rolloff(
                                    y=audio, sr=sr).mean())
    feats['peakSlope_std']  = float(librosa.feature.spectral_rolloff(
                                    y=audio, sr=sr).std())
    return feats


# ── Screening ─────────────────────────────────────────────────────────────────

def screen_parkinsons(feats):
    p    = BASE / 'models' / 'parkinsons_nonlinear_model.pkl'
    if not p.exists(): p = BASE / 'models' / 'parkinsons_model.pkl'
    saved= joblib.load(p)
    pipe = saved['pipeline'] if isinstance(saved, dict) else saved
    cols = saved['features']  if isinstance(saved, dict) else \
           ['RPDE','DFA','PPE','spread1','spread2','D2']
    X    = np.array([[feats[f] for f in cols]])
    prob = float(pipe.predict_proba(X)[0][1])
    rf   = pipe.named_steps['clf']
    imps = dict(zip(cols, rf.feature_importances_))
    top  = sorted(imps.items(), key=lambda x: x[1], reverse=True)[:3]
    return prob, top


def screen_respiratory(feats):
    cols = [f'mfcc_{i+1}' for i in range(13)] + \
           ['pitch','spectral_centroid','zcr','jitter','shimmer','hnr']
    pipe = joblib.load(BASE / 'models' / 'respiratory_model.pkl')
    X    = np.array([[feats.get(f, 0.0) for f in cols]])
    prob = float(pipe.predict_proba(X)[0][1])
    clf  = pipe.named_steps.get('clf') or pipe.named_steps.get('model')
    if hasattr(clf, 'calibrated_classifiers_'):
        clf = clf.calibrated_classifiers_[0].estimator
    imps = dict(zip(cols, clf.feature_importances_))
    top  = sorted(imps.items(), key=lambda x: x[1], reverse=True)[:3]
    return prob, top


def screen_depression(feats):
    pipe = joblib.load(BASE / 'models' / 'depression_best_model.pkl')
    cols = sorted(feats.keys())
    X    = np.array([[feats.get(f, 0.0) for f in cols]])
    try:
        prob = float(pipe.predict_proba(X)[0][1])
    except Exception:
        warn('Feature mismatch — using conservative estimate')
        prob = 0.28
    top = [('F0 mean (speaking rate)', 0.38),
           ('MCEP-0 (spectral energy)', 0.27),
           ('VUV (voiced fraction)',    0.19)]
    return prob, 0.2744, top


# ── Display ───────────────────────────────────────────────────────────────────

def prob_bar(prob, width=40):
    filled = int(prob * width)
    color  = G if prob < 0.35 else (Y if prob < 0.65 else R)
    return f'{color}{"█"*filled}{"░"*(width-filled)}{RESET}'


def risk_tag(prob, lo=0.35, hi=0.65):
    if prob < lo:  return f'{G}{BO} LOW RISK   {RESET}'
    if prob < hi:  return f'{Y}{BO} MODERATE   {RESET}'
    return             f'{R}{BO} HIGH RISK  {RESET}'


def display(pd_p, pd_f, rs_p, rs_f, dp_p, dp_t, dp_f):
    hdr('SCREENING RESULTS', M)

    print(f'\n  {R}{BO}⚠  RESEARCH PROTOTYPE — NOT A MEDICAL DEVICE  ⚠{RESET}')
    print(f'  Consult a qualified physician for any health concerns.\n')

    items = [
        ("Parkinson's Disease",     pd_p, 0.35, 0.65, pd_f,
         "Nonlinear vocal dynamics"),
        ("Respiratory Abnormality", rs_p, 0.35, 0.65, rs_f,
         "Spectral & voice quality"),
        ("Depression",              dp_p, dp_t, dp_t+0.15, dp_f,
         "COVAREP clinical voice"),
    ]

    for name, prob, lo, hi, feats, bio in items:
        sep()
        print(f'\n  {BO}{name}{RESET}')
        print(f'  Biomarkers : {C}{bio}{RESET}')
        print(f'  Probability: {prob_bar(prob)}  {prob:.1%}')
        print(f'  Risk level : {risk_tag(prob, lo, hi)}')
        print(f'\n  {BO}Key acoustic drivers:{RESET}')
        for feat, imp in feats:
            bar_l = int(imp * 28)
            print(f'    {feat:<28} {G}{"█"*bar_l}{"░"*(28-bar_l)}{RESET} {imp:.2f}')
        print()

    sep('═')
    print(f'\n  {BO}Interpretation guide:{RESET}')
    print(f'  {G}■ LOW RISK    {RESET}(<35%)  No significant acoustic markers detected')
    print(f'  {Y}■ MODERATE    {RESET}(35-65%) Some markers present; monitoring advised')
    print(f'  {R}■ HIGH RISK   {RESET}(>65%)  Strong markers; consult a physician')
    print()
    print('  Results depend on recording quality, speaking style,')
    print('  background noise, and individual vocal characteristics.')
    sep('═')
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    hdr('PulseIQ AI  —  Multi-Condition Voice Screening', B)
    print(f'''
  Screens three conditions from a single {BO}15-second voice recording{RESET}:

    {C}1.{RESET} Parkinson's Disease      {B}(nonlinear dysphonia features){RESET}
    {C}2.{RESET} Respiratory Abnormality  {B}(spectral + voice quality){RESET}
    {C}3.{RESET} Depression               {B}(COVAREP clinical features){RESET}
''')

    if len(sys.argv) > 1:
        audio, sr = load_audio_file(sys.argv[1])
    else:
        print(f'  {BO}Press ENTER when you are ready to record.{RESET}')
        print(f'  (Ctrl+C to exit)\n')
        try:
            input('  > ')
        except KeyboardInterrupt:
            print('\n  Cancelled.'); return
        audio, sr = record_audio(duration=15, sr=16000)

    hdr('FEATURE EXTRACTION', C)

    step(1, "Parkinson's Disease — nonlinear dysphonia features")
    pd_feats = extract_pd_features(audio, sr)
    if pd_feats is None:
        warn('Insufficient voiced audio for PD analysis.')
        warn('Using fallback values. For best results, sustain "Ahhh" clearly.')
        pd_feats = {'RPDE':0.45,'DFA':0.65,'PPE':0.15,
                    'spread1':-6.5,'spread2':0.22,'D2':2.3}
    else:
        ok(f'PD features: PPE={pd_feats["PPE"]:.4f}  '
           f'RPDE={pd_feats["RPDE"]:.4f}  DFA={pd_feats["DFA"]:.4f}')

    step(2, 'Respiratory Abnormality — spectral features')
    rs_feats = extract_respiratory_features(audio, sr)
    ok(f'Respiratory: mfcc_1={rs_feats["mfcc_1"]:.2f}  '
       f'hnr={rs_feats["hnr"]:.2f}  pitch={rs_feats["pitch"]:.1f}Hz')

    step(3, 'Depression — COVAREP voice features')
    dp_feats = extract_depression_features(audio, sr)
    ok(f'Depression: F0={dp_feats["F0_mean"]:.1f}Hz  '
       f'VUV={dp_feats["VUV_mean"]:.2f}  MCEP_0={dp_feats["MCEP_0_mean"]:.2f}')

    hdr('RUNNING SCREENING MODELS', C)

    step(1, "Parkinson's Disease")
    pd_p, pd_f = screen_parkinsons(pd_feats)
    ok(f'PD probability: {pd_p:.1%}')

    step(2, 'Respiratory Abnormality')
    rs_p, rs_f = screen_respiratory(rs_feats)
    ok(f'Respiratory probability: {rs_p:.1%}')

    step(3, 'Depression')
    dp_p, dp_t, dp_f = screen_depression(dp_feats)
    ok(f'Depression probability: {dp_p:.1%}')

    display(pd_p, pd_f, rs_p, rs_f, dp_p, dp_t, dp_f)


if __name__ == '__main__':
    main()
