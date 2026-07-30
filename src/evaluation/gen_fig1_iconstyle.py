"""
Figure 1 redesign — icon-based pipeline diagram in the style of the
reference image (dataset icons -> mic -> two branches -> converging
performance panel). Illustrative waveform/spectrogram panels are
synthetic (schematic), not a specific recording's real analysis output --
consistent with how Figure 1 in most papers is a schematic overview,
not a results figure.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Wedge
from pathlib import Path

BASE = Path.home() / 'Desktop/PULSE_IQ_AI'
OUT = BASE / 'results/figures/fig0_pipeline.png'

# ── Canvas ───────────────────────────────────────────────────────────────────
W, H = 16.5, 9.0
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.axis('off')
ax.set_facecolor('white'); fig.patch.set_facecolor('white')
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

INK    = '#1a1a1a'
GREY   = '#555555'
LGREY  = '#999999'
NL_C   = '#C62828'   # nonlinear / PD branch
MFCC_C = '#2E7D4A'   # mfcc / covid branch
COV_C  = '#7B1FA2'   # covarep
GOLD   = '#B8860B'
BLUE   = '#1565C0'

def person(ax, x, y, s=0.16, color=INK, lw=1.4):
    """Minimal line-art person icon (head + shoulders), reference-style."""
    head = Circle((x, y + s*1.5), s*0.6, fill=False, ec=color, lw=lw, zorder=5)
    ax.add_patch(head)
    body = Wedge((x, y - s*1.5), s*1.5, 20, 160, width=0, fill=False,
                 ec=color, lw=lw, zorder=5)
    ax.add_patch(body)

def person_group(ax, cx, cy, n=3, spread=0.34, s=0.15, color=INK):
    xs = np.linspace(cx - spread*(n-1)/2, cx + spread*(n-1)/2, n)
    for xi in xs:
        person(ax, xi, cy, s=s, color=color)

def mic_icon(ax, x, y, w=0.5, h=0.9, color=INK, lw=1.8):
    """Simple line-art microphone."""
    capsule = FancyBboxPatch((x - w/2, y), w, h, boxstyle=f'round,pad=0,rounding_size={w/2}',
                              fill=False, ec=color, lw=lw, zorder=5)
    ax.add_patch(capsule)
    arc = Wedge((x, y), w*1.15, 200, 340, width=0.06, fill=True, fc=color,
                ec='none', zorder=5)
    ax.add_patch(arc)
    ax.plot([x, x], [y - w*1.15, y - w*1.75], color=color, lw=lw, zorder=5)
    ax.plot([x - w*0.45, x + w*0.45], [y - w*1.75, y - w*1.75], color=color, lw=lw, zorder=5)

def doc_icon(ax, x, y, w=0.5, h=0.65, color=INK, lw=1.5, n_lines=4):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                                 boxstyle='round,pad=0.015,rounding_size=0.03',
                                 fill=False, ec=color, lw=lw, zorder=5))
    for i in range(n_lines):
        yy = y + h/2 - 0.12 - i * (h - 0.2) / (n_lines - 1)
        ax.plot([x - w/2 + 0.08, x + w/2 - 0.08], [yy, yy], color=color, lw=1.0, zorder=5)

def brain_box(ax, x, y, w=0.9, h=0.55, color=INK, lw=1.6, label='Random\nForest'):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                                 boxstyle='round,pad=0.02,rounding_size=0.08',
                                 fill=True, fc='#F5F5F5', ec=color, lw=lw, zorder=5))
    ax.text(x, y, label, ha='center', va='center', fontsize=8.3, color=color,
            fontweight='bold', zorder=6)

def arrow(ax, x1, y1, x2, y2, color='#888888', lw=1.6, style='-|>', rad=0.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), zorder=3,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                 mutation_scale=13, connectionstyle=f'arc3,rad={rad}'))

def txt(ax, x, y, s, sz=9, w='normal', c=INK, ha='center', va='center', style='normal'):
    ax.text(x, y, s, ha=ha, va=va, fontsize=sz, fontweight=w, color=c,
            fontfamily='DejaVu Sans', style=style, zorder=6)

# ══════════════════════════════════════════════════════════════════════════════
# TOP TITLE
# ══════════════════════════════════════════════════════════════════════════════
txt(ax, W/2, 8.82, 'Evaluation-Rigor and Within-Recording Attribution Pipeline',
    sz=15, w='bold')
ax.axhline(y=8.55, xmin=0.02, xmax=0.98, lw=0.5, color='#DDDDDD', zorder=1)

# ══════════════════════════════════════════════════════════════════════════════
# LEFT: three datasets -> converging to mic
# ══════════════════════════════════════════════════════════════════════════════
ds = [
    ('UCI Parkinson\u2019s', 'N=195 \u00b7 32 subjects', 6.9, NL_C),
    ('Coswara',              'N=5,238 recordings',       4.5, MFCC_C),
    ('DAIC-WOZ',             'N=142 sessions',           2.1, COV_C),
]
for name, sub, y, col in ds:
    person_group(ax, 1.15, y, n=3, s=0.13, color=col)
    txt(ax, 1.15, y - 0.62, name, sz=9.3, w='bold', c=col)
    txt(ax, 1.15, y - 0.86, sub, sz=7.6, c=GREY)
    arrow(ax, 1.85, y, 2.75, 4.5, color=col, lw=1.5,
          rad=(0.12 if y > 4.5 else (-0.12 if y < 4.5 else 0)))

mic_icon(ax, 3.05, 4.15, w=0.42, h=0.78, color=INK)
txt(ax, 3.05, 3.0, 'Voice\nrecording', sz=8.6, w='bold')

# ══════════════════════════════════════════════════════════════════════════════
# SPLIT into two branches
# ══════════════════════════════════════════════════════════════════════════════
arrow(ax, 3.55, 4.7, 4.45, 6.55, color=INK, lw=1.6, rad=0.12)
arrow(ax, 3.55, 3.6, 4.55, 1.55, color=INK, lw=1.6, rad=-0.12)

# ── TOP BRANCH: subject-independent classification ────────────────────────────
txt(ax, 6.15, 8.18, 'Subject-independent classification',
    sz=10.3, w='bold', c=NL_C)
txt(ax, 6.15, 7.90, '(UCI \u00b7 LOSO)  and  (Coswara \u00b7 5-fold CV)',
    sz=7.8, c=GREY, style='italic')

# illustrative waveform panel
wax = fig.add_axes([4.62/W, 6.35/H, 1.70/W, 1.35/H])
t = np.linspace(0, 3, 1200)
env = np.exp(-1.15 * t) + 0.12
sig = env * (np.sin(2*np.pi*7*t) + 0.35*np.sin(2*np.pi*19*t) + 0.15*np.random.default_rng(3).standard_normal(t.size))
wax.plot(t, sig, color=BLUE, lw=0.6)
wax.set_xlim(0, 3); wax.set_ylim(-1.6, 1.6)
wax.set_xticks([0, 1, 2, 3]); wax.set_yticks([])
wax.tick_params(labelsize=6, length=2)
wax.set_xlabel('Time (s)', fontsize=6.5, labelpad=1)
for spine in ['top', 'right']:
    wax.spines[spine].set_visible(False)

# illustrative spectrogram panel
sax = fig.add_axes([6.62/W, 6.35/H, 1.70/W, 1.35/H])
freqs = np.linspace(0, 4000, 90)
spec = np.zeros((90, 120))
rng = np.random.default_rng(7)
for k, f0 in enumerate([180, 360, 540, 720, 900, 1250, 1700]):
    band = np.exp(-((freqs - f0)[:, None])**2 / (2*45**2))
    wobble = 1 + 0.25*np.sin(np.linspace(0, 6*np.pi, 120) + k)
    spec += band * wobble[None, :] * (0.9**k)
spec += 0.05 * rng.standard_normal(spec.shape)
sax.imshow(spec, aspect='auto', origin='lower', cmap='magma',
           extent=[0, 3, 0, 4000])
sax.set_xticks([0, 1.5, 3]); sax.set_yticks([0, 2000, 4000])
sax.tick_params(labelsize=6, length=2, colors=INK)
sax.set_xlabel('Time (s)', fontsize=6.5, labelpad=1)
sax.set_ylabel('Hz', fontsize=6.5, labelpad=1)

arrow(ax, 8.5, 7.0, 9.35, 7.0, color=NL_C, lw=1.6)
brain_box(ax, 10.35, 7.0, w=1.35, h=0.6, color=NL_C,
          label='Calibrated\nRandom Forest')
arrow(ax, 11.05, 7.0, 11.95, 7.0, color=NL_C, lw=1.6)

# AUROC formula + result box
ax.add_patch(FancyBboxPatch((12.05, 6.55), 2.55, 0.95,
             boxstyle='round,pad=0.03,rounding_size=0.08',
             fill=True, fc='#FCECEA', ec=NL_C, lw=1.4, zorder=4))
txt(ax, 13.32, 7.28, r'$AUROC = \frac{1}{n_1 n_2}\sum \mathbb{1}[f(x_i){>}f(x_j)]$',
    sz=8.3, c=NL_C)
txt(ax, 13.32, 6.85, 'PD 0.802$^{***}$   \u00b7   COVID 0.758$^{***}$',
    sz=8.6, w='bold', c=INK)

# ── BOTTOM BRANCH: within-recording attribution ────────────────────────────────
txt(ax, 6.15, 3.55, 'Within-recording attribution (case study)',
    sz=10.3, w='bold', c=COV_C)
txt(ax, 6.15, 3.25, '(DAIC-WOZ \u00b7 same audio, 3 feature families)',
    sz=7.8, c=GREY, style='italic')

fx = [4.75, 5.65, 6.55]
fam_colors = [NL_C, MFCC_C, COV_C]
fam_labels = ['Nonlinear\n(5)', 'MFCC\n(26)', 'COVAREP\n(147)']
for xx, col, lbl in zip(fx, fam_colors, fam_labels):
    ax.add_patch(FancyBboxPatch((xx - 0.36, 1.6), 0.72, 0.58,
                 boxstyle='round,pad=0.02,rounding_size=0.06',
                 fill=True, fc=col, ec=col, lw=0, alpha=0.85, zorder=5))
    txt(ax, xx, 1.89, lbl, sz=6.9, c='white', w='bold')
    arrow(ax, xx, 1.58, 7.45, 1.38, color=col, lw=1.2,
          rad=(0.1 if xx < 5.65 else (-0.1 if xx > 5.65 else 0)))

ax.add_patch(FancyBboxPatch((7.6, 1.05), 1.55, 0.6,
             boxstyle='round,pad=0.02,rounding_size=0.08',
             fill=True, fc='#F5F5F5', ec=INK, lw=1.6, zorder=5))
txt(ax, 8.37, 1.35, 'TreeSHAP', sz=8.6, w='bold')
arrow(ax, 9.15, 1.35, 10.0, 1.35, color=COV_C, lw=1.6)

ax.add_patch(FancyBboxPatch((10.1, 0.75), 2.9, 1.2,
             boxstyle='round,pad=0.03,rounding_size=0.08',
             fill=True, fc='#F3E9F7', ec=COV_C, lw=1.4, zorder=4))
txt(ax, 11.55, 1.63, r'$a_g = \dfrac{\sum_{i \in g}\bar{\phi}_i}{\sum_{j \in F}\bar{\phi}_j}\times 100\%$',
    sz=8.6, c=COV_C)
txt(ax, 11.55, 1.02, 'NL 2.6%  \u00b7  MFCC 23.2%  \u00b7  COVAREP 74.2%',
    sz=8.0, w='bold', c=INK)

# ══════════════════════════════════════════════════════════════════════════════
# CONVERGE right -> rigor / performance panel
# ══════════════════════════════════════════════════════════════════════════════
arrow(ax, 14.65, 7.0, 15.35, 4.85, color=NL_C, lw=1.5, rad=-0.08)
arrow(ax, 13.0, 1.35, 15.15, 4.15, color=COV_C, lw=1.5, rad=0.12)

ax.add_patch(FancyBboxPatch((14.85, 3.55), 1.45, 1.9,
             boxstyle='round,pad=0.03,rounding_size=0.1',
             fill=True, fc='#EAF2FB', ec=BLUE, lw=1.8, zorder=4))
# small growth-bars icon
bar_x0 = 15.15
for i, hgt in enumerate([0.25, 0.42, 0.62]):
    ax.add_patch(mpatches.Rectangle((bar_x0 + i*0.28, 4.55), 0.2, hgt,
                 fill=False, ec=BLUE, lw=1.6, zorder=5))
arrow(ax, 15.05, 4.5, 16.15, 5.35, color=BLUE, lw=1.6)
txt(ax, 15.57, 3.85, 'Leakage-corrected,\ncalibrated, power-\nanalyzed results',
    sz=7.6, w='bold', c=BLUE)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTNOTE
# ══════════════════════════════════════════════════════════════════════════════
txt(ax, W/2, 0.28,
    'Waveform and spectrogram panels are schematic illustrations of the acoustic '
    'signal, not a specific analysed recording. Attribution branch uses DAIC-WOZ '
    'only; PD/COVID branches use separate datasets and feature spaces.',
    sz=7.3, c=LGREY, style='italic')

plt.savefig(OUT, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'Saved -> {OUT}')
