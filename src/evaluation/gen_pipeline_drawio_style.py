"""
PulseIQ AI — Pipeline Figure, draw.io style.
Clean flat blocks, sharp borders, no gradients, no shadows.
Looks exactly like a diagram made in draw.io / PowerPoint.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path.home() / 'Desktop/PULSE_IQ_AI/results/figures/fig0_pipeline.png'
OUT.parent.mkdir(parents=True, exist_ok=True)

W, H = 18, 7.2
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.axis('off')
ax.set_facecolor('white')
fig.patch.set_facecolor('white')
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

# ── draw.io flat colour palette ────────────────────────────────────
C = dict(
    # PD panel
    pd_bg    = '#F9FBF4',
    pd_feat  = '#6AAB5B',   # dark green (UCI features)
    pd_clf   = '#D9534F',   # classifier red
    pd_res   = '#D9534F',
    pd_bord  = '#3D8C31',

    # COVID panel
    cv_bg    = '#F5F9FE',
    cv_feat  = '#3A78C9',   # deep blue
    cv_clf   = '#D9534F',
    cv_res   = '#D9534F',
    cv_bord  = '#2558A6',

    # Attribution panel
    at_bg    = '#FFF9F2',
    at_bord  = '#C77B24',
    dep_clf  = '#F0C050',   # yellow
    nl_box   = '#6AAB5B',
    mf_box   = '#3A78C9',
    co_box   = '#E0943A',

    # Text colours
    white    = '#FFFFFF',
    dark     = '#1A1A1A',
    mid      = '#444444',
    sub      = '#666666',
    red_lbl  = '#B94040',
)

# ── Helpers ────────────────────────────────────────────────────────
def box(x, y, w, h, fc, ec, lw=1.4, ls='-', alpha=1.0, zorder=3):
    p = mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle='round,pad=0,rounding_size=0.06',
        fc=fc, ec=ec, lw=lw, ls=ls, alpha=alpha, zorder=zorder)
    ax.add_patch(p)

def panel(x, y, w, h, fc, ec, lw=1.6, ls='--', zorder=1):
    p = mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle='round,pad=0,rounding_size=0.12',
        fc=fc, ec=ec, lw=lw, ls=ls, zorder=zorder)
    ax.add_patch(p)

def txt(x, y, s, sz=9, weight='normal', color='#1A1A1A',
        ha='center', va='center', style='normal', zorder=5):
    ax.text(x, y, s, ha=ha, va=va, fontsize=sz,
            fontweight=weight, color=color, style=style,
            fontfamily='DejaVu Sans', zorder=zorder)

def arr(x1, y1, x2, y2, color='#333333', lw=1.5, zorder=4):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), zorder=zorder,
                arrowprops=dict(arrowstyle='-|>', color=color,
                                lw=lw, mutation_scale=14))

# ══════════════════════════════════════════════════════════════════
# PANEL 1 — Parkinson's Disease Detection
# ══════════════════════════════════════════════════════════════════
P1x, P1w = 0.25, 4.75

panel(P1x, 0.55, P1w, 6.05, C['pd_bg'], C['pd_bord'])

# Title
txt(P1x+P1w/2, 6.35, "Task 1: Parkinson's Disease Detection",
    sz=9.5, weight='bold', color=C['pd_bord'])
txt(P1x+P1w/2, 5.98, "Six nonlinear features outperform all 22 (LOSO)",
    sz=8.2, color=C['sub'], style='italic')

# AUROC result box
box(P1x+0.3, 4.7, 4.2, 0.95, C['pd_clf'], C['pd_clf'], lw=0)
txt(P1x+P1w/2, 5.22, "AUROC  0.802",
    sz=13, weight='bold', color=C['white'])
txt(P1x+P1w/2, 4.85, "p < 0.001   ·   [0.728 – 0.870]",
    sz=8.5, color=C['white'])

# Arrow
arr(P1x+P1w/2, 4.70, P1x+P1w/2, 4.05)

# Random Forest box
box(P1x+0.3, 3.25, 4.2, 0.75, '#F8F8F8', '#888888', lw=1.2)
txt(P1x+P1w/2, 3.68, "Random Forest", sz=9.5, weight='bold')
txt(P1x+P1w/2, 3.38, "LOSO  (subject-independent)", sz=8, color=C['sub'])

# Arrow
arr(P1x+P1w/2, 3.25, P1x+P1w/2, 2.60)

# Feature box
box(P1x+0.3, 1.75, 4.2, 0.82, C['pd_feat'], C['pd_feat'], lw=0)
txt(P1x+P1w/2, 2.21, "6 Nonlinear Features", sz=9.5, weight='bold', color=C['white'])
txt(P1x+P1w/2, 1.90, "RPDE · DFA · PPE · spread1 · spread2",
    sz=7.5, color='#E8F5E9')

# Arrow from data
arr(P1x+P1w/2, 1.75, P1x+P1w/2, 1.15)

# Dataset label
txt(P1x+P1w/2, 0.82, "UCI dataset  ·  N = 195  ·  32 subjects",
    sz=8, color=C['mid'])
# "Classification only" badge
box(P1x+0.6, 5.62, 3.6, 0.28, '#F0F0F0', '#AAAAAA', lw=1.0)
txt(P1x+P1w/2, 5.76, "Classification only  ·  separate dataset",
    sz=7, color='#666666', style='italic')

# ══════════════════════════════════════════════════════════════════
# PANEL 2 — COVID-19 Respiratory Screening
# ══════════════════════════════════════════════════════════════════
P2x, P2w = 5.25, 4.75

panel(P2x, 0.55, P2w, 6.05, C['cv_bg'], C['cv_bord'])

txt(P2x+P2w/2, 6.35, "Task 2: COVID-19 Respiratory Screening",
    sz=9.5, weight='bold', color=C['cv_bord'])
txt(P2x+P2w/2, 5.98, "Sensitivity analysis: max duplication inflation Δ ≤ 0.022 AUROC",
    sz=8.2, color=C['sub'], style='italic')

# AUROC
box(P2x+0.3, 4.7, 4.2, 0.95, C['cv_clf'], C['cv_clf'], lw=0)
txt(P2x+P2w/2, 5.22, "AUROC  0.758",
    sz=13, weight='bold', color=C['white'])
txt(P2x+P2w/2, 4.85, "p < 0.001   ·   [0.744 – 0.771]",
    sz=8.5, color=C['white'])

arr(P2x+P2w/2, 4.70, P2x+P2w/2, 4.05)

# Classifier
box(P2x+0.3, 3.25, 4.2, 0.75, '#F8F8F8', '#888888', lw=1.2)
txt(P2x+P2w/2, 3.68, "Random Forest", sz=9.5, weight='bold')
txt(P2x+P2w/2, 3.38, "5-fold stratified CV", sz=8, color=C['sub'])

arr(P2x+P2w/2, 3.25, P2x+P2w/2, 2.60)

# Features
box(P2x+0.3, 1.75, 4.2, 0.82, C['cv_feat'], C['cv_feat'], lw=0)
txt(P2x+P2w/2, 2.21, "19 MFCC Features", sz=9.5, weight='bold', color=C['white'])
txt(P2x+P2w/2, 1.90, "13 MFCCs · ZCR · SC · jitter · shimmer · HNR",
    sz=7.5, color='#E3F2FD')

arr(P2x+P2w/2, 1.75, P2x+P2w/2, 1.15)

txt(P2x+P2w/2, 0.82, "Coswara dataset  ·  N = 5,238",
    sz=8, color=C['mid'])
# "Classification only" badge
box(P2x+0.6, 5.62, 3.6, 0.28, '#F0F0F0', '#AAAAAA', lw=1.0)
txt(P2x+P2w/2, 5.76, "Classification only  ·  separate dataset",
    sz=7, color='#666666', style='italic')

# ══════════════════════════════════════════════════════════════════
# PANEL 3 — Within-Recording Attribution Analysis
# ══════════════════════════════════════════════════════════════════
P3x, P3w = 10.25, 7.50

panel(P3x, 0.55, P3w, 6.05, C['at_bg'], C['at_bord'])

txt(P3x+P3w/2, 6.35, "Within-Recording Attribution Analysis",
    sz=9.5, weight='bold', color=C['at_bord'])
txt(P3x+P3w/2, 5.98, "PD-type NL features: lowest in all analyses",
    sz=8.2, color=C['red_lbl'], weight='bold', style='italic')
box(P3x+0.8, 5.62, 5.9, 0.28, '#FFF0E0', '#D79B00', lw=1.0)
txt(P3x+P3w/2, 5.76,
    "Within-recording attribution · depression classifier only · PD/COVID attribution = future work",
    sz=7, color='#7A4B00', style='italic')

# Attribution values row
AW = 2.0   # attribution box width
AH = 0.88  # attribution box height
A_Y = 4.6
A_CX = [P3x+0.65+AW*0+0.0*0.2,
        P3x+0.65+AW*1+0.2,
        P3x+0.65+AW*2+0.4]

# NL
box(A_CX[0], A_Y, AW, AH, C['nl_box'], C['nl_box'], lw=0)
txt(A_CX[0]+AW/2, A_Y+0.59, "2.6%", sz=14, weight='bold', color=C['white'])
txt(A_CX[0]+AW/2, A_Y+0.22, "NL (PD-type)", sz=8, color='#E8F5E9')

# MFCC
box(A_CX[1], A_Y, AW, AH, C['mf_box'], C['mf_box'], lw=0)
txt(A_CX[1]+AW/2, A_Y+0.59, "23.2%", sz=14, weight='bold', color=C['white'])
txt(A_CX[1]+AW/2, A_Y+0.22, "MFCC", sz=8, color='#E3F2FD')

# COVAREP
box(A_CX[2], A_Y, AW, AH, C['co_box'], C['co_box'], lw=0)
txt(A_CX[2]+AW/2, A_Y+0.59, "74.2%", sz=14, weight='bold', color=C['white'])
txt(A_CX[2]+AW/2, A_Y+0.22, "COVAREP", sz=8, color='#FFF8F0')

# Dashed arrows from classifier up to attribution values
for cx in A_CX:
    arr(P3x+P3w/2, 3.90, cx+AW/2, A_Y, color='#999999', lw=1.0)

# Depression Classifier box
CX, CY, CW, CH = P3x+0.5, 3.1, 6.6, 0.75
box(CX, CY, CW, CH, C['dep_clf'], '#B8940A', lw=1.4)
txt(P3x+P3w/2, 3.54, "Depression Classifier", sz=9.5, weight='bold')
txt(P3x+P3w/2, 3.24, "Random Forest (attribution)  ·  SVM (primary)  ·  AVEC 2017",
    sz=7.8, color='#5A4B00')

# Arrows from features to classifier
FBW = 1.85; FBH = 0.72
fb_y = 1.95
fb_cx = [P3x+0.75, P3x+0.75+FBW+0.2, P3x+0.75+2*(FBW+0.2)]
fb_colors = [C['nl_box'], C['mf_box'], C['co_box']]
fb_labels = ["Nonlinear", "MFCC", "COVAREP"]
fb_subs   = ["(5)", "(26)", "(147)"]
fb_tcol   = [C['white'], C['white'], C['white']]

for i, (fx, fc, lb, sb, tc) in enumerate(zip(fb_cx, fb_colors, fb_labels, fb_subs, fb_tcol)):
    box(fx, fb_y, FBW, FBH, fc, fc, lw=0)
    txt(fx+FBW/2, fb_y+0.46, lb, sz=9, weight='bold', color=tc)
    txt(fx+FBW/2, fb_y+0.18, sb, sz=8.5, color=tc)
    arr(fx+FBW/2, fb_y+FBH, fx+FBW/2, CY)

# Shared audio label with arrows
arr(fb_cx[0]+FBW/2, 1.95, fb_cx[0]+FBW/2, 1.32)
arr(fb_cx[1]+FBW/2, 1.95, fb_cx[1]+FBW/2, 1.32)
arr(fb_cx[2]+FBW/2, 1.95, fb_cx[2]+FBW/2, 1.32)

# Data label
txt(P3x+P3w/2, 0.90, "Same DAIC-WOZ audio  ·  N = 142",
    sz=8.5, color=C['red_lbl'], weight='bold')
txt(P3x+P3w/2, 0.65, "NL (5)   MFCC (26)   COV (147)   —  178 shared features",
    sz=7.5, color=C['mid'])

# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════
txt(W/2, 0.22,
    "† DAIC-WOZ: p = 0.110, N = 35 dev, N ≥ 159 required. "
    "No clinical screening claim is supported.",
    sz=7.5, color='#888888', style='italic')

plt.savefig(OUT, dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"Saved → {OUT}")