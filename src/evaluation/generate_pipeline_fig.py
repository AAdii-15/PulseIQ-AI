"""
PulseIQ AI — methodology pipeline (clean block-diagram style).
v3: within-recording panel content pushed down; panel title has clear space.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path.home() / 'Desktop/PULSE_IQ_AI/results/figures/fig0_pipeline.png'
OUT.parent.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 8.0))
ax.set_xlim(0, 16); ax.set_ylim(0, 8.0)
ax.axis('off')
ax.set_facecolor('white'); fig.patch.set_facecolor('white')
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

C_PD    = '#ECF3FC'
C_COV   = '#EAF6F1'
C_DEP   = '#FDF5E7'
C_NEUT  = '#F7F7F7'
C_PANEL = '#FEF3F2'
BORDER  = '#4A4A4A'
DGRAY   = '#606060'
RED     = '#BB4433'

# 4 columns (x_left, width)
C = [(0.20, 2.60), (3.22, 3.10), (6.74, 3.00), (10.15, 5.25)]
BOX_H = 1.12
ROW_Y = [6.30, 4.95, 3.60]   # shifted up to give more room for panel

def box(ax, x, y, w, h, title, sub='', fill=C_NEUT, lw=0.6):
    p = mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04',
                                  lw=lw, edgecolor=BORDER, facecolor=fill, zorder=2)
    ax.add_patch(p)
    cx = x + w / 2
    if sub:
        ax.text(cx, y + h*0.63, title, ha='center', va='center',
                fontsize=10.5, fontweight='bold', color='#181818', zorder=3)
        ax.text(cx, y + h*0.26, sub, ha='center', va='center',
                fontsize=9.0, color=DGRAY, zorder=3)
    else:
        ax.text(cx, y + h/2, title, ha='center', va='center',
                fontsize=10.5, fontweight='bold', color='#181818', zorder=3)

def arr(ax, x1, yc, x2):
    ax.annotate('', xy=(x2, yc), xytext=(x1, yc),
                arrowprops=dict(arrowstyle='->', color='#888888',
                               lw=0.9, mutation_scale=11))

# ── Column headers
for (x, w), lbl in zip(C, ['Datasets', 'Feature extraction',
                             'Model & protocol', 'Attribution']):
    ax.text(x + w/2, 7.78, lbl, ha='center', va='center',
            fontsize=11.5, fontweight='bold', color='#1a1a1a')
ax.axhline(y=7.55, xmin=0.013, xmax=0.987, lw=0.5, color='#CCCCCC')

# ── ROW 1: Parkinson's
box(ax, C[0][0], ROW_Y[0], C[0][1], BOX_H, "UCI Parkinson's", "N=195 · 32 subjects", C_PD)
box(ax, C[1][0], ROW_Y[0], C[1][1], BOX_H, "22 features", "incl. 6 nonlinear  (RPDE, DFA, PPE…)")
box(ax, C[2][0], ROW_Y[0], C[2][1], BOX_H, "Random Forest", "LOSO  (subject-independent)")
box(ax, C[3][0], ROW_Y[0], C[3][1], BOX_H, "AUROC 0.802", "p < 0.001   [0.728 – 0.870]")
cy = ROW_Y[0] + BOX_H/2
for i in range(3): arr(ax, C[i][0]+C[i][1]+0.02, cy, C[i+1][0]-0.02)

# ── ROW 2: COVID-19
box(ax, C[0][0], ROW_Y[1], C[0][1], BOX_H, "Coswara COVID-19", "N=5,238 recordings", C_COV)
box(ax, C[1][0], ROW_Y[1], C[1][1], BOX_H, "19 features", "13 MFCCs, ZCR, SC, HNR…")
box(ax, C[2][0], ROW_Y[1], C[2][1], BOX_H, "Random Forest", "5-fold stratified CV")
box(ax, C[3][0], ROW_Y[1], C[3][1], BOX_H, "AUROC 0.758", "p < 0.001   [0.744 – 0.771]")
cy = ROW_Y[1] + BOX_H/2
for i in range(3): arr(ax, C[i][0]+C[i][1]+0.02, cy, C[i+1][0]-0.02)

# ── ROW 3: Depression
box(ax, C[0][0], ROW_Y[2], C[0][1], BOX_H, "DAIC-WOZ  †", "N=142  ·  exploratory", C_DEP)
box(ax, C[1][0], ROW_Y[2], C[1][1], BOX_H, "COVAREP features", "147 features  (H1H2, MCEP…)")
box(ax, C[2][0], ROW_Y[2], C[2][1], BOX_H, "SVM", "SelectKBest  ·  AVEC 2017")
box(ax, C[3][0], ROW_Y[2], C[3][1], BOX_H, "AUROC 0.626 ns", "p = 0.110  (underpowered)")
cy = ROW_Y[2] + BOX_H/2
for i in range(3): arr(ax, C[i][0]+C[i][1]+0.02, cy, C[i+1][0]-0.02)

# ══════════════════════════════════════════════════════════════════════════════
# WITHIN-RECORDING PANEL
# Panel title sits alone at top. All boxes start 0.55" below the title.
# ══════════════════════════════════════════════════════════════════════════════
PX, PY, PW, PH = 0.20, 0.38, 15.20, 2.88
panel_top = PY + PH   # = 3.26

p = mpatches.FancyBboxPatch((PX, PY), PW, PH, boxstyle='round,pad=0.06',
                               lw=1.1, edgecolor=RED, linestyle='--',
                               facecolor=C_PANEL, zorder=1)
ax.add_patch(p)

# Panel title — top strip, well above all box content
TITLE_Y = panel_top - 0.28
ax.text(PX + PW/2, TITLE_Y,
        "Within-recording attribution analysis   (same DAIC-WOZ audio)",
        ha='center', va='center', fontsize=10.5,
        fontweight='bold', color=RED, zorder=4)

# Thin separator below title
SEP_Y = TITLE_Y - 0.22
ax.axhline(y=SEP_Y, xmin=(PX+0.15)/16, xmax=(PX+PW-0.15)/16,
           lw=0.5, color='#DDAAAA', zorder=4)

# ── Content zone: all boxes stay below SEP_Y - 0.10 (= 2.66)
# Feature boxes
CONTENT_TOP = SEP_Y - 0.12   # = ~2.54
FX = 0.40; FW = 2.10; FH = 0.43; FGAP = 0.13
FY = [CONTENT_TOP - FH,
      CONTENT_TOP - FH - FGAP - FH,
      CONTENT_TOP - FH - FGAP - FH - FGAP - FH]
# FY[0]=2.11, FY[1]=1.55, FY[2]=0.99

FC = [C_PD, C_COV, C_DEP]
FL = ['NL features  (5)', 'MFCC features  (26)', 'COVAREP  (147 features)']
for fy, fc, fl in zip(FY, FC, FL):
    p2 = mpatches.FancyBboxPatch((FX, fy), FW, FH, boxstyle='round,pad=0.03',
                                   lw=0.6, edgecolor=BORDER, facecolor=fc, zorder=3)
    ax.add_patch(p2)
    ax.text(FX + FW/2, fy + FH/2, fl, ha='center', va='center',
            fontsize=9.5, color='#222222', fontweight='bold', zorder=4)

# CY = centre of the middle feature box
CY = FY[1] + FH/2   # ≈ 1.765

# Merge wires
MX = FX + FW + 0.20
for fy in FY:
    fc_y = fy + FH/2
    ax.plot([FX+FW, MX], [fc_y, fc_y], color='#999999', lw=0.85, zorder=3)
ax.plot([MX, MX], [FY[2]+FH/2, FY[0]+FH/2], color='#999999', lw=0.85, zorder=3)

# Arrow: merge → classifier
CLF_X = MX + 0.26; CLF_W = 3.0; CLF_H = 0.72
arr(ax, MX+0.02, CY, CLF_X-0.02)
box(ax, CLF_X, CY - CLF_H/2, CLF_W, CLF_H,
    "Depression classifier", "SVM + SelectKBest")

# Arrow: classifier → results
RX = CLF_X + CLF_W + 0.26
arr(ax, CLF_X+CLF_W+0.02, CY, RX-0.02)

# Results box — top at CONTENT_TOP
RW = 3.65; RH = CONTENT_TOP - (PY + 0.22)   # fills from ~panel bottom to content_top
RY = PY + 0.22   # bottom of results box ≈ 0.60
RH = CONTENT_TOP - RY   # ≈ 1.94

res = mpatches.FancyBboxPatch((RX, RY), RW, RH, boxstyle='round,pad=0.04',
                               lw=0.6, edgecolor=BORDER, facecolor='white', zorder=3)
ax.add_patch(res)
ax.text(RX + RW/2, RY + RH - 0.24, "SHAP attribution",
        ha='center', va='center', fontsize=10.5,
        fontweight='bold', color='#1a1a1a', zorder=4)
ax.axhline(y=RY+RH-0.46, xmin=(RX+0.12)/16, xmax=(RX+RW-0.12)/16,
           lw=0.4, color='#CCCCCC')
rows = [("NL  (PD-type):", "2.6%"),
        ("MFCC  (spectral):", "23.2%"),
        ("COVAREP:", "74.2%")]
entry_ys = [RY+RH-0.76, RY+RH-1.22, RY+RH-1.68]
for (lbl, val), ey in zip(rows, entry_ys):
    ax.text(RX+0.18, ey, lbl, ha='left', va='center',
            fontsize=9.5, color=DGRAY, zorder=4)
    ax.text(RX+RW-0.18, ey, val, ha='right', va='center',
            fontsize=10.5, fontweight='bold', color='#1a1a1a', zorder=4)

# Callout box (same height as results)
KX = RX + RW + 0.30; KW = 4.85
cal = mpatches.FancyBboxPatch((KX, RY), KW, RH, boxstyle='round,pad=0.04',
                               lw=0.6, edgecolor='#AAAAAA', linestyle='--',
                               facecolor='white', zorder=3)
ax.add_patch(cal)
ax.text(KX + KW/2, RY+RH-0.24, "NL consistently lowest",
        ha='center', va='center', fontsize=10.5,
        fontweight='bold', color='#333333', zorder=4)
ax.axhline(y=RY+RH-0.46, xmin=(KX+0.12)/16, xmax=(KX+KW-0.12)/16,
           lw=0.4, color='#CCCCCC')
krows = ["2.6%     full feature space  (Tier 1 primary)",
         "12.4%   without COVAREP  (counterfactual)",
         "26.5%   cardinality-matched  5+5+5"]
for txt, ey in zip(krows, entry_ys):
    ax.text(KX+0.22, ey, txt, ha='left', va='center',
            fontsize=9.3, color='#333333', zorder=4)

# Footnote (below panel)
ax.text(0.22, 0.20,
        "† Underpowered exploratory analysis  (p=0.110, N=35 dev set)."
        "  N ≥ 159 required.  No clinical screening claim is supported.",
        ha='left', va='center', fontsize=8.8, color='#888888', style='italic')

plt.savefig(OUT, dpi=200, facecolor='white', edgecolor='none')
print(f'Saved → {OUT}')
plt.close()