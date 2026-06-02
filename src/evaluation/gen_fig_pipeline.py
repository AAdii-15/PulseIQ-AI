"""
Fig 1: Methodology Pipeline
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path

OUT = Path.home()/'Desktop/PULSE_IQ_AI/results/figures'

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis('off')

def box(ax, x, y, w, h, text, color, fontsize=9, text_color='white', bold=False):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                           facecolor=color, edgecolor='white', linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x+w/2, y+h/2, text, ha='center', va='center',
            fontsize=fontsize, color=text_color, fontweight=weight,
            wrap=True, zorder=4)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#555555',
                                lw=1.8, connectionstyle='arc3,rad=0.0'))

C = {'data':'#37474F', 'feat':'#1565C0', 'model':'#2E7D32',
     'shap':'#6A1B9A', 'fix':'#C62828', 'head':'#212121'}

# Title
ax.text(7, 6.65, 'PulseIQ AI — Methodology Pipeline',
        ha='center', va='center', fontsize=13, fontweight='bold', color=C['head'])

# ── Column 1: Datasets ────────────────────────────────────────────────────────
ax.text(1.5, 6.15, 'Datasets', ha='center', fontsize=9, color='#555', style='italic')
box(ax,0.2,4.8,2.6,1.1,'UCI Parkinson\'s\n195 recordings\n32 subjects',C['data'])
box(ax,0.2,3.3,2.6,1.1,'Coswara\nCOVID-19\n5,238 recordings',C['data'])
box(ax,0.2,1.8,2.6,1.1,'DAIC-WOZ\nDepression\n142 sessions',C['data'])

# ── Column 2: Feature Extraction ─────────────────────────────────────────────
ax.text(5.2, 6.15, 'Feature Extraction', ha='center', fontsize=9, color='#555', style='italic')
box(ax,3.7,4.8,3.0,1.1,'22 Acoustic Features\n(Nonlinear: RPDE,DFA,PPE,\nspread1,spread2,D2)',C['feat'],8)
box(ax,3.7,3.3,3.0,1.1,'19 Acoustic Features\n(13 MFCCs, Pitch,\nJitter, Shimmer, HNR)',C['feat'],8)
box(ax,3.7,1.8,3.0,1.1,'146 COVAREP Features\n(Mean+Std of F0, MCEP,\nHMPDM, HMPDD, H1H2...)',C['feat'],8)

# ── Column 3: Models ──────────────────────────────────────────────────────────
ax.text(8.9, 6.15, 'Models & Protocols', ha='center', fontsize=9, color='#555', style='italic')
box(ax,7.5,4.8,2.8,1.1,'Random Forest\n(6 nonlinear features)\nLOSO',C['model'],8)
box(ax,7.5,3.3,2.8,1.1,'Random Forest\n(19 features)\n5-Fold CV',C['model'],8)
box(ax,7.5,1.8,2.8,1.1,'SVM + SelectKBest\n(k=30 features)\nAVEC 2017 Split',C['model'],8)

# ── Column 4: SHAP ────────────────────────────────────────────────────────────
ax.text(12.1, 6.15, 'Attribution', ha='center', fontsize=9, color='#555', style='italic')
box(ax,10.9,4.8,2.5,1.1,'TreeSHAP\nTop features:\nspread1,spread2,PPE',C['shap'],8)
box(ax,10.9,3.3,2.5,1.1,'TreeSHAP\nTop features:\nmfcc_10,mfcc_6,mfcc_8',C['shap'],8)
box(ax,10.9,1.8,2.5,1.1,'Bootstrap SHAP\nAggregate type\nattribution only',C['shap'],8)

# ── Fix 2B shared space ───────────────────────────────────────────────────────
box(ax,3.7,0.2,7.0,1.1,
    'Within-Recording Attribution Analysis (Fix 2B): PD-type + MFCC-type + COVAREP features '
    'extracted from same DAIC-WOZ audio\n'
    'Counterfactual: MFCC=77%  Nonlinear=12%  |  With COVAREP: MFCC=23%  Nonlinear=2.6%',
    C['fix'], fontsize=8, bold=False)

# ── Arrows ────────────────────────────────────────────────────────────────────
for y in [5.35, 3.85, 2.35]:
    arrow(ax, 2.8, y, 3.7, y)
    arrow(ax, 6.7, y, 7.5, y)
    arrow(ax, 10.3, y, 10.9, y)

arrow(ax, 5.2, 1.8, 5.2, 1.3)
arrow(ax, 8.9, 1.8, 8.9, 1.3)

# ── AUROC Results ─────────────────────────────────────────────────────────────
ax.text(11.2, 4.3, 'AUROC 0.802***', ha='center', fontsize=8, color='#2E7D32', fontweight='bold')
ax.text(11.2, 2.85, 'AUROC 0.758***', ha='center', fontsize=8, color='#2E7D32', fontweight='bold')
ax.text(11.2, 1.35, 'AUROC 0.626 ns', ha='center', fontsize=8, color='#B71C1C', fontweight='bold')

plt.tight_layout()
plt.savefig(OUT/'fig0_pipeline.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print('Saved -> fig0_pipeline.png')
