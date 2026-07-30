"""
BSPC Graphical Abstract — wide banner format (2.5:1 ratio).
Spec: min 531x1328px (h x w), readable at 5x13cm, PDF preferred.
Fixed: canvas resized so nothing clips; removed bbox_inches='tight'
which was cropping content.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path.home()/'Desktop/PULSE_IQ_AI/results/figures/graphical_abstract_bspc.pdf'

W, H = 17.7, 7.1   # 2.49:1 ratio, matches BSPC's 1328:531 spec
fig = plt.figure(figsize=(W, H), facecolor='white')
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.axis('off')

C = dict(nl='#4CAF50', mfcc='#2196F3', cov='#FF9800',
         purple='#7B1FA2', navy='#1A237E',
         bg='#F8F9FA', dark='#1A1A1A', mid='#555555')

def box(x, y, w, h, fc, ec='none', lw=1.2, r=0.1, alpha=1, z=3):
    ax.add_patch(mpatches.FancyBboxPatch((x,y), w, h,
        boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec,
        lw=lw, alpha=alpha, zorder=z))

def t(x, y, s, sz=9, w='normal', c='#1A1A1A', ha='center', va='center', z=5):
    ax.text(x, y, s, ha=ha, va=va, fontsize=sz, fontweight=w, color=c,
            fontfamily='DejaVu Sans', zorder=z)

def arr(x1, y1, x2, y2, c='#888888', lw=2.0):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1), zorder=4,
        arrowprops=dict(arrowstyle='-|>', color=c, lw=lw, mutation_scale=16))

# Layout: title bar at TOP (1.0 high), content below (6.1 high)
TITLE_H = 1.0
CONTENT_TOP = H - 0.15
CONTENT_BOT = TITLE_H + 0.25

# ── Title bar at TOP ──────────────────────────────────────────
box(0.4, H-1.0, 16.9, 0.8, C['navy'], C['navy'], lw=0, r=0.06)
t(8.85, H-0.6,
  'A Within-Recording Framework for Evaluating Acoustic Attribution Separability in Voice Biomarker Models',
  sz=11, w='bold', c='white')

# ── Block 1: Input ────────────────────────────────────────────
box(0.4, 1.3, 2.6, 3.9, C['bg'], '#888888', lw=1.4)
t(1.7, 4.7, 'Voice', sz=13, w='bold')
t(1.7, 4.3, 'Recording', sz=13, w='bold')
t(1.7, 3.55, 'Shared feature', sz=9, c=C['mid'])
t(1.7, 3.25, 'space across', sz=9, c=C['mid'])
t(1.7, 2.95, 'condition-specific', sz=9, c=C['mid'])
t(1.7, 2.65, 'classifiers', sz=9, c=C['mid'])
arr(3.15, 3.25, 3.7, 3.25)

# ── Block 2: 4-component framework ───────────────────────────
box(3.7, 0.5, 7.0, 4.7, '#FAFAFA', '#BBBBBB', r=0.12)
t(7.2, 4.95, 'Within-Recording Attribution', sz=12, w='bold')
t(7.2, 4.65, 'Separability Framework', sz=12, w='bold')

comps = [
    ('1. SHAP\nAttribution',         C['mfcc'],   3.95, 2.55),
    ('2. Cardinality\nControl',      C['nl'],     7.65, 2.55),
    ('3. Group\nAblation',           C['cov'],    3.95, 0.7),
    ('4. Cross-Condition\nTransfer', C['purple'], 7.65, 0.7),
]
for label, col, bx, by in comps:
    box(bx, by, 3.0, 1.65, col, col, lw=0, alpha=0.85, r=0.1)
    t(bx+1.5, by+0.825, label, sz=10.5, w='bold', c='white')

arr(10.7, 3.25, 11.25, 3.25)

# ── Block 3: Key finding ─────────────────────────────────────
box(11.25, 0.8, 3.0, 4.4, '#E8F5E9', C['nl'], lw=2.4, r=0.12)
t(12.75, 4.85, 'KEY FINDING', sz=10.5, w='bold', c=C['nl'])
t(12.75, 4.3, 'Nonlinear', sz=13, w='bold', c=C['nl'])
t(12.75, 3.95, 'features lowest', sz=13, w='bold', c=C['nl'])
t(12.75, 3.6, 'attribution', sz=13, w='bold', c=C['nl'])
t(12.75, 3.05, '2.6%', sz=10.5, w='bold', c=C['dark'])
t(12.75, 2.8, 'full feature space', sz=8, c=C['mid'])
t(12.75, 2.45, '26.5%', sz=10.5, w='bold', c=C['dark'])
t(12.75, 2.2, 'matched cardinality', sz=8, c=C['mid'])
t(12.75, 1.85, 'p > 0.05', sz=10.5, w='bold', c=C['dark'])
t(12.75, 1.6, 'no cross-condition transfer', sz=8, c=C['mid'])
t(12.75, 1.1, 'Exploratory case study', sz=7.5, c='#888888')

# ── PD/COVID quick stats strip ────────────────────────────────
box(14.55, 0.8, 2.75, 4.4, C['bg'], '#CCCCCC', lw=1.2, r=0.1)
t(15.93, 4.85, 'Other Tasks', sz=9.5, w='bold', c=C['dark'])
t(15.93, 4.35, 'PD Detection', sz=8.5, w='bold')
t(15.93, 4.05, 'AUROC 0.802', sz=9, c=C['mid'])
t(15.93, 3.5, 'COVID-19', sz=8.5, w='bold')
t(15.93, 3.2, 'AUROC 0.758', sz=9, c=C['mid'])
t(15.93, 2.5, 'Classification', sz=7.5, c='#888888')
t(15.93, 2.25, 'demonstrations', sz=7.5, c='#888888')
t(15.93, 2.0, 'of the same', sz=7.5, c='#888888')
t(15.93, 1.75, 'framework', sz=7.5, c='#888888')

ax.set_xlim(0, W)
ax.set_ylim(0, H)

plt.savefig(OUT, facecolor='white', edgecolor='none')
plt.close()
print(f'Saved -> {OUT}')
