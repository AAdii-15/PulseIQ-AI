"""
PulseIQ AI — BSPC Figure 1: FRAMEWORK-FIRST pipeline diagram.
Input -> Shared Feature Space -> 4 Framework Components -> Attribution
Separability Assessment. Disease names appear only as small examples,
NOT as the main visual organizing principle.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path.home()/'Desktop/PULSE_IQ_AI/results/figures/fig0_framework.png'

W, H = 17.6, 7.5
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.axis('off')
ax.set_facecolor('white')
fig.patch.set_facecolor('white')
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

C = dict(
    input_bg='#F5F5F5', input_bord='#888888',
    shap='#2196F3', card='#4CAF50', abl='#FF9800', transfer='#9C27B0',
    output_bg='#E8F5E9', output_bord='#2E7D32',
    dark='#1A1A1A', mid='#555555', white='#FFFFFF',
)

def box(x, y, w, h, fc, ec, lw=1.4, r=0.08, alpha=1.0, z=3):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle=f'round,pad=0,rounding_size={r}',
        fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z))

def txt(x, y, s, sz=9, weight='normal', color='#1A1A1A', ha='center', va='center', z=5, style='normal'):
    ax.text(x, y, s, ha=ha, va=va, fontsize=sz, fontweight=weight,
            color=color, fontfamily='DejaVu Sans', zorder=z, style=style)

def arr(x1, y1, x2, y2, color='#333333', lw=1.8, z=4):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), zorder=z,
        arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=15))

# ══════════════════════════════════════════════════════════════════
# STAGE 0 — INPUT
# ══════════════════════════════════════════════════════════════════
box(0.3, 5.1, 2.3, 1.6, C['input_bg'], C['input_bord'])
txt(1.45, 6.45, 'Input', sz=10, weight='bold')
txt(1.45, 6.05, 'Voice recording(s);', sz=7.8, color=C['mid'])
txt(1.45, 5.75, 'one or more condition-', sz=7.8, color=C['mid'])
txt(1.45, 5.48, 'specific classifiers', sz=7.8, color=C['mid'])

arr(2.6, 5.9, 3.05, 5.9)

# ══════════════════════════════════════════════════════════════════
# STAGE 1 — SHARED FEATURE SPACE
# ══════════════════════════════════════════════════════════════════
box(3.05, 4.85, 2.3, 2.1, '#EDE7F6', '#7B1FA2')
txt(4.2, 6.65, 'Shared Feature', sz=9.5, weight='bold', color='#4A148C')
txt(4.2, 6.35, 'Space', sz=9.5, weight='bold', color='#4A148C')
txt(4.2, 5.92, 'Multiple condition-', sz=7.6, color=C['mid'])
txt(4.2, 5.65, 'associated feature', sz=7.6, color=C['mid'])
txt(4.2, 5.38, 'families extracted from', sz=7.6, color=C['mid'])
txt(4.2, 5.11, 'the same recording', sz=7.6, color=C['mid'])

arr(5.35, 5.9, 5.85, 5.9)

# ══════════════════════════════════════════════════════════════════
# STAGE 2 — FOUR FRAMEWORK COMPONENTS (the heart of the figure)
# ══════════════════════════════════════════════════════════════════
panel_x, panel_w = 5.85, 8.4
box(panel_x, 0.35, panel_w, 6.9, '#FAFAFA', '#BBBBBB', r=0.1)
txt(panel_x+panel_w/2, 7.0, 'Attribution Separability Framework', sz=11.5, weight='bold')

comp_w, comp_h = 3.95, 2.85
gx = 0.15
comp_x = [panel_x+0.15, panel_x+0.15+comp_w+gx]
comp_y = [3.55, 0.65]

components = [
    ("1. Within-Recording\nSHAP Attribution",
     "TreeSHAP on the shared\nfeature space; computes\nper-group % attribution",
     C['shap'], comp_x[0], comp_y[0]),
    ("2. Cardinality\nControl",
     "Matched-size resampling\n(equal group size, 200 reps)\nremoves group-size confound",
     C['card'], comp_x[1], comp_y[0]),
    ("3. Group\nAblation",
     "Paired AUROC \u0394 on feature-\ngroup removal; an attribution-\nindependent corroboration",
     C['abl'], comp_x[0], comp_y[1]),
    ("4. Cross-Condition\nTransfer Test",
     "Applies a classifier trained on\none condition to another's\nfeatures \u2014 a falsifiable check",
     C['transfer'], comp_x[1], comp_y[1]),
]

for title, desc, color, bx, by in components:
    box(bx, by, comp_w, comp_h, color, color, lw=0, alpha=0.10)
    box(bx, by+comp_h-0.85, comp_w, 0.85, color, color, lw=0, r=0.08)
    txt(bx+comp_w/2, by+comp_h-0.42, title, sz=9.8, weight='bold', color=C['white'])
    txt(bx+comp_w/2, by+comp_h/2-0.45, desc, sz=7.8, color=C['dark'])

# Arrows connecting the 4 components (1->2->3->4 logical flow, dashed)
arr(comp_x[0]+comp_w, comp_y[0]+comp_h/2, comp_x[1], comp_y[0]+comp_h/2,
    color='#999999', lw=1.2)
arr(comp_x[0]+comp_w/2, comp_y[0], comp_x[0]+comp_w/2, comp_y[1]+comp_h,
    color='#999999', lw=1.2)
arr(comp_x[0]+comp_w, comp_y[1]+comp_h/2, comp_x[1], comp_y[1]+comp_h/2,
    color='#999999', lw=1.2)

arr(panel_x+panel_w, 3.8, panel_x+panel_w+0.5, 3.8)

# ══════════════════════════════════════════════════════════════════
# STAGE 3 — OUTPUT
# ══════════════════════════════════════════════════════════════════
out_x = panel_x+panel_w+0.5
box(out_x, 2.8, 2.3, 2.0, C['output_bg'], C['output_bord'], lw=1.8)
txt(out_x+1.15, 4.45, 'Output', sz=10, weight='bold', color=C['output_bord'])
txt(out_x+1.15, 4.05, 'Attribution', sz=8.5, weight='bold', color=C['dark'])
txt(out_x+1.15, 3.78, 'Separability', sz=8.5, weight='bold', color=C['dark'])
txt(out_x+1.15, 3.51, 'Assessment', sz=8.5, weight='bold', color=C['dark'])
txt(out_x+1.15, 3.10, '(condition-associated', sz=7, color=C['mid'])
txt(out_x+1.15, 2.90, 'vs. shared reliance)', sz=7, color=C['mid'])

# ══════════════════════════════════════════════════════════════════
# BOTTOM STRIP — illustrative example (small, NOT the main story)
# ══════════════════════════════════════════════════════════════════
txt(W/2, 0.18,
    'Illustrative demonstration in this paper: PD detection (UCI), COVID-19 screening (Coswara), '
    'and an exploratory depression case study (DAIC-WOZ) \u2014 see Section III for dataset details.',
    sz=7.5, color='#888888', style='italic')

plt.savefig(OUT, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'Saved -> {OUT} (600dpi, BSPC)')
