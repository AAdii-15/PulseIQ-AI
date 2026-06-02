"""
PulseIQ AI — MIT Figure-1 style: three side-by-side dashed panels.
Wide landscape, compact, matches the professor's paper format.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path.home() / 'Desktop/PULSE_IQ_AI/results/figures/fig0_pipeline.png'
OUT.parent.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 5.8))
ax.set_xlim(0, 14); ax.set_ylim(0, 5.8)
ax.axis('off')
ax.set_facecolor('white'); fig.patch.set_facecolor('white')
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

# ── Colours ────────────────────────────────────────────────────────────────────
G_DARK='#2E7D4A'; G_MID='#7DC89A'; G_LIGHT='#C8EDD6'; G_VLIGHT='#EAF7EE'
B_DARK='#1E5E8A'; B_MID='#74AED4'; B_LIGHT='#C0DDF0'; B_VLIGHT='#EAF3FA'
A_DARK='#B86010'; A_MID='#E8A845'; A_LIGHT='#FAE0B4'; A_VLIGHT='#FEF5E7'
SALMON='#F7B8B0'; YELLOW='#FDE9A5'; DARK='#1a1a1a'; GRAY='#666'; RED='#CC3333'

# ── Panel dimensions ──────────────────────────────────────────────────────────
PY, PH = 0.18, 5.40          # all panels same y/height
P1 = (0.18, 4.18)             # (x, w) panel 1 — PD
P2 = (4.72, 4.18)             # panel 2 — COVID
P3 = (9.26, 4.56)             # panel 3 — within-recording

def panel(ax, x, y, w, h, title, title_color='#333', bg='white'):
    """Draw a dashed-border panel like MIT Fig 1."""
    ax.add_patch(mpatches.FancyBboxPatch((x,y), w, h,
                                           boxstyle='square,pad=0',
                                           lw=0.9, edgecolor='#888888',
                                           linestyle='--', facecolor=bg, zorder=1))
    ax.text(x+0.18, y+h-0.22, title, ha='left', va='top',
            fontsize=8.2, color=title_color, style='normal', zorder=2)

def rbox(ax,x,y,w,h,text,sub='',fill='white',ec='#555',lw=0.7,
         fs=9,fw='bold',fc='#111',zorder=3):
    ax.add_patch(mpatches.FancyBboxPatch((x,y),w,h,
                                           boxstyle='round,pad=0.05',
                                           lw=lw,edgecolor=ec,facecolor=fill,zorder=zorder))
    cx=x+w/2
    if sub:
        ax.text(cx,y+h*0.65,text,ha='center',va='center',
                fontsize=fs,fontweight=fw,color=fc,zorder=zorder+1)
        ax.text(cx,y+h*0.28,sub,ha='center',va='center',
                fontsize=fs-1.2,color=GRAY,zorder=zorder+1)
    else:
        ax.text(cx,y+h/2,text,ha='center',va='center',
                fontsize=fs,fontweight=fw,color=fc,zorder=zorder+1)

def toks(ax, x, y, n, color, tsz=0.19, gap=0.04, zorder=4):
    for i in range(n):
        ax.add_patch(mpatches.FancyBboxPatch((x+i*(tsz+gap),y),tsz,tsz,
                                               boxstyle='round,pad=0.01',
                                               lw=0,facecolor=color,zorder=zorder))

def uarrow(ax, x, y1, y2, c='#666', lw=0.9):
    ax.annotate('', xy=(x,y2), xytext=(x,y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw, mutation_scale=9))

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 1 — Parkinson's Disease
# ══════════════════════════════════════════════════════════════════════════════
px, pw = P1
panel(ax, px, PY, pw, PH, 'Task 1: Parkinson\'s Disease Detection', G_DARK, G_VLIGHT)

cx1 = px + pw/2
# Input label
ax.text(cx1, PY+0.40, 'UCI dataset  ·  N=195  ·  32 subjects',
        ha='center', va='center', fontsize=7.8, color=GRAY)
# Token row (nonlinear features visualised)
n1 = 10; tsz1 = (pw-0.60)/n1 - 0.04
toks(ax, px+0.30, PY+0.62, n1, G_MID, tsz=tsz1)
uarrow(ax, cx1, PY+0.62+0.19, PY+1.12, G_DARK)
# Feature block
rbox(ax, px+0.28, PY+1.14, pw-0.56, 0.62,
     '6 Nonlinear Features', sub='RPDE, DFA, PPE, spread1, spread2',
     fill=G_DARK, ec=G_DARK, lw=0, fs=8.5, fc='white')
uarrow(ax, cx1, PY+1.76, PY+2.15, G_DARK)
# Model block
rbox(ax, px+0.28, PY+2.17, pw-0.56, 0.62,
     'Random Forest', sub='LOSO (subject-independent)',
     fill=SALMON, ec='#C06060', lw=0.6, fs=8.5, fc=DARK)
uarrow(ax, cx1, PY+2.79, PY+3.12, G_DARK)
# Result block
rbox(ax, px+0.28, PY+3.14, pw-0.56, 0.75,
     'AUROC  0.802', sub='p < 0.001  ·  [0.728 – 0.870]',
     fill=G_LIGHT, ec=G_DARK, lw=0.9, fs=11, fc=G_DARK)
# Small label at top
ax.text(cx1, PY+PH-0.55, 'Six nonlinear features\noutperform all 22 (LOSO)',
        ha='center', va='center', fontsize=7.5, color=G_DARK, style='italic')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 2 — COVID-19 Screening
# ══════════════════════════════════════════════════════════════════════════════
px, pw = P2
panel(ax, px, PY, pw, PH, 'Task 2: COVID-19 Respiratory Screening', B_DARK, B_VLIGHT)

cx2 = px + pw/2
ax.text(cx2, PY+0.40, 'Coswara dataset  ·  N=5,238',
        ha='center', va='center', fontsize=7.8, color=GRAY)
n2 = 10; tsz2 = (pw-0.60)/n2 - 0.04
toks(ax, px+0.30, PY+0.62, n2, B_MID, tsz=tsz2)
uarrow(ax, cx2, PY+0.62+0.19, PY+1.12, B_DARK)
rbox(ax, px+0.28, PY+1.14, pw-0.56, 0.62,
     '19 MFCC Features', sub='13 MFCCs, ZCR, SC, jitter, shimmer, HNR',
     fill=B_DARK, ec=B_DARK, lw=0, fs=8.5, fc='white')
uarrow(ax, cx2, PY+1.76, PY+2.15, B_DARK)
rbox(ax, px+0.28, PY+2.17, pw-0.56, 0.62,
     'Random Forest', sub='5-fold stratified CV',
     fill=SALMON, ec='#C06060', lw=0.6, fs=8.5, fc=DARK)
uarrow(ax, cx2, PY+2.79, PY+3.12, B_DARK)
rbox(ax, px+0.28, PY+3.14, pw-0.56, 0.75,
     'AUROC  0.758', sub='p < 0.001  ·  [0.744 – 0.771]',
     fill=B_LIGHT, ec=B_DARK, lw=0.9, fs=11, fc=B_DARK)
ax.text(cx2, PY+PH-0.55, 'Sensitivity analysis: max duplication\ninflation Δ ≤ 0.022 AUROC',
        ha='center', va='center', fontsize=7.5, color=B_DARK, style='italic')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 3 — Within-recording attribution analysis
# ══════════════════════════════════════════════════════════════════════════════
px, pw = P3
panel(ax, px, PY, pw, PH, 'Within-Recording Attribution Analysis', RED, '#FEF5F4')

cx3 = px + pw/2
ax.text(cx3, PY+0.40, 'Same DAIC-WOZ audio  ·  N=142',
        ha='center', va='center', fontsize=7.8, color=RED)

# Three mini-token rows (NL, MFCC, COVAREP)
sub_w = (pw-0.50)/3 - 0.06
sub_xs = [px+0.25, px+0.25+sub_w+0.06, px+0.25+2*(sub_w+0.06)]
sub_colors = [G_MID, B_MID, A_MID]
sub_labels = ['NL (5)', 'MFCC (26)', 'COV (147)']
n3 = 5; tsz3 = (sub_w-0.10)/n3 - 0.03
for sxs, scol, slbl in zip(sub_xs, sub_colors, sub_labels):
    toks(ax, sxs+0.05, PY+0.62, n3, scol, tsz=tsz3)
    ax.text(sxs+sub_w/2, PY+0.42, slbl,
            ha='center', va='center', fontsize=7.2, color=GRAY)

# Merge arrows upward
for sxs in sub_xs:
    uarrow(ax, sxs+sub_w/2, PY+0.62+0.19, PY+1.12, '#888', 0.7)

# Three small feature boxes
small_fills = [G_LIGHT, B_LIGHT, A_LIGHT]
small_ecs   = [G_DARK,  B_DARK,  A_DARK ]
small_lbls  = ['Nonlinear', 'MFCC', 'COVAREP']
for sxs,sf,se,sl in zip(sub_xs, small_fills, small_ecs, small_lbls):
    rbox(ax, sxs, PY+1.14, sub_w, 0.52, sl,
         fill=sf, ec=se, lw=0.7, fs=7.8, fc=DARK)

# Converging arrows
for sxs in sub_xs:
    ax.annotate('', xy=(cx3, PY+1.84),
                xytext=(sxs+sub_w/2, PY+1.66),
                arrowprops=dict(arrowstyle='->', color='#888',
                               lw=0.7, mutation_scale=8))

# Classifier block
rbox(ax, px+0.30, PY+1.86, pw-0.60, 0.58,
     'Depression Classifier', sub='SVM + SelectKBest  ·  AVEC 2017',
     fill=YELLOW, ec='#C0A010', lw=0.8, fs=8.5, fc=DARK)
uarrow(ax, cx3, PY+2.44, PY+2.74, '#888')

# Attribution results (three mini badges side by side)
badge_w = (pw-0.50)/3 - 0.06
for i,(bc,blbl,bval) in enumerate(
    [(G_DARK,'NL (PD-type)','2.6%'),
     (B_DARK,'MFCC','23.2%'),
     (A_DARK,'COVAREP','74.2%')]):
    bx = px+0.25 + i*(badge_w+0.06)
    ax.add_patch(mpatches.FancyBboxPatch((bx,PY+2.76),badge_w,0.72,
                                           boxstyle='round,pad=0.04',
                                           lw=0.8,edgecolor=bc,
                                           facecolor='white',zorder=3))
    ax.text(bx+badge_w/2, PY+2.76+0.50, bval,
            ha='center', va='center',
            fontsize=11, fontweight='bold', color=bc, zorder=4)
    ax.text(bx+badge_w/2, PY+2.76+0.20, blbl,
            ha='center', va='center',
            fontsize=7.5, color=GRAY, zorder=4)

# Panel 3 top callout
ax.text(cx3, PY+PH-0.42,
        'PD-type NL features: lowest in all analyses',
        ha='center', va='center',
        fontsize=7.8, fontweight='bold', color=RED, style='italic')
ax.text(cx3, PY+PH-0.72,
        '2.6% (full)  ·  12.4% (−COV)  ·  26.5% (matched)',
        ha='center', va='center', fontsize=7.5, color='#555')

# ── Footnote ──────────────────────────────────────────────────────────────────
ax.text(0.20, 0.08,
        '† DAIC-WOZ analysis: p=0.110, N=35 dev, N≥159 required. '
        'No clinical screening claim is supported.',
        ha='left', va='center', fontsize=7.5, color='#999', style='italic')

plt.savefig(OUT, dpi=200, facecolor='white', edgecolor='none')
print(f'Saved → {OUT}')
plt.close()