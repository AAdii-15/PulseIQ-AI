"""
Fig 3: Two-panel — SHAP Jaccard stability (left) + Power analysis curve (right)

Corrected after discovering the depression AUROC used throughout the
paper was wrong (0.626/0.634, untraceable to any current pipeline) --
now 0.621, the mean across 10 seeds (depression_final_v2.py). The
required-N line moves from 159 to 196 accordingly (a smaller observed
effect needs more data to confirm at 80% power -- verified via the same
Hanley-McNeil procedure already used elsewhere in this repo, sanity-
checked to reproduce the existing 0.757 MDE-at-N=35 value exactly).
The "AVEC 2017 baseline = 0.630" line is UNCHANGED -- that is the
challenge's own published historical benchmark (Ringeval et al.), not
our result, and is unrelated to this fix.
600dpi, no in-image heading (unchanged from the prior fix).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from pathlib import Path

OUT = Path.home()/'Desktop/PULSE_IQ_AI/results/figures'
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# ── Panel A: SHAP Jaccard stability comparison ────────────────────────────────
ax = axes[0]
conditions  = ["Parkinson's\n(PD)", "COVID-19\nRespiratory", "Depression\n(case study)"]
jaccard_5   = [1.000, 1.000, 0.063]
jaccard_10  = [1.000, 0.918, 0.114]
colors_cond = ['#1565C0', '#2E7D32', '#B71C1C']
x = np.arange(len(conditions)); w = 0.35

b1 = ax.bar(x-w/2, jaccard_5,  w, label='Top-5  features', color=colors_cond,
            alpha=0.9, edgecolor='white', linewidth=1.5)
b2 = ax.bar(x+w/2, jaccard_10, w, label='Top-10 features', color=colors_cond,
            alpha=0.5, edgecolor='white', linewidth=1.5, hatch='//')

for bar, val in zip(b1, jaccard_5):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.02,
            f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
for bar, val in zip(b2, jaccard_10):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.02,
            f'{val:.3f}', ha='center', fontsize=9, color='#444')

ax.set_xticks(x); ax.set_xticklabels(conditions, fontsize=10)
ax.set_ylim(0, 1.18); ax.set_ylabel('Jaccard Set Overlap (20 bootstraps)', fontsize=10)
ax.set_title('(A)  SHAP Feature Ranking Stability\nAcross 20 Bootstrap Replicates',
             fontsize=11, fontweight='bold', pad=10)
ax.axhline(0.6, color='#FF6F00', lw=1.5, ls='--', alpha=0.7, label='Min. acceptable (0.6)')
ax.text(2.55, 0.62, 'Acceptable\nthreshold', fontsize=8, color='#FF6F00')
ax.set_facecolor('#FAFAFA'); ax.grid(axis='y', alpha=0.3)
ax.spines[['top','right']].set_visible(False)
from matplotlib.patches import Patch
legend_els = [Patch(facecolor='grey', alpha=0.9, label='Top-5 features'),
              Patch(facecolor='grey', alpha=0.5, hatch='//', label='Top-10 features')]
ax.legend(handles=legend_els, fontsize=9, loc='upper right')
ax.annotate('Noise\nfloor', xy=(2-w/2, 0.063), xytext=(1.35, 0.25),
            arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=1.5),
            fontsize=9, color='#B71C1C', fontweight='bold')

# ── Panel B: Power analysis curve ─────────────────────────────────────────────
ax = axes[1]

def hanley_se(auc, n1, n2):
    Q1, Q2 = auc/(2-auc), 2*auc**2/(1+auc)
    v = (auc*(1-auc)+(n1-1)*(Q1-auc**2)+(n2-1)*(Q2-auc**2))/(n1*n2)
    return np.sqrt(max(v, 1e-10))

za, zb = norm.ppf(0.975), norm.ppf(0.80)
N_range = np.arange(20, 500, 5)
mde_list = []
for N in N_range:
    n1, n2 = max(2, int(N*0.343)), max(2, N-int(N*0.343))
    auc = 0.55
    for _ in range(150):
        new = 0.5 + (za+zb)*hanley_se(max(auc,0.501), n1, n2)
        if abs(new-auc) < 1e-5: break
        auc = new
    mde_list.append(min(auc, 0.99))

ax.plot(N_range, mde_list, color='#1565C0', lw=2.5, label='Min. detectable AUROC\n(80% power, \u03b1=0.05)')
ax.fill_between(N_range, mde_list, 0.5, alpha=0.08, color='#1565C0')

ax.axhline(0.621, color='#B71C1C', lw=2, ls='--', label='Observed AUROC = 0.621')
ax.axhline(0.630, color='#FF6F00', lw=1.5, ls=':', alpha=0.7, label='AVEC 2017 baseline = 0.630')
ax.axvline(35,   color='#B71C1C', lw=2,   ls='--')
ax.axvline(196,  color='#2E7D32', lw=1.8, ls='-.',
           label='Required N for observed effect = 196')

ax.annotate('N = 35\n(this study)', xy=(35, 0.76), xytext=(70, 0.85),
            arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=1.5),
            fontsize=9, color='#B71C1C', fontweight='bold')
ax.annotate('N = 196\nrequired', xy=(196, 0.621), xytext=(260, 0.70),
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5),
            fontsize=9, color='#2E7D32', fontweight='bold')

ax.fill_betweenx([0.5, 1.0], 0, 35, alpha=0.06, color='#B71C1C', label='Underpowered region')
ax.set_xlim(15, 500); ax.set_ylim(0.5, 0.98)
ax.set_xlabel('Total Sample Size (N)', fontsize=10)
ax.set_ylabel('Minimum Detectable AUROC', fontsize=10)
ax.set_title('(B)  Statistical Power Analysis\nDepression Task (29.6% prevalence)',
             fontsize=11, fontweight='bold', pad=10)
ax.legend(fontsize=8.5, loc='upper right')
ax.set_facecolor('#FAFAFA'); ax.grid(alpha=0.3)
ax.spines[['top','right']].set_visible(False)

# No suptitle -- moved to Results prose and figure caption.
plt.tight_layout()
plt.savefig(OUT/'fig_stability_power.png', dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print('Saved -> fig_stability_power.png (600dpi, AUROC 0.621, N>=196, AVEC baseline 0.630 unchanged)')
