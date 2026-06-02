"""
Fig: Fix 2B Two-Panel — Original + Counterfactual
The headline figure for the rewritten paper.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path.home()/'Desktop/PULSE_IQ_AI/results/figures'

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

colors = {'COVAREP\n(depression-specific)': '#7B1FA2',
          'MFCC\n(respiratory-type)':        '#388E3C',
          'Nonlinear\n(PD-type)':            '#C62828',
          'Other\n(ZCR, SC, HNR)':           '#0277BD'}

# ── Panel A: Original (with COVAREP) ─────────────────────────────────────────
ax = axes[0]
labels_a  = ['COVAREP\n(depression-specific)','MFCC\n(respiratory-type)',
              'Nonlinear\n(PD-type)']
values_a  = [74.2, 23.2, 2.6]
ci_lo_a   = [65.0, 12.0, 0.89]
ci_hi_a   = [83.0, 31.0, 5.25]
clrs_a    = [colors[l] for l in labels_a]
yerr_lo_a = [v-lo for v,lo in zip(values_a, ci_lo_a)]
yerr_hi_a = [hi-v for v,hi in zip(values_a, ci_hi_a)]

bars = ax.bar(labels_a, values_a, color=clrs_a, alpha=0.88, width=0.55,
              yerr=[yerr_lo_a, yerr_hi_a], capsize=6,
              error_kw={'elinewidth':2,'ecolor':'#333','capthick':2})
for bar, val in zip(bars, values_a):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylim(0, 95); ax.set_ylabel('Mean |SHAP| Attribution (%)', fontsize=11)
ax.set_title('(A)  Full Feature Space\n(COVAREP + MFCC + Nonlinear)',
             fontsize=11, fontweight='bold', pad=10)
ax.axhline(50, color='grey', lw=0.8, ls='--', alpha=0.4)
ax.text(2.4, 52, '50%', color='grey', fontsize=8)
ax.set_facecolor('#FAFAFA'); ax.grid(axis='y', alpha=0.3)
ax.spines[['top','right']].set_visible(False)
ax.text(1.0, 3.5, '2.6%\n[0.89–5.25]', ha='center', va='bottom',
        fontsize=8.5, color='white', fontweight='bold')

# ── Panel B: Counterfactual (without COVAREP) ─────────────────────────────────
ax = axes[1]
labels_b = ['MFCC\n(respiratory-type)','Nonlinear\n(PD-type)','Other\n(ZCR, SC, HNR)']
values_b = [77.86, 12.40, 9.74]
ci_lo_b  = [68.41, 7.11,  4.49]
ci_hi_b  = [86.59, 19.31, 18.58]
clrs_b   = [colors[l] for l in labels_b]
yerr_lo_b= [v-lo for v,lo in zip(values_b, ci_lo_b)]
yerr_hi_b= [hi-v for v,hi in zip(values_b, ci_hi_b)]

bars = ax.bar(labels_b, values_b, color=clrs_b, alpha=0.88, width=0.55,
              yerr=[yerr_lo_b, yerr_hi_b], capsize=6,
              error_kw={'elinewidth':2,'ecolor':'#333','capthick':2})
for bar, val in zip(bars, values_b):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylim(0, 95); ax.set_ylabel('Mean |SHAP| Attribution (%)', fontsize=11)
ax.set_title('(B)  Counterfactual — COVAREP Removed\n(MFCC + Nonlinear only)',
             fontsize=11, fontweight='bold', pad=10)
ax.axhline(50, color='grey', lw=0.8, ls='--', alpha=0.4)
ax.set_facecolor('#FAFAFA'); ax.grid(axis='y', alpha=0.3)
ax.spines[['top','right']].set_visible(False)
ax.text(1.0, 13.5, '12.4%\n[7.1–19.3]', ha='center', va='bottom',
        fontsize=8.5, color='white', fontweight='bold')

# ── Shared annotation ─────────────────────────────────────────────────────────
fig.suptitle(
    'Depression SHAP Attribution by Feature Type — Original vs. Counterfactual\n'
    'In both conditions, PD-type nonlinear features receive far less attribution than MFCC-type features',
    fontsize=11, fontweight='bold', y=1.02)

fig.text(0.5, -0.04,
    'Error bars: 95% bootstrap CI (200 resamples).  '
    'Panel B retrains the depression classifier on shared features only (no COVAREP), '
    'addressing the circularity concern.',
    ha='center', fontsize=8.5, color='#444', style='italic', wrap=True)

plt.tight_layout()
plt.savefig(OUT/'fig_fix2b_combined.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print('Saved -> fig_fix2b_combined.png')
