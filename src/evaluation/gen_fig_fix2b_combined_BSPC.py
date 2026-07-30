"""
Fig 2: Within-recording SHAP attribution — Full feature space (A) vs
Counterfactual without COVAREP (B).

Changes from the previous BSPC version:
  1. BUG FIX: label placement now uses each bar's own (val, ci_lo, ci_hi)
     via zip() -- the prior version placed the "2.6% [CI]" text at a
     hardcoded x=1.0, which lands on the MFCC bar, not the Nonlinear bar
     it's meant to label. This is the same fix gen_fig_fix2b_v2.py already
     made; it just never got merged into the BSPC-worded/600dpi branch.
  2. Removed fig.suptitle() and the bottom fig.text() note. That content
     (why Panel B exists, what the error bars are) now lives in the
     paper's Results prose and figure caption instead of being baked
     into the image pixels.
  3. Feature-group labels matched exactly to Table II's wording in the
     manuscript: "Nonlinear (PD-type)", not "Nonlinear (dynamics)".
  4. Kept 600dpi (BSPC's color/halftone requirement).
All data values UNCHANGED from the validated original.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path.home()/'Desktop/PULSE_IQ_AI/results/figures'

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

colors = {'COVAREP\n(phonatory)':   '#7B1FA2',
          'MFCC\n(spectral)':       '#388E3C',
          'Nonlinear\n(PD-type)':   '#C62828',
          'Other\n(ZCR, SC, HNR)':  '#0277BD'}

# ── Panel A: Original (with COVAREP) ─────────────────────────────────────────
ax = axes[0]
labels_a  = ['COVAREP\n(phonatory)', 'MFCC\n(spectral)', 'Nonlinear\n(PD-type)']
values_a  = [74.2, 23.2, 2.6]
ci_lo_a   = [65.0, 12.0, 0.89]
ci_hi_a   = [83.0, 31.0, 5.25]
clrs_a    = [colors[l] for l in labels_a]
yerr_lo_a = [v - lo for v, lo in zip(values_a, ci_lo_a)]
yerr_hi_a = [hi - v for v, hi in zip(values_a, ci_hi_a)]

bars = ax.bar(labels_a, values_a, color=clrs_a, alpha=0.88, width=0.55,
              yerr=[yerr_lo_a, yerr_hi_a], capsize=6,
              error_kw={'elinewidth': 2, 'ecolor': '#333', 'capthick': 2})
for bar, val, lo, hi in zip(bars, values_a, ci_lo_a, ci_hi_a):
    ax.text(bar.get_x() + bar.get_width()/2, hi + 2.5,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(bar.get_x() + bar.get_width()/2, hi + 0.3,
             f'[{lo:.1f}\u2013{hi:.1f}]', ha='center', va='bottom',
             fontsize=7.5, color='#444')

ax.set_ylim(0, 95); ax.set_ylabel('Mean |SHAP| Attribution (%)', fontsize=11)
ax.set_title('(A)  Full Feature Space\n(COVAREP + MFCC + Nonlinear)',
             fontsize=11, fontweight='bold', pad=10)
ax.axhline(50, color='grey', lw=0.8, ls='--', alpha=0.4)
ax.text(2.4, 52, '50%', color='grey', fontsize=8)
ax.set_facecolor('#FAFAFA'); ax.grid(axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

# ── Panel B: Counterfactual (without COVAREP) ─────────────────────────────────
ax = axes[1]
labels_b = ['MFCC\n(spectral)', 'Nonlinear\n(PD-type)', 'Other\n(ZCR, SC, HNR)']
values_b = [77.86, 12.40, 9.74]
ci_lo_b  = [68.41, 7.11,  4.49]
ci_hi_b  = [86.59, 19.31, 18.58]
clrs_b   = [colors[l] for l in labels_b]
yerr_lo_b = [v - lo for v, lo in zip(values_b, ci_lo_b)]
yerr_hi_b = [hi - v for v, hi in zip(values_b, ci_hi_b)]

bars = ax.bar(labels_b, values_b, color=clrs_b, alpha=0.88, width=0.55,
              yerr=[yerr_lo_b, yerr_hi_b], capsize=6,
              error_kw={'elinewidth': 2, 'ecolor': '#333', 'capthick': 2})
for bar, val, lo, hi in zip(bars, values_b, ci_lo_b, ci_hi_b):
    ax.text(bar.get_x() + bar.get_width()/2, hi + 2.5,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(bar.get_x() + bar.get_width()/2, hi + 0.3,
             f'[{lo:.1f}\u2013{hi:.1f}]', ha='center', va='bottom',
             fontsize=7.5, color='#444')

ax.set_ylim(0, 95); ax.set_ylabel('Mean |SHAP| Attribution (%)', fontsize=11)
ax.set_title('(B)  Counterfactual \u2014 COVAREP Removed\n(MFCC + Nonlinear only)',
             fontsize=11, fontweight='bold', pad=10)
ax.axhline(50, color='grey', lw=0.8, ls='--', alpha=0.4)
ax.set_facecolor('#FAFAFA'); ax.grid(axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

# No suptitle, no bottom fig.text note -- that content now lives in the
# Results prose and figure caption in the manuscript, not the image.
plt.tight_layout()
plt.savefig(OUT/'fig_fix2b_combined.png', dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print('Saved -> fig_fix2b_combined.png (600dpi, label bug fixed, no in-image heading)')
