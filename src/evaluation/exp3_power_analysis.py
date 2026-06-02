"""
Hanley-McNeil power analysis for the depression task (N=35, 12 dep, 23 non).
Computes the minimum detectable AUROC at 80% power and required N for
observed effect.
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import norm

BASE = Path.home()/'Desktop/PULSE_IQ_AI'
RES  = BASE/'results/metrics'

n1, n2, alpha, power = 12, 23, 0.05, 0.80
za, zb = norm.ppf(1-alpha/2), norm.ppf(power)

def se(auc, a, b):
    Q1, Q2 = auc/(2-auc), 2*auc**2/(1+auc)
    return np.sqrt((auc*(1-auc) + (a-1)*(Q1-auc**2) + (b-1)*(Q2-auc**2)) / (a*b))

auc = 0.55
for _ in range(200):
    new = 0.5 + (za+zb)*se(max(auc,0.501), n1, n2)
    if abs(new-auc) < 1e-5: break
    auc = new

print('='*55)
print(' DEPRESSION POWER ANALYSIS')
print('='*55)
print(f'N (dep/non/total)        : {n1}/{n2}/{n1+n2}')
print(f'Alpha (two-sided)        : {alpha}    Power: {power}')
print(f'Minimum Detectable AUROC : {auc:.3f}')
print(f'Observed AUROC           : 0.634')
print(f'Gap                      : {auc - 0.634:+.3f}')
print()
print('INTERPRETATION:')
print(f'  Study at N=35 can only detect AUROC ≥ {auc:.2f} with 80% power.')
print(f'  Observed 0.634 is {"BELOW" if 0.634 < auc else "above"} this threshold —')
print(f'  p=0.110 reflects underpowering, not absence of effect.')
print()
print('REQUIRED SAMPLE SIZES (34%/66% class split, 80% power):')
for tgt in [0.634, 0.65, 0.70, 0.75, 0.80]:
    req_se = (tgt-0.5)/(za+zb)
    lo, hi = 10, 5000
    while hi-lo > 1:
        m = (lo+hi)//2
        a, b = max(2,int(m*0.34)), max(2,m-int(m*0.34))
        if se(tgt,a,b) <= req_se: hi = m
        else: lo = m
    print(f'  AUROC {tgt:.2f} → N ≥ {hi}')

pd.DataFrame([{
    'n_dep':n1,'n_nondep':n2,'alpha':alpha,'power':power,
    'mde_auroc':round(auc,4),'observed_auroc':0.634,
    'underpowered_by':round(auc-0.634,4)
}]).to_csv(RES/'depression_power_analysis.csv', index=False)
print(f'\nSaved -> depression_power_analysis.csv')
