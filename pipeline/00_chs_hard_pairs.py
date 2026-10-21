# Per CHS hard pair x named word: baseline-corrected target looking (main-analysis exclusions), subject-averaged.
# Used to drop pairs at floor and as the "with background" reference for the new study.
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import stats
from config import OUT, chs_trials

d = chs_trials()
d = d[d.kind == 'hard']
d['named'] = d['Trials.targetImage']
s = d.groupby(['pair', 'named', 'SubjectInfo.subjID']).agg(c=('corrected_target_looking', 'mean'), crit=('mean_target_looking_critical_window', 'mean'), base=('mean_target_looking_baseline_window', 'mean')).reset_index()
def summ(g):
    t = stats.ttest_1samp(g.c, 0)
    return pd.Series(dict(n=len(g), corrected=g.c.mean(), se=g.c.std() / np.sqrt(len(g)), p=t.pvalue, crit=g.crit.mean(), base=g.base.mean()))
r = s.groupby(['pair', 'named']).apply(summ).reset_index()
r['lo95'] = r.corrected - 1.96 * r.se
pr = s.groupby(['pair', 'SubjectInfo.subjID']).c.mean().groupby('pair').agg(['mean', 'sem', 'count']).rename(columns=lambda c: 'pair_' + c)
r = r.merge(pr, left_on='pair', right_index=True)
r['section'] = r.pair.map(d.groupby('pair').section.first())
r.round(3).to_csv(f'{OUT}/chs_hard_pair_performance.csv', index=False)
print(r.sort_values(['pair_mean', 'named'], ascending=[False, True]).round(3).to_string(index=False))
