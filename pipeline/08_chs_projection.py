# CHS observed baseline-corrected target looking per hard pair (with backgrounds), and a projection for the cut-out version.
# Fit: naming-level corrected looking ~ CLIP ViT-B/32 image similarity of the CHS photos (all 32 pairs x 2 namings; subject-averaged first).
# Projection: observed + slope * (sim(cut-out pair) - sim(photo pair)), both from the new stimuli, so the shift is the format change
# alone (not the exemplar swap). Keeps each pair's own item effects; only similarity changes.
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch, open_clip, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats
from config import OUT, FIG, CHS_IMG, chs_trials
CHS = CHS_IMG + '/'
d = chs_trials()
s = d.groupby(['pair', 'kind', 'Trials.targetImage', 'SubjectInfo.subjID']).corrected_target_looking.mean().reset_index()
nm = s.groupby(['pair', 'kind', 'Trials.targetImage']).corrected_target_looking.agg(['mean', 'sem', 'count']).reset_index()
m, _, pre = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai'); m.eval()
@torch.no_grad()
def e(p): v = m.encode_image(pre(Image.open(p).convert('RGB'))[None])[0]; return v / v.norm()
E = {}
for p in nm.pair.unique():
    for w in p.split('-'): E.setdefault(w, e(CHS + w + '.jpg'))
nm['sim_chs'] = [float(E[a] @ E[b]) for a, b in nm.pair.str.split('-')]
fit = stats.linregress(nm.sim_chs, nm['mean'])
fit_h = stats.linregress(nm[nm.kind == 'hard'].sim_chs, nm[nm.kind == 'hard']['mean'])
print(f'all 64 namings: slope={fit.slope:.3f} se={fit.stderr:.3f} p={fit.pvalue:.3f} r={fit.rvalue:.2f}')
ez = nm.groupby('kind').sim_chs.mean(); gap = .021   # within-child easy - hard corrected looking (n = 154, p = .046)
implied = -gap / (ez['hard'] - ez['easy'])
print(f'CLIP sim easy {ez["easy"]:.3f} hard {ez["hard"]:.3f}; implied slope if the easy/hard gap were all visual similarity: {implied:.3f}')
print(f'hard only (32): slope={fit_h.slope:.3f} se={fit_h.stderr:.3f} p={fit_h.pvalue:.3f} r={fit_h.rvalue:.2f}')
sim = pd.read_csv(f'{OUT}/pair_similarity.csv').set_index('pair')
hp = nm[nm.kind == 'hard'].copy()
pair = hp.groupby('pair').agg(obs=('mean', 'mean'), sim_chs=('sim_chs', 'first')).join(
    s[s.kind == 'hard'].groupby(['pair', 'SubjectInfo.subjID']).corrected_target_looking.mean().groupby('pair').sem().rename('se'))
pair['sim_cut'] = sim.cutout.reindex(pair.index); pair['dsim'] = (sim.cutout - sim.photo).reindex(pair.index)
pair['proj'] = pair.obs + fit.slope * pair.dsim
pair['proj_up'] = pair.obs + implied * pair.dsim
pair['proj_se'] = np.sqrt(pair.se ** 2 + (fit.stderr * pair.dsim) ** 2)
new = sim[~sim.index.isin(pair.index)]
pair.round(3).to_csv(f'{OUT}/chs_projection.csv'); print(pair.round(3).sort_values('obs', ascending=False).to_string())
# plot
kept = pair.dropna(subset=['sim_cut']).sort_values('obs', ascending=False)
dropped = pair[pair.sim_cut.isna()].sort_values('obs', ascending=False)
order = list(kept.index) + list(new.index) + list(dropped.index)
fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 6.5), gridspec_kw=dict(width_ratios=[1.5, 1]))
y = np.arange(len(order))[::-1]
for yi, p in zip(y, order):
    if p in pair.index:
        r = pair.loc[p]; c = '#555' if p in dropped.index else '#1f78b4'
        ax.errorbar(r.obs, yi + .12, xerr=1.96 * r.se, fmt='o', color=c, ms=6, capsize=0, lw=1)
        if p in kept.index:
            ax.errorbar(r.proj, yi - .12, xerr=1.96 * r.proj_se, fmt='o', mfc='white', color='#e66101', ms=6, capsize=0, lw=1)
            ax.plot([r.obs, r.proj_up], [yi - .3, yi - .3], color='#b2182b', lw=1); ax.plot(r.proj_up, yi - .3, marker='<' if r.proj_up < r.obs else '>', color='#b2182b', ms=5)
    else:
        ax.text(0.0, yi, '  new pair: no CHS data', va='center', fontsize=8, color='#999')
ax.axvline(0, color='k', lw=.6); ax.set_yticks(y); ax.set_yticklabels([p + (' (dropped)' if p in dropped.index else '') for p in order], fontsize=8.5)
ax.set_xlabel('Baseline-corrected target looking (mean ± 95% CI)')
ax.legend(handles=[plt.Line2D([], [], marker='o', ls='', color='#1f78b4', label='CHS observed: photos with background'),
                   plt.Line2D([], [], marker='o', ls='', mfc='white', color='#e66101', label='projected cut-out: CHS similarity slope (≈ flat)'),
                   plt.Line2D([], [], marker='<', color='#b2182b', label='upper bound: if easy/hard gap were all visual similarity'),
                   plt.Line2D([], [], marker='o', ls='', color='#555', label='CHS observed: pair dropped')], fontsize=8, loc='lower right', frameon=False)
ax.set_title('Hard pairs: CHS data and projected cut-out performance', fontsize=10.5)
for k, c in [('easy', '#a6dba0'), ('hard', '#1f78b4')]:
    g = nm[nm.kind == k]; bx.scatter(g.sim_chs, g['mean'], s=18, color=c, label=f'{k} namings (CHS photos)')
xs = np.linspace(nm.sim_chs.min(), max(nm.sim_chs.max(), np.nanmax(pair.sim_cut)), 50)
bx.plot(xs, fit.intercept + fit.slope * xs, color='k', lw=1, label=f'fit: slope {fit.slope:.2f} ± {fit.stderr:.2f} (p = {fit.pvalue:.2f})')
bx.axhline(0, color='k', lw=.4, ls=':'); bx.set_xlabel('CLIP ViT-B/32 image similarity (CHS photos)'); bx.set_ylabel('Corrected target looking (naming mean)')
bx.legend(fontsize=8, frameon=False); bx.set_title('The similarity → looking relation used for projection', fontsize=10.5)
fig.tight_layout(); fig.savefig(f'{FIG}/chs_projection.png', dpi=150); print(f'{FIG}/chs_projection.png')
