# Visual (CLIP ViT-B/32 image cosine) similarity of each pair as photo vs as cut-out, both from the final 600 px stimuli,
# and semantic similarity (CLIP text cosine, "a <word>"). ViT-B/32 OpenAI is the model the CHS analyses used.
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch, open_clip, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
from config import STIM, OUT, FIG, pairs

m, _, pre = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai'); m.eval(); tok = open_clip.get_tokenizer('ViT-B-32')
@torch.no_grad()
def img(p): e = m.encode_image(pre(Image.open(p).convert('RGB'))[None])[0]; return e / e.norm()
@torch.no_grad()
def txt(w): e = m.encode_text(tok([f'a {w}']))[0]; return e / e.norm()
P = pairs()
d = pd.DataFrame([dict(pair=r.pair, domain=r.domain, text=float(txt(r.word_a) @ txt(r.word_b)),
                       photo=float(img(f'{STIM}/images/photo/{r.word_a}.png') @ img(f'{STIM}/images/photo/{r.word_b}.png')),
                       cutout=float(img(f'{STIM}/images/cutout/{r.word_a}.png') @ img(f'{STIM}/images/cutout/{r.word_b}.png'))) for r in P.itertuples()])
d.round(3).to_csv(f'{OUT}/pair_similarity.csv', index=False); print(d.round(3).to_string(index=False))

d = d.sort_values('cutout').reset_index(drop=True); art = set(d.pair[d.domain != 'animal'])
fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 6.8), gridspec_kw=dict(width_ratios=[1, 1.15]))
cmap = plt.cm.viridis; norm = plt.Normalize(d.text.min(), d.text.max())
lab = d.cutout.values.copy(); gap = .0095            # spread cut-out labels so none overlap (d is sorted by cutout)
for i in range(1, len(lab)): lab[i] = max(lab[i], lab[i - 1] + gap)
lab -= max(0, lab[-1] - d.cutout.max() - .01)
for i in range(len(lab) - 2, -1, -1): lab[i] = min(lab[i], lab[i + 1] - gap)
for r, ly in zip(d.itertuples(), lab):           # A: slope chart, photo -> cut-out
    c = cmap(norm(r.text)); ax.plot([0, 1], [r.photo, r.cutout], color=c, lw=1.6, marker='o', ms=5)
    ax.plot([1.01, 1.06], [r.cutout, ly], color='#bbb', lw=.6)
    ax.text(1.07, ly, r.pair, va='center', fontsize=8, color='#333' if r.pair not in art else '#b35806')
ax.set_xticks([0, 1]); ax.set_xticklabels(['photo\n(with background)', 'cut-out\n(grey background)']); ax.set_xlim(-.15, 1.55)
ax.set_ylabel('Visual similarity of the pair (CLIP ViT-B/32 image cosine)'); ax.grid(axis='y', alpha=.25)
ax.set_title('A. Visual similarity with and without the background', fontsize=10.5, loc='left')
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); cb = fig.colorbar(sm, ax=ax, fraction=.04, pad=.01); cb.set_label('semantic similarity (CLIP text)', fontsize=8)
for r in d.itertuples():           # B: semantic (x) vs visual (y), arrow photo -> cut-out
    c = '#b35806' if r.pair in art else '#542788'
    bx.annotate('', xy=(r.text, r.cutout), xytext=(r.text, r.photo), arrowprops=dict(arrowstyle='->', color=c, lw=1.2, alpha=.8))
    bx.scatter(r.text, r.photo, s=28, facecolors='white', edgecolors=c, zorder=3); bx.scatter(r.text, r.cutout, s=28, color=c, zorder=3)
    bx.annotate(r.pair, (r.text, r.cutout), xytext=(5, 2), textcoords='offset points', fontsize=7.5, color=c)
bx.set_xlabel('Semantic similarity (CLIP ViT-B/32 text cosine, "a <word>")'); bx.set_ylabel('Visual similarity (CLIP image cosine)')
bx.set_title('B. Visual vs semantic: open = photo, filled = cut-out', fontsize=10.5, loc='left'); bx.grid(alpha=.25)
bx.legend(handles=[plt.Line2D([], [], marker='o', ls='', color='#542788', label='animal pair'), plt.Line2D([], [], marker='o', ls='', color='#b35806', label='artifact / nature / plant pair')], fontsize=8, frameon=False, loc='lower right')
fig.tight_layout(); fig.savefig(f'{FIG}/similarity_photo_vs_cutout.png', dpi=160)
print('mean visual: photo %.3f -> cut-out %.3f; pairs that increase: %d/%d; r(visual photo, semantic)=%.2f, r(visual cut-out, semantic)=%.2f' % (
    d.photo.mean(), d.cutout.mean(), (d.cutout > d.photo).sum(), len(d), np.corrcoef(d.photo, d.text)[0, 1], np.corrcoef(d.cutout, d.text)[0, 1]))
