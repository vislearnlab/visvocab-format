# Per-item moderators for the format effect, computed from the stimuli only (before any data):
#  bg_diag   CLIP 2AFC on the BACKGROUND ALONE (grey bounding box over object + 15px margin, so no silhouette survives): p(own word vs partner word). 0.5 = background uninformative.
#  rec_photo / rec_cut  CLIP 2AFC on the full photo / the cut-out; rec_drop = photo - cut-out (how much recognition leans on the background).
#  bg_sal    share of the photo's GBVS saliency mass that falls on the background (outside the object mask).
#  context_discrim_pair = bg_diag(A) + bg_diag(B) - 1, the same value for both namings of a pair: how well the two backgrounds,
#            both on screen, point to the right words. ~0 = backgrounds can't separate the pair (e.g. cow-horse, cloud-tree).
#  CLIP = average of ViT-B/32 (OpenAI) and ViT-L/14 (LAION-2B) logits-based 2AFC.
import warnings; warnings.filterwarnings('ignore')
import os, numpy as np, pandas as pd, torch, open_clip, cv2
from PIL import Image
from gbvs_py import gbvs
from config import STIM, OUT, GREY, GBVS_ARGS, words as _words, partner as _partner
words, partner = _words(), _partner(); I = f'{STIM}/images'
os.makedirs(f'{I}/background_only', exist_ok=True); bg_sal = {}
for w in words:
    ph = np.array(Image.open(f'{I}/photo/{w}.png')); m = np.array(Image.open(f'{I}/cutout/{w}.png'))
    mask = (np.abs(ph.astype(int) - m.astype(int)).sum(-1) < 3) & (np.abs(m.astype(int) - GREY).sum(-1) > 3)   # unchanged, non-grey pixels = object
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    obj = cv2.dilate(mask.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
    ys, xs = np.nonzero(obj); bgo = ph.copy(); bgo[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = GREY   # shape-free occluder: grey bounding box
    Image.fromarray(bgo).save(f'{I}/background_only/{w}.png')
    s = gbvs(ph, **GBVS_ARGS)['master_map_resized']
    bg_sal[w] = float(s[~mask].sum() / s.sum())
rows = {w: dict(word=w, partner=partner[w], bg_sal=bg_sal[w]) for w in words}
for arch, tag in [('ViT-B-32', 'openai'), ('ViT-L-14', 'laion2b_s32b_b82k')]:
    m, _, pre = open_clip.create_model_and_transforms(arch, pretrained=tag); m.eval(); tok = open_clip.get_tokenizer(arch)
    with torch.no_grad():
        T = m.encode_text(tok([f'a photo of a {w}.' for w in words])); T /= T.norm(dim=-1, keepdim=True); ti = {w: i for i, w in enumerate(words)}
        for fmt, key in [('photo', 'rec_photo'), ('cutout', 'rec_cut'), ('background_only', 'bg_diag')]:
            for w in words:
                e = m.encode_image(pre(Image.open(f'{I}/{fmt}/{w}.png').convert('RGB'))[None])[0]; e /= e.norm()
                lg = 100 * torch.stack([e @ T[ti[w]], e @ T[ti[partner[w]]]])
                rows[w][f'{key}_{arch}'] = float(lg.softmax(0)[0])
d = pd.DataFrame(rows.values())
for k in ['rec_photo', 'rec_cut', 'bg_diag']: d[k] = d[[c for c in d if c.startswith(k + '_')]].mean(1)
d['rec_drop'] = d.rec_photo - d.rec_cut
d['context_discrim_pair'] = d.bg_diag + d.word.map(dict(zip(d.partner, d.bg_diag))) - 1
d = d[['word', 'partner', 'bg_diag', 'rec_photo', 'rec_cut', 'rec_drop', 'bg_sal'] + [c for c in d if 'ViT' in c] + ['context_discrim_pair']]
d.round(3).to_csv(f'{OUT}/item_moderators.csv', index=False)
print(d[['word', 'partner', 'bg_diag', 'rec_photo', 'rec_cut', 'rec_drop', 'bg_sal']].sort_values('bg_diag', ascending=False).round(2).to_string(index=False))
print('\nr(bg_diag, bg_sal) = %.2f;  r(bg_diag, rec_drop) = %.2f' % (np.corrcoef(d.bg_diag, d.bg_sal)[0, 1], np.corrcoef(d.bg_diag, d.rec_drop)[0, 1]))
