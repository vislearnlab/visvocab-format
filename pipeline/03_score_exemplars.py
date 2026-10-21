# Score every exemplar's cut-out: whole object (not touching frame), single object, size, CLIP recognition (photo + cut-out).
# Flags which exemplar the CHS study used. Writes output/exemplar_scores.csv and one pick sheet per word (work/sheets/{word}.png).
# Only the first 5 exemplars per word are scored; picks for words where all 5 failed came from inspecting work/cut directly.
import warnings; warnings.filterwarnings('ignore')
import os, glob, numpy as np, pandas as pd, torch, open_clip
from PIL import Image, ImageDraw
from config import WORK, OUT, REPO, CHS_IMG
D = WORK; SRC = f'{WORK}/things'; CHS = CHS_IMG + '/'
st = pd.read_csv(f'{D}/seg_stats.csv').sort_values('image').groupby('word').head(5).reset_index(drop=True)   # first 5 per word
vocab = pd.read_csv(f'{REPO}/pipeline/input/things_concepts.tsv', sep='\t').Word.str.strip().drop_duplicates().tolist()
m, _, pre = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k'); m.eval(); tok = open_clip.get_tokenizer('ViT-L-14')
with torch.no_grad():
    T = torch.cat([m.encode_text(tok([f'a photo of a {v}.' for v in vocab[i:i + 256]])) for i in range(0, len(vocab), 256)]); T /= T.norm(dim=-1, keepdim=True)
    def enc(paths):
        E = torch.cat([m.encode_image(torch.stack([pre(Image.open(p).convert('RGB')) for p in paths[i:i + 32]])) for i in range(0, len(paths), 32)])
        return E / E.norm(dim=-1, keepdim=True)
    ph = enc([f'{SRC}/{r.word}/{r.image}.jpg' for r in st.itertuples()]); cu = enc([f'{D}/cut/{r.word}/{r.image}_best.png' for r in st.itertuples()])
    chs = {w: enc([CHS + w + '.jpg'])[0] for w in st.word.unique() if os.path.exists(CHS + w + '.jpg')}   # new words have no CHS image
for nm, E in [('photo', ph), ('cut', cu)]:
    P = (100 * E @ T.T).softmax(-1)
    st[f'p_{nm}'] = [float(P[i, vocab.index(w)]) for i, w in enumerate(st.word)]
    st[f'rank_{nm}'] = [int((P[i] > P[i, vocab.index(w)]).sum()) + 1 for i, w in enumerate(st.word)]
st['sim_to_chs'] = [float(ph[i] @ chs[w]) if w in chs else np.nan for i, w in enumerate(st.word)]
st['is_chs'] = st.groupby('word').sim_to_chs.transform('max').eq(st.sim_to_chs) & (st.sim_to_chs > .97)
st['whole'] = st.edge_best < .02
st['single'] = st.ncomp_best == 1
st['size_ok'] = st.area_best.between(.08, .6)
st['score'] = st.whole * 2 + st.single + st.size_ok + st.p_cut.clip(0, 1) + st.sam_iou.clip(0, 1) - (st.area_whole - st.area_best).clip(0, None) * 2
st = st.sort_values(['word', 'score'], ascending=[True, False]); st.round(3).to_csv(f'{OUT}/exemplar_scores.csv', index=False)
os.makedirs(f'{D}/sheets', exist_ok=True); Tl = 200
for w, g in st.groupby('word', sort=False):
    n = len(g); cols = 5; nr = -(-n // cols)
    sh = Image.new('RGB', (cols * Tl, nr * (2 * Tl + 30)), (255, 255, 255))
    for k, r in enumerate(g.itertuples()):
        x, y = (k % cols) * Tl, (k // cols) * (2 * Tl + 30)
        for j, p in enumerate([f'{SRC}/{w}/{r.image}.jpg', f'{D}/cut/{w}/{r.image}_best.png']):
            im = Image.open(p).convert('RGB'); im.thumbnail((Tl - 4, Tl - 4)); sh.paste(im, (x + 2, y + 30 + j * Tl))
        dr = ImageDraw.Draw(sh)
        tag = f'#{k + 1} {r.image}' + ('  [CHS]' if r.is_chs else '')
        flags = ('' if r.whole else 'EDGE ') + ('' if r.single else f'{r.ncomp_best}obj ') + ('' if r.size_ok else 'SIZE ') + f'p={r.p_cut:.2f}'
        dr.text((x + 3, y + 2), tag, fill=(200, 0, 0) if r.is_chs else (0, 0, 0)); dr.text((x + 3, y + 15), flags, fill=(0, 0, 160))
    sh.save(f'{D}/sheets/{w}.png')
print(st.groupby('word').agg(n=('image', 'size'), n_whole=('whole', 'sum'), n_clean=('score', lambda s: (s > 4.3).sum()), chs_rank=('is_chs', lambda s: (np.flatnonzero(s.values) + 1).tolist())).to_string())
