# Photo and cut-out from the SAME square crop: the object keeps its position and size; only the background is replaced by grey 130.
# Mask = the picked exemplar's SAM "best" mask (work/mask, copied to stimuli/masks so this step reruns without re-segmenting).
import os, shutil, numpy as np, pandas as pd, cv2
from PIL import Image, ImageOps
from config import STIM, WORK, SIZE as C, GREY, picks, exemplar_path

for f in ('photo', 'cutout'): os.makedirs(f'{STIM}/images/{f}', exist_ok=True)
rows = []
for w, s in picks().items():
    seg = f'{WORK}/mask/{w}/{s}_best.png'
    if os.path.exists(seg): shutil.copy(seg, f'{STIM}/masks/{w}.png')
    im = Image.open(exemplar_path(w, s)).convert('RGB'); m = Image.open(f'{STIM}/masks/{w}.png').convert('L')
    ph = np.array(ImageOps.fit(im, (C, C), Image.LANCZOS), np.float32)            # centre square crop, identical for image and mask
    mk = np.array(ImageOps.fit(m, (C, C), Image.BILINEAR), np.float32) / 255
    a = cv2.GaussianBlur(mk, (3, 3), 0.7)[..., None]
    cut = a * ph + (1 - a) * GREY
    Image.fromarray(ph.round().astype(np.uint8)).save(f'{STIM}/images/photo/{w}.png'); Image.fromarray(cut.round().astype(np.uint8)).save(f'{STIM}/images/cutout/{w}.png')
    b = mk > .5; ys, xs = np.nonzero(b)
    rows.append(dict(word=w, image=s, src_w=im.size[0], src_h=im.size[1], obj_area=round(b.mean(), 3),
                     touches_edge=round(float(np.mean(np.r_[b[0], b[-1], b[:, 0], b[:, -1]])), 3), cx=round(xs.mean() / C, 2), cy=round(ys.mean() / C, 2)))
st = pd.DataFrame(rows); st.to_csv(f'{STIM}/stim_stats.csv', index=False); print(st.to_string(index=False))
