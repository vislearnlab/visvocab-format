# All stimuli: one row per pair, [word A in each format]  [word B in each format], with the GBVS L/R imbalance per format.
# Formats are whatever subfolders of stimuli/images exist among photo, cutout, illustration, illustration_cutout.
import os, pandas as pd
from PIL import Image, ImageDraw
from config import STIM, OUT, FIG, picks, pairs

Tl = 230; I = f'{STIM}/images'
fmts = [f for f in ['photo', 'cutout', 'illustration', 'illustration_cutout'] if os.path.isdir(f'{I}/{f}')]
pk = picks(); P = pairs().pair.tolist(); nf = len(fmts)
st = pd.read_csv(f'{STIM}/stim_stats.csv').set_index('word'); sal = pd.read_csv(f'{OUT}/pair_saliency.csv').set_index(['pair', 'fmt']).imbalance
sh = Image.new('RGB', (2 * (nf * Tl) + 30, len(P) * (Tl + 40) + 10), (255, 255, 255)); dr = ImageDraw.Draw(sh)
for i, p in enumerate(P):
    y = i * (Tl + 40) + 5
    for k, w in enumerate(p.split('-')):
        x = k * (nf * Tl + 30)
        for j, f in enumerate(fmts):
            sh.paste(Image.open(f'{I}/{f}/{w}.png').convert('RGB').resize((Tl - 4, Tl - 4), Image.LANCZOS), (x + j * Tl, y + 20))
        edge = st.loc[w, 'touches_edge'] > .02
        dr.text((x, y + 4), f'{w} ({pk[w]})  area {st.loc[w, "obj_area"]:.2f}' + ('  EDGE' if edge else ''), fill=(200, 0, 0) if edge else (0, 0, 0))
    dr.text((4, y + Tl + 20), f'{p}:  GBVS saliency imbalance (left - right)/(left + right)  photo {sal[(p, "photo")]:+.2f}  cut-out {sal[(p, "cutout")]:+.2f}   columns: {", ".join(fmts)}', fill=(0, 0, 160))
    dr.line((nf * Tl + 13, y + 20, nf * Tl + 13, y + Tl + 16), fill=(200, 200, 200), width=2)
sh.save(f'{FIG}/all_stimuli.png'); print(f'{FIG}/all_stimuli.png', len(P), 'pairs,', fmts)
