# GBVS (Python port, settings matched to the CHS MATLAB maps) on side-by-side pair displays, photo vs cut-out.
# Display: 1600x700 page in CSS gray (128), two 600px images with a 200px gap. Saliency = mean of the master map inside each image.
import numpy as np, pandas as pd
from multiprocessing import Pool
from PIL import Image
from gbvs_py import gbvs
from config import STIM, OUT, GBVS_ARGS, pairs

L, R, T = (100, 700), (900, 1500), (50, 650)
def one(job):
    p, fmt = job; a, b = p.split('-')
    page = np.full((700, 1600, 3), 128, np.uint8)
    page[T[0]:T[1], L[0]:L[1]] = np.array(Image.open(f'{STIM}/images/{fmt}/{a}.png')); page[T[0]:T[1], R[0]:R[1]] = np.array(Image.open(f'{STIM}/images/{fmt}/{b}.png'))
    s = gbvs(page, **GBVS_ARGS)['master_map_resized']
    sa, sb = s[T[0]:T[1], L[0]:L[1]].mean(), s[T[0]:T[1], R[0]:R[1]].mean()
    return dict(pair=p, fmt=fmt, sal_a=sa, sal_b=sb, imbalance=(sa - sb) / (sa + sb))
if __name__ == '__main__':
    with Pool(6) as pool: rows = pool.map(one, [(p, f) for p in pairs().pair for f in ['photo', 'cutout']])
    d = pd.DataFrame(rows); d.to_csv(f'{OUT}/pair_saliency.csv', index=False)
    w = d.pivot(index='pair', columns='fmt', values='imbalance'); w['abs_photo'] = w.photo.abs(); w['abs_cutout'] = w.cutout.abs()
    print(w.round(3).sort_values('abs_photo', ascending=False).to_string())
    print(f"mean |imbalance|: photo {w.abs_photo.mean():.3f}  cut-out {w.abs_cutout.mean():.3f};  pairs where cut-out is more balanced: {(w.abs_cutout < w.abs_photo).sum()}/{len(w)}")
