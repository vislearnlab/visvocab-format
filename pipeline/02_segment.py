# Grounding DINO (prompt "a {word}.") boxes -> SAM ViT-H masks for every exemplar fetched by 01_fetch_things.py.
# Two variants per image: best = SAM's highest-IoU mask, whole = its largest mask. Writes work/cut, work/mask, work/seg_stats.csv.
# Run with PYTORCH_ENABLE_MPS_FALLBACK=1 on Apple silicon.
import warnings; warnings.filterwarnings('ignore')
import os, glob, numpy as np, pandas as pd, torch, cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor
from config import WORK, GREY
SRC = f'{WORK}/things'; OUT = WORK
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
dp = AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base'); dm = AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base').eval().to(DEV)
sp = SamProcessor.from_pretrained('facebook/sam-vit-huge'); sm = SamModel.from_pretrained('facebook/sam-vit-huge').eval().to(DEV)
def clean(mask):
    H, W = mask.shape; m8 = mask.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m8)
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 0.005 * H * W]
    m8 = np.isin(lab, keep).astype(np.uint8)
    pad = np.pad(m8, 1); ff = pad.copy(); cv2.floodFill(ff, np.zeros((H + 4, W + 4), np.uint8), (0, 0), 1)
    return (pad | (1 - ff))[1:-1, 1:-1], len(keep)
rows = []; done = set(pd.read_csv(f'{OUT}/seg_stats.csv').image) if os.path.exists(f'{OUT}/seg_stats.csv') else set()
if done: rows = pd.read_csv(f'{OUT}/seg_stats.csv').to_dict('records')
files = sorted(glob.glob(f'{SRC}/*/*.jpg'))
for k, f in enumerate(files):
    w = os.path.basename(os.path.dirname(f)); stem = os.path.splitext(os.path.basename(f))[0]
    if stem in done: continue
    im = Image.open(f).convert('RGB'); W, H = im.size
    with torch.no_grad():
        inp = dp(images=im, text=f'a {w}.', return_tensors='pt').to(DEV); out = dm(**inp)
    r = dp.post_process_grounded_object_detection(out, inp.input_ids, threshold=0.2, text_threshold=0.2, target_sizes=[(H, W)])[0]
    sc = r['scores'].cpu().numpy(); bx = r['boxes'].cpu().numpy()
    if len(sc): keep = sc >= max(0.25, 0.5 * sc.max()); bx, sc = bx[keep], sc[keep]
    if not len(sc): bx, sc = np.array([[2, 2, W - 2, H - 2]], np.float32), np.array([0.])
    with torch.no_grad():
        emb = sm.get_image_embeddings(sp(im, return_tensors='pt')['pixel_values'].to(DEV))
    var = {'best': np.zeros((H, W), bool), 'whole': np.zeros((H, W), bool)}; ious = []
    for b in bx:
        si = sp(im, input_boxes=[[b.tolist()]], return_tensors='pt')
        with torch.no_grad():
            so = sm(image_embeddings=emb, input_boxes=si['input_boxes'].float().to(DEV), multimask_output=True)
        ms = sp.image_processor.post_process_masks(so.pred_masks.cpu(), si['original_sizes'], si['reshaped_input_sizes'])[0][0].numpy()
        iou = so.iou_scores[0, 0].cpu().numpy(); ious.append(iou.max())
        var['best'] |= ms[iou.argmax()]; var['whole'] |= ms[np.argmax([m.sum() for m in ms])]
    os.makedirs(f'{OUT}/cut/{w}', exist_ok=True); os.makedirs(f'{OUT}/mask/{w}', exist_ok=True)
    row = dict(word=w, image=stem, W=W, H=H, n_boxes=len(bx), dino=float(sc.max()), sam_iou=float(np.mean(ious)))
    for v, m in var.items():
        m8, ncomp = clean(m)
        alpha = cv2.GaussianBlur(m8.astype(np.float32), (3, 3), 0.7)[..., None]
        cut = (alpha * np.array(im, np.float32) + (1 - alpha) * GREY).round().astype(np.uint8)
        Image.fromarray(cut).save(f'{OUT}/cut/{w}/{stem}_{v}.png'); Image.fromarray(m8 * 255).save(f'{OUT}/mask/{w}/{stem}_{v}.png')
        row.update({f'area_{v}': float(m8.mean()), f'edge_{v}': float(np.mean(np.r_[m8[0], m8[-1], m8[:, 0], m8[:, -1]])), f'ncomp_{v}': ncomp})
    rows.append(row)
    if k % 20 == 0: pd.DataFrame(rows).to_csv(f'{OUT}/seg_stats.csv', index=False); print(k, len(files), w, stem, flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/seg_stats.csv', index=False); print('DONE', len(rows), flush=True)
