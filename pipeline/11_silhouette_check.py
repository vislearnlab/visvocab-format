# Does each stylized image keep the photo's silhouette? Segment it with the same Grounding DINO + SAM ViT-H recipe as the
# photos and compare with the photo's mask on the same 600 px crop: IoU, and area ratio (stylized / photo).
# Line drawings and doodles are skipped: SAM on white-page line art does not give a comparable filled mask.
#   python 11_silhouette_check.py [illustration ...]
import warnings; warnings.filterwarnings('ignore')
import os, sys
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')    # SAM needs CPU fallback for one op; set before importing torch
import numpy as np, pandas as pd, torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor
from config import STIM, OUT, SIZE as C, words

styles = sys.argv[1:] or ['illustration']
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
dp = AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base'); dm = AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base').eval().to(DEV)
sp = SamProcessor.from_pretrained('facebook/sam-vit-huge'); sm = SamModel.from_pretrained('facebook/sam-vit-huge').eval().to(DEV)
rows = []
for style in styles:
    for w in words():
        im = Image.open(f'{STIM}/images/{style}/{w}.png').convert('RGB')
        with torch.no_grad():
            inp = dp(images=im, text=f'a {w}.', return_tensors='pt').to(DEV); out = dm(**inp)
        r = dp.post_process_grounded_object_detection(out, inp.input_ids, threshold=0.2, text_threshold=0.2, target_sizes=[(C, C)])[0]
        sc, bx = r['scores'].cpu().numpy(), r['boxes'].cpu().numpy()
        if len(sc): bx = bx[sc >= max(0.25, 0.5 * sc.max())]
        if not len(bx): bx = np.array([[2, 2, C - 2, C - 2]], np.float32)
        with torch.no_grad(): emb = sm.get_image_embeddings(sp(im, return_tensors='pt')['pixel_values'].to(DEV))
        m_st = np.zeros((C, C), bool)
        for b in bx:
            si = sp(im, input_boxes=[[b.tolist()]], return_tensors='pt')
            with torch.no_grad(): so = sm(image_embeddings=emb, input_boxes=si['input_boxes'].float().to(DEV), multimask_output=True)
            ms = sp.image_processor.post_process_masks(so.pred_masks.cpu(), si['original_sizes'], si['reshaped_input_sizes'])[0][0].numpy()
            m_st |= ms[so.iou_scores[0, 0].cpu().numpy().argmax()]
        m_ph = np.array(ImageOps.fit(Image.open(f'{STIM}/masks/{w}.png').convert('L'), (C, C), Image.BILINEAR)) > 127
        rows.append(dict(style=style, word=w, iou=(m_st & m_ph).sum() / (m_st | m_ph).sum(), area_ratio=m_st.sum() / m_ph.sum()))
        print(style, w, f"iou {rows[-1]['iou']:.2f}  area x{rows[-1]['area_ratio']:.2f}", flush=True)
d = pd.DataFrame(rows).round(3); d.to_csv(f'{OUT}/silhouette_check.csv', index=False)
print(d.groupby('style')[['iou', 'area_ratio']].describe().round(2).to_string())
