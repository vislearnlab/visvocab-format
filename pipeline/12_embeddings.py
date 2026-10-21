# Image embeddings for every stimulus in every format (+ the CHS photos, + earlier illustration rounds), two models:
#   dinov3  facebook/dinov3-vitb16-pretrain-lvd1689m, pooler output (CLS after final norm), 224x224
#   clip    OpenAI CLIP ViT-B/32 (the CHS model), image embedding
# Writes output/embeddings_{model}.npz with arrays `emb` (L2-normalised) and `key` ("{version}|{word}").
# Illustration history: stimuli/images/history/round{k}/{illustration,illustration_cutout}/ hold only the words that
# differed from later rounds; a word missing from round k is taken from the next round that has it, else the final set.
import warnings; warnings.filterwarnings('ignore')
import os, glob, numpy as np, torch, open_clip
from PIL import Image
from transformers import AutoModel
from config import STIM, OUT, CHS_IMG, words

FORMATS = ['photo', 'cutout', 'illustration', 'illustration_cutout', 'line', 'line_cutout', 'doodle', 'doodle_cutout']
I = f'{STIM}/images'; W = words()
paths = {}
for f in FORMATS:
    for w in W:
        if os.path.exists(f'{I}/{f}/{w}.png'): paths[f'{f}|{w}'] = f'{I}/{f}/{w}.png'
for w in W:
    if os.path.exists(f'{CHS_IMG}/{w}.jpg'): paths[f'CHS photo|{w}'] = f'{CHS_IMG}/{w}.jpg'
rounds = sorted(int(d.split('round')[-1]) for d in glob.glob(f'{I}/history/round*'))
for k in rounds:
    for f in ['illustration', 'illustration_cutout']:
        for w in W:
            src = next((f'{I}/history/round{j}/{f}/{w}.png' for j in rounds if j >= k and os.path.exists(f'{I}/history/round{j}/{f}/{w}.png')), f'{I}/{f}/{w}.png')
            paths[f'{f} round {k}|{w}'] = src
keys = sorted(paths)
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'

# DINOv3: its image processor is fast-only and needs torch >= 2.3, so preprocess by hand
# (preprocessor_config.json: resize 224x224 bilinear, rescale 1/255, ImageNet mean/std)
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
dino = AutoModel.from_pretrained('facebook/dinov3-vitb16-pretrain-lvd1689m').eval().to(DEV)
@torch.no_grad()
def e_dino(p):
    x = (np.array(Image.open(p).convert('RGB').resize((224, 224), Image.BILINEAR), np.float32) / 255 - MEAN) / STD
    return dino(pixel_values=torch.from_numpy(x.transpose(2, 0, 1))[None].to(DEV)).pooler_output[0].float().cpu().numpy()
clip, _, pre = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai'); clip.eval()
@torch.no_grad()
def e_clip(p): return clip.encode_image(pre(Image.open(p).convert('RGB'))[None])[0].numpy()

for name, fn in [('dinov3', e_dino), ('clip', e_clip)]:
    E = np.stack([fn(paths[k]) for k in keys]); E /= np.linalg.norm(E, axis=1, keepdims=True)
    np.savez(f'{OUT}/embeddings_{name}.npz', emb=E.astype(np.float32), key=np.array(keys))
    print(name, E.shape)
