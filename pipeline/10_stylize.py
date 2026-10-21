# Stylized versions of the photo stimuli, each from the same 600 px crop so the object keeps its position and size:
#   illustration  picture-book art: SDXL img2img from the photo, canny ControlNet on the smoothed photo's edges
#   line          picture-book line art: SDXL from the edge map alone, then only thin dark strokes are kept
#   doodle        QuickDraw / kiddraw-style sketch: SDXL steered only by the object's outline, then strokes kept
# Each style also gets a cut-out: the stylized image inside the photo's object mask, grey 130 outside.
#
#   python 10_stylize.py illustration                      # every word, as recorded in stimuli/stylize_params.csv
#   python 10_stylize.py illustration --words snail --seeds 2 3 4 --cscale 0.9 --label "snail=garden snail with a single pair of eye stalks" --candidates
# --candidates writes to work/candidates/{style}/{word}_s{seed}.png for picking instead of overwriting the stimuli;
# record the chosen seed/cscale/label in stimuli/stylize_params.csv. SDXL runs on MPS (fp16), ~2.5 min per image.
import os, sys, time, argparse
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')    # set before importing torch
import numpy as np, pandas as pd, cv2, torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from config import STIM, WORK, GREY, SIZE as C, picks, words as all_words

BASE = 'stabilityai/stable-diffusion-xl-base-1.0'; CNET = 'diffusers/controlnet-canny-sdxl-1.0'; VAE = 'madebyollin/sdxl-vae-fp16-fix'
NEG = 'photo, photograph, photorealistic, realistic, 3d render, text, letters, watermark, signature, extra limbs, deformed, blurry'
STYLES = {
    'illustration': dict(txt2img=False, lineart=False, control='canny', cscale=.7, guide=9.0,
                         prompt="children's picture book illustration of a {word}, flat 2D storybook art, gouache and colored pencil, soft flat colors, simple shapes, clean outlines, hand-drawn",
                         neg=''),
    'line': dict(txt2img=True, lineart=True, control='canny', cscale=.8, guide=7.5,
                 prompt="coloring book page, black and white line art of a {word}, children's picture book, simple clean thick outlines, minimal abstract shapes, white background",
                 neg='color, colored, shading, grayscale, gradient, texture, crosshatching, '),
    'doodle': dict(txt2img=True, lineart=True, control='outline', cscale=.6, guide=7.5,
                   prompt="a quick crude doodle of a {word} drawn in a few seconds with a black marker, like Google Quick Draw, a few simple wobbly strokes, childlike sketch, white background",
                   neg='color, colored, shading, grayscale, gradient, texture, detailed, realistic, background scenery, '),
}
GEN, CANNY, STEPS, STRENGTH = 1024, (100, 200), 30, 1.0

ap = argparse.ArgumentParser()
ap.add_argument('style', choices=STYLES); ap.add_argument('--words', nargs='*'); ap.add_argument('--seeds', nargs='*', type=int)
ap.add_argument('--cscale', type=float); ap.add_argument('--label', action='append', default=[]); ap.add_argument('--candidates', action='store_true')
a = ap.parse_args(); S = STYLES[a.style]
params = pd.read_csv(f'{STIM}/stylize_params.csv').query('style == @a.style').set_index('word') if os.path.exists(f'{STIM}/stylize_params.csv') else pd.DataFrame()
labels = dict(l.split('=', 1) for l in a.label)
out = f'{WORK}/candidates/{a.style}' if a.candidates else f'{STIM}/images/{a.style}'
outc = f'{WORK}/candidates/{a.style}_cutout' if a.candidates else f'{STIM}/images/{a.style}_cutout'
os.makedirs(out, exist_ok=True); os.makedirs(outc, exist_ok=True)

dt = torch.float16
pipe = (StableDiffusionXLControlNetPipeline if S['txt2img'] else StableDiffusionXLControlNetImg2ImgPipeline).from_pretrained(
    BASE, controlnet=ControlNetModel.from_pretrained(CNET, torch_dtype=dt), vae=AutoencoderKL.from_pretrained(VAE, torch_dtype=dt),
    torch_dtype=dt, variant='fp16', use_safetensors=True).to('mps')
pipe.set_progress_bar_config(disable=True)
pk = picks()

def mask(w, size):
    return ImageOps.fit(Image.open(f'{STIM}/masks/{w}.png').convert('L'), (size, size), Image.BILINEAR)

for w in a.words or all_words():
    p = params.loc[w] if w in params.index else {}
    seeds = a.seeds or [int(p.get('seed', 1))]
    cscale = a.cscale or float(p.get('cscale', S['cscale']))
    label = labels.get(w) or (p.get('label') if isinstance(p.get('label'), str) else w)
    big = Image.open(f'{STIM}/images/photo/{w}.png').convert('RGB').resize((GEN, GEN), Image.LANCZOS)
    if S['control'] == 'canny':     # edge-preserving blur first so only object/scene contours survive, not photo texture
        ed = cv2.Canny(cv2.bilateralFilter(cv2.pyrMeanShiftFiltering(np.array(big), 15, 30), 9, 75, 75), *CANNY)
    else:                           # the object's silhouette only
        ed = np.zeros((GEN, GEN), np.uint8)
        cs, _ = cv2.findContours((np.array(mask(w, GEN)) > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ed, cs, -1, 255, 3)
    ctrl = Image.fromarray(np.stack([ed] * 3, -1))
    mk = cv2.GaussianBlur(np.array(mask(w, C), np.float32) / 255, (3, 3), 0.7)[..., None]
    for seed in seeds:
        t = time.time()
        kw = dict(image=ctrl) if S['txt2img'] else dict(image=big, control_image=ctrl, strength=STRENGTH)
        im = pipe(prompt=S['prompt'].format(word=label), negative_prompt=S['neg'] + NEG, **kw, controlnet_conditioning_scale=cscale,
                  num_inference_steps=STEPS, guidance_scale=S['guide'], generator=torch.Generator('cpu').manual_seed(seed)).images[0].resize((C, C), Image.LANCZOS)
        if S['lineart']:            # keep only thin dark strokes (black-hat + threshold), drop specks: pure black lines on white
            g = np.array(im.convert('L')); bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            n, lab, st, _ = cv2.connectedComponentsWithStats((bh > 18).astype(np.uint8))
            keep = np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 40])
            im = Image.fromarray(np.stack([np.where(keep, 25, 255).astype(np.uint8)] * 3, -1))
        name = f'{w}_s{seed}' if a.candidates else w
        im.save(f'{out}/{name}.png')
        Image.fromarray((mk * np.array(im, np.float32) + (1 - mk) * GREY).round().astype(np.uint8)).save(f'{outc}/{name}.png')
        print(a.style, name, f'cscale {cscale} label "{label}"', f'{time.time() - t:.0f}s', flush=True)
