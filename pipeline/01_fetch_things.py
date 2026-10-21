# Copy THINGS exemplars from Polygon into pipeline/work/things/{word}/: the first 5 per word, plus every exemplar
# of the words given on the command line (used when none of the first 5 cut out well, e.g. car).
#   python 01_fetch_things.py            # first 5 for all 32 words
#   python 01_fetch_things.py car kite   # also all exemplars of car and kite
import os, sys, glob, shutil
from config import THINGS, THINGS_FOLDER, WORK, words

extra = set(sys.argv[1:])
for w in words():
    src = sorted(glob.glob(f'{THINGS}/{THINGS_FOLDER.get(w, w)}/*.jpg'))
    if not src: sys.exit(f'no THINGS images for {w} under {THINGS} (VPN + Polygon mounted?)')
    os.makedirs(f'{WORK}/things/{w}', exist_ok=True)
    for f in (src if w in extra else src[:5]):
        dst = f'{WORK}/things/{w}/{os.path.basename(f)}'
        if not os.path.exists(dst): shutil.copy(f, dst)
    print(w, len(src if w in extra else src[:5]))
