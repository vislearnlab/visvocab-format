# Contact sheet of every format, arranged by trial: one row per pair (the two images shown together on a trial),
# word A's formats then word B's, rows grouped by domain. Formats = whichever image folders are complete.
#   -> docs/figures/format_sheet.png
import os
from PIL import Image, ImageDraw, ImageFont
from config import STIM, FIG, pairs

FMTS = [('photo', 'photo'), ('cutout', 'cut-out'), ('illustration', 'illustration'), ('illustration_cutout', 'illus. cut-out'),
        ('line', 'line drawing'), ('line_cutout', 'line cut-out'), ('doodle', 'doodle'), ('doodle_cutout', 'doodle cut-out')]
I = f'{STIM}/images'
FMTS = [(d, n) for d, n in FMTS if os.path.isdir(f'{I}/{d}') and len(os.listdir(f'{I}/{d}')) >= 32]
T, GAP, LAB, HEAD = 150, 24, 20, 30; nf = len(FMTS)
order = ['animal', 'artifact', 'nature', 'plant']
P = pairs(); P['k'] = P.domain.map(order.index); P = P.sort_values(['k', 'pair']).reset_index(drop=True)
W = 2 * nf * T + GAP; H = HEAD + len(P) * (T + LAB) + P.domain.nunique() * HEAD
sh = Image.new('RGB', (W, H), 'white'); dr = ImageDraw.Draw(sh)
try: f16, f13 = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 18), ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 13)
except OSError: f16 = f13 = ImageFont.load_default()
for side in range(2):                                  # column headers
    for j, (_, n) in enumerate(FMTS): dr.text((side * (nf * T + GAP) + j * T + 4, 8), n, fill=(90, 90, 90), font=f13)
y = HEAD; cur = None
for r in P.itertuples():
    if r.domain != cur:
        cur = r.domain; dr.text((4, y + 6), f'{cur} pairs', fill=(150, 60, 0), font=f16); y += HEAD
    for side, w in enumerate((r.word_a, r.word_b)):
        x0 = side * (nf * T + GAP)
        for j, (d, _) in enumerate(FMTS):
            sh.paste(Image.open(f'{I}/{d}/{w}.png').convert('RGB').resize((T - 2, T - 2), Image.LANCZOS), (x0 + j * T, y))
        dr.text((x0 + 3, y + T + 1), w, fill='black', font=f13)
    dr.line((nf * T + GAP // 2, y, nf * T + GAP // 2, y + T), fill=(180, 180, 180), width=2)
    y += T + LAB
sh.save(f'{FIG}/format_sheet.png'); print(f'{FIG}/format_sheet.png', sh.size, [n for _, n in FMTS])
