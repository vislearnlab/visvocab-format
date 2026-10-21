# visvocab-format

Stimuli, item scores and the CHS jsPsych experiment for the **visual format study**: a rerun of the
[`visvocab`](https://github.com/vislearnlab/visvocab) / [`visual-precision`](https://github.com/vislearnlab/visual-precision)
Children Helping Science (CHS) looking-while-listening study with hard pairs only. Each pair is shown once as full
photos and once as the *same* photos with the background replaced by grey, to test whether backgrounds help
(scene context) or hurt (salient clutter) word recognition.

The design rationale, item-level predictions and analysis plan are in `docs/report.md`, an export of the shared
"Visual format study: photos vs cut-outs" doc.

## What's here

```
stimuli/
  pairs.csv                 16 hard pairs (13 from CHS + cow-horse, butterfly-ladybug, rabbit-hamster)
  picks.csv                 the THINGS exemplar used for each of the 32 words, with the reason for each pick
  images/photo/{word}.png   600x600 centre crop of the THINGS exemplar
  images/cutout/{word}.png  the same crop, background -> RGB(130,130,130); object position and size unchanged
  images/background_only/   photo with the object's bounding box greyed out (used only to score background diagnosticity)
  images/calib_duck.jpg     calibration / get-ready image (copy of the hosted one)
  masks/{word}.png          SAM mask of the picked exemplar, full source resolution
  stim_stats.csv            object area, edge contact, centroid per stimulus
  audio/{mp3,ogg}/          {find,where,see,look_at}_{word} for all 32 words + get_ready, great_job, ready_to_go, pinwheel
pipeline/                   numbered scripts; run from pipeline/ (see below)
  output/                   CHS hard-pair performance, exemplar scores, pair similarity, saliency, item moderators, projection
experiment/
  chs_jspsych/experiment.js paste into "jsPsych Experiment Code" on the CHS study page
  chs_jspsych/preview.html  runs experiment.js locally with stand-ins for the CHS recording/consent/survey plugins
  lookit_reference/         the original CHS Lookit protocol generators (sample1, sample2), unmodified
docs/
  report.md                 the study write-up
  figures/                  all_stimuli.png, similarity_photo_vs_cutout.png, chs_projection.png
```

## Experiment

`experiment/chs_jspsych/experiment.js` is a jsPsych 8 port of `lookit_reference/experiment_sample2.js`. It uses only
packages the CHS jsPsych runner loads by default (html-button-response, video-button-response, preload,
fullscreen) plus `chsRecord` (video config, video consent, per-trial recording) and `chsSurvey` (exit survey).

- **Kept from CHS:** gray page, 7.2 s trials with the audio starting at image onset (each file opens with ~2 s of silence),
  the four carrier phrases cycled once per block of four, an attention-getter video after every fourth trial,
  4-point calibration (centre, left, right, centre), space bar to pause (the paused trial is skipped, as with
  `frameOffsetAfterPause: 1`), the US-eligibility question, consent text, CDI link by age, exit survey.
- **New:** 32 trials = 16 pairs x {photo, cut-out}. A pair's two presentations fall in opposite halves, the second
  names the other word, and they never run back to back. Four lists: list L puts pair *i* in cell (*i* + L) mod 4, which
  fixes the first format and the first-named word, so across lists every word is named in every format equally often
  and each list has 8 photo-first and 8 cut-out-first pairs. The list is random per session (or `?list=0..3`) and is
  saved on every row of data, with target side, carrier, image onset and audio onset times.
- **Needs IRB review:** the intro summary and debrief now describe the photo vs grey-background manipulation. Consent
  text is unchanged from the CHS study.
- **Before launch:** upload `stimuli/images/photo` and `stimuli/images/cutout` to
  `https://ucsdlearninglabs.org/stimuli/visvocab-format/{photo,cutout}/` (that is `IMG_BASE` in experiment.js). The
  audio, videos and calibration image are already hosted under `https://ucsdlearninglabs.org/stimuli/`.

Local check:

```bash
python3 -m http.server 8000          # from the repo root
# open http://localhost:8000/experiment/chs_jspsych/preview.html?list=0&skip=1
```

`skip=1` starts at the looking task. Never use it on a live CHS link, because it skips consent.

## Pipeline

```bash
pip install -r requirements.txt
cd pipeline
python 00_chs_hard_pairs.py          # CHS baseline-corrected looking per hard pair x named word (reads visvocab)
python 01_fetch_things.py car        # copy THINGS exemplars from Polygon (first 5 per word; all of car)
PYTORCH_ENABLE_MPS_FALLBACK=1 python 02_segment.py   # Grounding DINO + SAM ViT-H, best/whole masks
python 03_score_exemplars.py         # whole / single / size / CLIP checks + per-word pick sheets
python 04_make_stimuli.py            # photo + cut-out from the same crop (uses stimuli/picks.csv)
python 05_similarity.py              # CLIP ViT-B/32 visual (photo, cut-out) and semantic similarity per pair
python 06_saliency.py                # GBVS left/right imbalance of each pair display, per format
python 07_item_moderators.py         # background diagnosticity, context discriminability, background saliency share
python 08_chs_projection.py          # CHS observed looking + similarity-based cut-out projection
python 09_stimulus_sheet.py          # docs/figures/all_stimuli.png
```

Steps 04-09 run from what is in the repo plus the source photos. Without Polygon, `04` needs `pipeline/work/things`.
The masks are in `stimuli/masks`, so nothing needs re-segmenting. Steps 00 and 08 read the CHS processed data from
a local `visvocab` checkout (`VISVOCAB=...` to point elsewhere). `02` and `03` regenerate the exemplar search.
Exemplars were picked by eye from the sheets, and the reasons are in `picks.csv`.

Rerunning 04 reproduces the committed stimuli pixel for pixel.

## Notes

- **Grey levels.** The cut-out background is RGB 130 and the page is CSS `gray` (128).
- **CLIP is at ceiling for recognition.** Photo and cut-out alike are named correctly, so `rec_*` in
  `item_moderators.csv` is not a usable recognition proxy.
- **Context discriminability** (`context_discrim_pair` = bg_diag(A) + bg_diag(B) − 1) is a pair-level score, not a
  per-naming one: both backgrounds are on screen in every trial.
- **GBVS** is a Python port (`gbvs_py.py`) matched to visvocab's MATLAB maps (r = .996) with channels `DIO`,
  salmapmaxsize 32, blurfrac .02.
- **Provenance.** CHS images, audio and processed data come from `visvocab`. THINGS exemplars come from
  `Polygon:data/THINGS-dataset/object_images` (chicken = `chicken2`). `pipeline/input/things_concepts.tsv` comes from
  `visvocab-eyetracking/norms`.
