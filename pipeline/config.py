# Paths shared by every pipeline script. Override the external locations with environment variables.
import os
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STIM = f'{REPO}/stimuli'
OUT = f'{REPO}/pipeline/output'
FIG = f'{REPO}/docs/figures'
WORK = os.environ.get('VF_WORK', f'{REPO}/pipeline/work')     # big intermediates (exemplar copies, all candidate masks); gitignored

# External inputs (read only)
VISVOCAB = os.environ.get('VISVOCAB', os.path.expanduser('~/Documents/GitHub/visvocab'))      # CHS processed data + CHS images
THINGS = os.environ.get('THINGS', '/Volumes/vislearnlab/data/THINGS-dataset/object_images')    # Polygon (needs UCSD VPN)
CHS_TRIALS = f'{VISVOCAB}/data/main/processed_data/level-trials_data.csv'
CHS_IMG = f'{VISVOCAB}/stimuli/lookit/exp1/img'

THINGS_FOLDER = {'chicken': 'chicken2'}        # THINGS concept folder when it differs from the word
GREY = 130                                     # cut-out background, RGB
SIZE = 600                                     # stimulus side, px
GBVS_ARGS = dict(channels='DIO', salmapmaxsize=32, blurfrac=0.02)    # matches visvocab's MATLAB GBVS maps (r = .996)

for d in (OUT, FIG, WORK):
    os.makedirs(d, exist_ok=True)


def pairs():
    return pd.read_csv(f'{STIM}/pairs.csv')


def words():
    p = pairs()
    return [w for a, b in zip(p.word_a, p.word_b) for w in (a, b)]


def partner():
    p = pairs()
    return {**dict(zip(p.word_a, p.word_b)), **dict(zip(p.word_b, p.word_a))}


def picks():
    return pd.read_csv(f'{STIM}/picks.csv').dropna(subset=['pick']).set_index('word').pick


def exemplar_path(word, stem):
    """Local copy made by 01_fetch_things.py, falling back to Polygon."""
    local = f'{WORK}/things/{word}/{stem}.jpg'
    return local if os.path.exists(local) else f'{THINGS}/{THINGS_FOLDER.get(word, word)}/{stem}.jpg'


def chs_trials():
    """CHS trials with the visvocab main-analysis exclusions."""
    d = pd.read_csv(CHS_TRIALS)
    d = d[(d.trial_exclusion == 0) & (d.exclude_participant == 0) & (d.exclude_participant_insufficient_data == 0) & (d['SubjectInfo.subjID'] != 'VVI128')]
    d['pair'] = d['Trials.trialID'].str.split('-').str[1:3].str.join('-')
    d['kind'] = d['Trials.trialType'].str.split('-').str[0]
    return d
