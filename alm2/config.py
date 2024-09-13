from os.path import join, expanduser, dirname
import os
path_to_repo = dirname(dirname(os.path.abspath(__file__)))


if 'ALM_DIR' in os.environ:
    ALM_DIR = os.environ['ALM_DIR']
else:
    ALM_DIR = path_to_repo
DATA_DIR_ROOT = join(ALM_DIR, 'data')
SAVE_DIR_ROOT = join(ALM_DIR, 'results')

# individual datasets...
BABYLM_DATA_DIR = join(DATA_DIR_ROOT, 'babylm')

HDLM_EXP_DIR = join(ALM_DIR, 'hd_lm/experiments')


def get_data_dir(primary_dir, fallback_dir):
    # Try to access the primary directory
    if os.path.exists(primary_dir):
        return primary_dir  
    else:
        return fallback_dir

# Usage
fallback = join("/home/t-smantena/internblobdl", 'infini_ind')
INFINIGRAM_INDEX_PATH = get_data_dir('/blob_data/infini_ind', fallback)

# hugging face token
TOKEN_HF = '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'
