from transformers import AutoTokenizer
from tqdm import tqdm
from os.path import join
import joblib
import sys
import json

from os.path import abspath, dirname, join

# Add the root directory of your project to sys.path
ROOT_DIR = abspath(join(dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT_DIR)

# Now try importing the modules
print(sys.path)
from encoding.config import REPO_DIR, EM_DATA_DIR, DATA_DIR
from encoding.ridge_utils.stimulus_utils import load_textgrids, load_simulated_trfiles
from encoding.ridge_utils.DataSequence import DataSequence

DEFAULT_BAD_WORDS = frozenset(["sentence_start", "sentence_end", "br", "lg", "ls", "ns", "sp"])

TRAIN_STORIES = ['alternateithicatom', 'souls', 'avatar', 'legacy', 'odetostepfather', 'undertheinfluence', 'howtodraw', 'myfirstdaywiththeyankees', 'naked', 'life', 'stagefright', 'tildeath', 'fromboyhoodtofatherhood', 'sloth', 'exorcism', 'inamoment', 'theclosetthatateeverything', 'adventuresinsayingyes', 'buck', 'swimmingwithastronauts', 'thatthingonmyarm', 'eyespy', 'itsabox', 'hangtime']

TRAIN_STORIES = []

TEST_STORIES = ['wheretheressmoke', 'haveyoumethimyet']

OUTPUT_DIR = "/home/t-smantena/internblobdl/hd_data"

def get_story_wordseqs(stories):
	grids = load_textgrids(stories, DATA_DIR)
	with open(join(DATA_DIR, "ds003020/derivative/respdict.json"), "r") as f:
		respdict = json.load(f)
	# a dictionary of words listened to between the start and end time
	trfiles = load_simulated_trfiles(respdict)
	# returns a dictionary with dataseq stories that have a transcript with filtered words and metadata
	wordseqs = make_word_ds(grids, trfiles)
	return wordseqs


def make_word_ds(grids, trfiles, bad_words=DEFAULT_BAD_WORDS):
    """Creates DataSequence objects containing the words from each grid, with any words appearing
    in the [bad_words] set removed.
    """
    ds = dict()
    stories = list(set(trfiles.keys()) & set(grids.keys()))
    for st in stories:
        grtranscript = grids[st].tiers[1].make_simple_transcript()
        ## Filter out bad words
        goodtranscript = [x for x in grtranscript
                          if x[2].lower().strip("{}").strip() not in bad_words]
        d = DataSequence.from_grid(goodtranscript, trfiles[st][0])
        ds[st] = d

    return ds

def preprocess_story_data(allstories, tokenizer_checkpoint='gpt2'):
    '''Tokenizes all the babylm text and dumps it into a joblib file
    '''
    wordseqs = get_story_wordseqs(allstories)
    data = []
    for story in allstories:
        data.extend(wordseqs[story].data)
    texts = '\n\n'.join(data)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
        
    tokens = []
    chunk_size = 1000
    for i in tqdm(range(0, len(texts), chunk_size)):
        tokens += tokenizer(texts[i:i+chunk_size])['input_ids']
    joblib.dump(tokens, join(OUTPUT_DIR, 'full_test.joblib'))

if __name__ == '__main__':
    # first need to manually download raw babyLM data into alm.config.BABYLM_DATA_DIR
    allstories = list(set(TRAIN_STORIES) | set(TEST_STORIES))
    preprocess_story_data(allstories)
    print('Done!')