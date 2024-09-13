import os
from os.path import join, dirname


REPO_DIR = join(dirname(dirname(os.path.abspath(__file__))))
EM_DATA_DIR = join(REPO_DIR, 'em_data')


def get_data_dir(primary_dir, fallback_dir):
    # Try to access the primary directory
    if os.path.exists(primary_dir):
        return primary_dir  
    else:
        return fallback_dir

# Usage
fallback = join("/home/t-smantena/internblobdl", 'data')
DATA_DIR = get_data_dir('/blob_data/data', fallback)
#DATA_DIR = "/blobdata/data"