import os
import sys
import numpy as np
import argparse
import json
from os.path import join, dirname
import logging

from encoding_utils import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from feature_spaces import _FEATURE_CONFIG, get_feature_space
from ridge_utils.ridge import bootstrap_ridge, bootstrap_ridge_3_models
from config import  REPO_DIR, EM_DATA_DIR

from config import REPO_DIR, EM_DATA_DIR, DATA_DIR


import torch
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel, BertForMaskedLM, LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# Print paths for debugging
print(f"REPO_DIR from config: {REPO_DIR}")
print(f"EM_DATA_DIR from config: {EM_DATA_DIR}")
print(f"DATA_DIR from config: {DATA_DIR}")



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--subject", type=str, required=True)
	parser.add_argument("--feature_1", type=str, required=True)
	parser.add_argument("--feature_2", type=str, required=True)
	parser.add_argument("--feature_3", type=str, required=True)
	parser.add_argument("--x", type=int)
	parser.add_argument("--k", type=int)
	parser.add_argument("--sessions", nargs='+', type=int, default=[1, 2, 3, 4, 5])
	parser.add_argument("--trim", type=int, default=5)
	parser.add_argument("--ndelays", type=int, default=4)
	parser.add_argument("--nboots", type=int, default=50)
	parser.add_argument("--chunklen", type=int, default=40)
	parser.add_argument("--nchunks", type=int, default=125)
	parser.add_argument("--singcutoff", type=float, default=1e-10)
	parser.add_argument("-use_corr", action="store_true")
	parser.add_argument("-single_alpha", action="store_true")
	logging.basicConfig(level=logging.INFO)


	args = parser.parse_args()
	globals().update(args.__dict__)

	fs = " ".join(_FEATURE_CONFIG.keys())
	assert feature_1 in _FEATURE_CONFIG.keys(), "Available feature spaces:" + fs
	assert feature_2 in _FEATURE_CONFIG.keys(), "Available feature spaces:" + fs
	assert feature_3 in _FEATURE_CONFIG.keys(), "Available feature spaces:" + fs
	assert np.amax(sessions) <= 5 and np.amin(sessions) >=1, "1 <= session <= 5"

	sessions = list(map(str, sessions))
	with open(join(EM_DATA_DIR, "sess_to_story.json"), "r") as f:
		sess_to_story = json.load(f) 
	train_stories, test_stories = [], []
	for sess in sessions:
		stories, tstory = sess_to_story[sess][0], sess_to_story[sess][1]
		train_stories.extend(stories)
		if tstory not in test_stories:
			test_stories.append(tstory)

	# There are 2 test stories and 
	train_stories.remove("haveyoumethimyet")
	test_stories.append("haveyoumethimyet")	

	assert len(set(train_stories) & set(test_stories)) == 0, "Train - Test overlap!"

	print("HELLO")
	sys.stdout.flush()
	
	# all stories is encoded into features
	allstories = list(set(train_stories) | set(test_stories))

	print(test_stories)
	print(train_stories)
	print(len(train_stories))


	# REMOVE LATER
	allstories.remove("adollshouse")
	train_stories.remove("adollshouse")

	
	# train_stories = ["wheretheressmoke"]
	# test_stories = ["alternateithicatom"]
	# all_stories = ["alternateithicatom", "alternateithicatom"]
	
	def get_save_location(primary_dir, fallback_dir, feature, numeric_mod, subject):
		primary_location = join(primary_dir, "results", feature + "_" + numeric_mod, subject)
		fallback_location = join(fallback_dir, "results", feature + "_" + numeric_mod, subject)
		
		try:
			# Try to create the directory in the primary location
			os.makedirs(primary_location, exist_ok=True)
			return primary_location
		except (OSError, IOError) as e:
			# If an error occurs, use the fallback location
			os.makedirs(fallback_location, exist_ok=True)
			return fallback_location
	
	numeric_mod = ""
	if x is not None:
		numeric_mod += str(x) + "x"
		if k is not None:
			numeric_mod += "_" + str(k) + "k"
	
	if x is None and k is not None:
		numeric_mod += str(k) + "k"
	
	save_location = get_save_location("/blob_data/", "/home/t-smantena/internblobdl", "llama_3_models", numeric_mod, subject)

	#save_location = join("/blob_data/", "results", feature + "_" + numeric_mod, subject)
	# save_location = join(REPO_DIR, "results", feature + "_" + numeric_mod, subject)

	print("Saving encoding model & results to:", save_location)
	sys.stdout.flush()
	os.makedirs(save_location, exist_ok=True)

	os.environ['HUGGINGFACE_TOKEN'] = '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'
	token = os.getenv('HUGGINGFACE_TOKEN')

	# Load the tokenizer and model with authentication token
	tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B', use_auth_token=token)
	model = AutoModel.from_pretrained('meta-llama/Meta-Llama-3-8B', use_auth_token=token)
	model.eval()

	# Check if ROCm is available and move model to GPU
	if torch.cuda.is_available():  # ROCm uses the same torch.cuda API
		device = torch.device('cuda')
		print(f"Using GPU: {torch.cuda.get_device_name(0)}")
		sys.stdout.flush()
	else:
		device = torch.device('cpu')
		print("ROCm not available, using CPU.")
		sys.stdout.flush()

	model.to(device)

	print("DONE LOADING")

	downsampled_feat_1 = get_feature_space(feature_1, allstories, x, k, model, tokenizer)
	downsampled_feat_2 = get_feature_space(feature_2, allstories, x, k, model, tokenizer)
	downsampled_feat_3 = get_feature_space(feature_3, allstories, x, k, model, tokenizer)

	print("Stimulus & Response parameters:")
	print("trim: %d, ndelays: %d" % (trim, ndelays))

	# Create 3 pairs of stimuli and store in a dictionary

	model1 = {}
	# Delayed stimulus
	model1["Rstim"] = apply_zscore_and_hrf(train_stories, downsampled_feat_1, trim, ndelays)
	print("delRstim_1: ", model1["Rstim"].shape)
	model1["Pstim"] = apply_zscore_and_hrf(test_stories, downsampled_feat_1, trim, ndelays)
	print("delPstim_1: ", model1["Pstim"].shape)

	model2 = {}
	# Delayed stimulus
	model2["Rstim"] = apply_zscore_and_hrf(train_stories, downsampled_feat_2, trim, ndelays)
	print("delRstim_1: ", model2["Rstim"].shape)
	model2["Pstim"] = apply_zscore_and_hrf(test_stories, downsampled_feat_2, trim, ndelays)
	print("delPstim_1: ", model2["Pstim"].shape)

	model3 = {}
	# Delayed stimulus
	model3["Rstim"] = apply_zscore_and_hrf(train_stories, downsampled_feat_3, trim, ndelays)
	print("delRstim_3: ", model3["Rstim"].shape)
	model3["Pstim"] = apply_zscore_and_hrf(test_stories, downsampled_feat_3, trim, ndelays)
	print("delPstim_3: ", model3["Pstim"].shape)

	models = [model1, model2, model3]

	# Response
	zRresp = get_response(train_stories, subject)
	print("zRresp: ", zRresp.shape)
	zPresp = get_response(test_stories, subject)
	print("zPresp: ", zPresp.shape)

	# Ridge
	alphas = np.logspace(1, 3, 10)

	print("Ridge parameters:")
	print("nboots: %d, chunklen: %d, nchunks: %d, single_alpha: %s, use_corr: %s" % (
		nboots, chunklen, nchunks, single_alpha, use_corr))

	wt, corrs, valphas, bscorrs, valinds = bootstrap_ridge_3_models(
		models, zRresp, zPresp, alphas, nboots, chunklen, 
		nchunks, singcutoff=singcutoff, single_alpha=single_alpha, 
		use_corr=use_corr)
	
	print("FINISHED ROUND")
	sys.stdout.flush()

	# Save regression results.
	np.savez("%s/weights" % save_location, wt)
	np.savez("%s/corrs" % save_location, corrs)
	np.savez("%s/valphas" % save_location, valphas)
	np.savez("%s/bscorrs" % save_location, bscorrs)
	np.savez("%s/valinds" % save_location, np.array(valinds))
	print("Total r2: %d" % sum(corrs * np.abs(corrs)))
