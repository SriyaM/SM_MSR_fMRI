import os
import sys
import numpy as np
import argparse
import json
from os.path import join, dirname
import logging

from encoding_utils import *
from feature_spaces import _FEATURE_CONFIG, get_feature_space
from ridge_utils.ridge import bootstrap_ridge, bootstrap_ridge_3_models
from config import  REPO_DIR, EM_DATA_DIR

from config import REPO_DIR, EM_DATA_DIR, DATA_DIR

# Print paths for debugging
print(f"REPO_DIR from config: {REPO_DIR}")
print(f"EM_DATA_DIR from config: {EM_DATA_DIR}")
print(f"DATA_DIR from config: {DATA_DIR}")



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--subject", type=str, required=True)
	parser.add_argument("--feature_ind", type=str, required=True)
	parser.add_argument("--feature_b", type=str, required=True)
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
	assert feature_ind in _FEATURE_CONFIG.keys(), "Available feature spaces:" + fs
	assert feature_b in _FEATURE_CONFIG.keys(), "Available feature spaces:" + fs
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
	
	save_location_b = get_save_location("/blob_data/", "/home/t-smantena/internblobdl", feature_b, numeric_mod, subject)
	save_location_ind = get_save_location("/blob_data/", "/home/t-smantena/internblobdl", feature_ind, numeric_mod, subject)

	#save_location = join("/blob_data/", "results", feature + "_" + numeric_mod, subject)
	# save_location = join(REPO_DIR, "results", feature + "_" + numeric_mod, subject)

	print("Saving induction encoding model & results to:", save_location_b)
	print("Saving basline encoding model & results to:", save_location_ind)
	sys.stdout.flush()
	os.makedirs(save_location_b, exist_ok=True)
	os.makedirs(save_location_ind, exist_ok=True)

	# should only return the relevant words
	# the inductoin should be trained on some, the baseline trained on all
	# Each one returns the correct number of downsampled feeatures now
	downsampled_feat_ind, rem = get_feature_space(feature_ind, allstories, x, k)
	downsampled_feat_b = get_feature_space(feature_b, allstories, test_stories, train_stories, rem, x, k)
	print("Stimulus & Response parameters:")
	print("trim: %d, ndelays: %d" % (trim, ndelays))

	print("FEATURE_SHAPE_ind_test_alt", downsampled_feat_ind["alternateithicatom"].shape)

	print("FEATURE_SHAPE_ind_train_where", downsampled_feat_ind["wheretheressmoke"].shape)

	print("FEATURE_SHAPE_b_test_alt", downsampled_feat_b["alternateithicatom"].shape)
	print("FEATURE_SHAPE_b_train_where", downsampled_feat_b["wheretheressmoke"].shape)


	# Delayed stimulus
	# need to change the test_stories here??
	delRstim_ind = apply_zscore_and_hrf(train_stories, downsampled_feat_ind, trim, ndelays)
	print("delRstim_ind: ", delRstim_ind.shape)
	delPstim_ind = apply_zscore_and_hrf(test_stories, downsampled_feat_ind, trim, ndelays)
	print("delPstim_ind: ", delPstim_ind.shape)

	delRstim_b = apply_zscore_and_hrf(train_stories, downsampled_feat_b, trim, ndelays)
	print("delRstim_b: ", delRstim_b.shape)
	# Need to have the test_stories only include the points that are not inducted
	delPstim_b = apply_zscore_and_hrf(test_stories, downsampled_feat_b, trim, ndelays)
	print("delPstim_b: ", delPstim_b.shape)

	zRresp_ind = get_response_ind_filt(train_stories, subject, rem)
	print("zRresp_ind: ", zRresp_ind.shape)
	zPresp_ind = get_response_ind_filt(test_stories, subject, rem)
	print("zPresp_ind: ", zPresp_ind.shape)

	print("rem_len", len(rem["wheretheressmoke"]))

	# Response
	zRresp_b = get_response(train_stories, subject)
	print("zRresp_b: ", zRresp_b.shape)
	zPresp_b = get_response_nonind_filt(test_stories, subject, rem)
	print("zPresp_b: ", zPresp_b.shape)

	# Ridge
	alphas = np.logspace(1, 3, 10)

	print("Ridge parameters:")
	print("nboots: %d, chunklen: %d, nchunks: %d, single_alpha: %s, use_corr: %s" % (
		nboots, chunklen, nchunks, single_alpha, use_corr))

	# adjust this to use switch between the models for prediction.
	# Create a new pred vector and then predict for all voxels
	wt_ind, corrs_ind, valphas_ind, bscorrs_ind, valinds_ind = bootstrap_ridge(
		delRstim_ind, zRresp_ind, delPstim_ind, zPresp_ind, alphas, nboots, chunklen, 
		nchunks, singcutoff=singcutoff, single_alpha=single_alpha, 
		use_corr=use_corr)
	
	wt_b, corrs_b, valphas_b, bscorrs_b, valinds_b = bootstrap_ridge(
		delRstim_b, zRresp_b, delPstim_b, zPresp_b, alphas, nboots, chunklen, 
		nchunks, singcutoff=singcutoff, single_alpha=single_alpha, 
		use_corr=use_corr)
	
	print("FINISHED ROUND")
	sys.stdout.flush()

	# Save regression results, induction
	np.savez("%s/weights" % save_location_b, wt_b)
	np.savez("%s/corrs" % save_location_b, corrs_b)
	np.savez("%s/valphas" % save_location_b, valphas_b)
	np.savez("%s/bscorrs" % save_location_b, bscorrs_b)
	np.savez("%s/valinds" % save_location_b, np.array(valinds_b))
	print("Total r2: %d" % sum(corrs_b * np.abs(corrs_b)))

	# Save regression results, non induction
	np.savez("%s/weights" % save_location_ind, wt_ind)
	np.savez("%s/corrs" % save_location_ind, corrs_ind)
	np.savez("%s/valphas" % save_location_ind, valphas_ind)
	np.savez("%s/bscorrs" % save_location_ind, bscorrs_ind)
	np.savez("%s/valinds" % save_location_ind, np.array(valinds_ind))
	print("Total r2: %d" % sum(corrs_ind * np.abs(corrs_ind)))

