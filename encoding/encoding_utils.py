import numpy as np
import time
import pathlib
import os
import h5py
from multiprocessing.pool import ThreadPool
from os.path import join, dirname

from ridge_utils.npp import zscore, mcorr
from ridge_utils.utils import make_delayed
from config import DATA_DIR

def apply_zscore_and_hrf(stories, downsampled_feat, trim, ndelays):
	"""Get (z-scored and delayed) stimulus for train and test stories.
	The stimulus matrix is delayed (typically by 2,4,6,8 secs) to estimate the
	hemodynamic response function with a Finite Impulse Response model.

	Args:
		stories: List of stimuli stories.

	Variables:
		downsampled_feat (dict): Downsampled feature vectors for all stories.
		trim: Trim downsampled stimulus matrix.
		delays: List of delays for Finite Impulse Response (FIR) model.

	Returns:
		delstim: <float32>[TRs, features * ndelays]
	"""
	stim = [zscore(downsampled_feat[s][5+trim:-trim]) for s in stories]
	stim = np.vstack(stim)
	delays = range(1, ndelays+1)
	delstim = make_delayed(stim, delays)
	return delstim


def get_response(stories, subject):
    """Get the subject's fMRI response for stories."""
    main_path = pathlib.Path(__file__).parent.parent.resolve()
    subject_dir = join(DATA_DIR, "ds003020/derivative/preprocessed_data/%s" % subject)
    resp = []
    for story in stories:
        resp_path = os.path.join(subject_dir, "%s.hf5" % story)
        with h5py.File(resp_path, "r") as hf:
            data = hf["data"][:]
            resp.extend(data)
            #print(data.shape, story)
    resp = np.array(resp)
    return resp


def get_response_ind_filt(stories, subject, include):
    """Get the subject's fMRI response for stories, filtered by specified indices."""
    main_path = pathlib.Path(__file__).parent.parent.resolve()
    subject_dir = os.path.join(DATA_DIR, "ds003020/derivative/preprocessed_data/%s" % subject)
    resp = []
    for story in stories:
        resp_path = os.path.join(subject_dir, "%s.hf5" % story)
        with h5py.File(resp_path, "r") as hf:
            data = hf["data"][:]
            # Get the indices to include for the current story
            #print("start_shape", data.shape)
            indices_to_include = include[story]
            # print("valid_ind", len(indices_to_include), story)
            
            
            # print("valid_ind", indices_to_include)
            # print("story_len", data.shape[0])

            valid_indices = indices_to_include[10:-5]
            #print("VALID", valid_indices)
            valid_indices = [i-10 for i in valid_indices]
            # print("after_filt", indices_to_include, story)
            # print("valid_ind", len(valid_indices), story)
            # If valid_indices is not empty, filter the data
            if valid_indices:
                data = data[valid_indices, :]
            #print("final_shape", data.shape, story)
            resp.extend(data)
    
    resp = np.array(resp)
    return resp

def get_response_nonind_filt(stories, subject, include):
    """Get the subject's fMRI response for stories, filtered by specified indices."""
    main_path = pathlib.Path(__file__).parent.parent.resolve()
    subject_dir = join(DATA_DIR, "ds003020/derivative/preprocessed_data/%s" % subject)
    resp = []
    for story in stories:
        resp_path = os.path.join(subject_dir, "%s.hf5" % story)
        with h5py.File(resp_path, "r") as hf:
            data = hf["data"][:]
            total_indices = np.arange(data.shape[0] + 15) 
            #print("total_len", len(total_indices))
            # Get the indices to exclude for the current story
            indices_to_exclude = include[story]
            #print("inc_len", len(indices_to_exclude))
            indices_to_keep = np.setdiff1d(total_indices, indices_to_exclude)
            #print("keep", len(indices_to_keep), indices_to_keep)
            indices_to_keep = indices_to_keep[10:-5]
            indices_to_keep = [i-10 for i in indices_to_keep]
            #print("final_keep", len(indices_to_keep), indices_to_keep)
            if len(indices_to_keep) > 0:
                data = data[indices_to_keep, :]
            resp.extend(data)
    resp = np.array(resp)
    return resp

def get_permuted_corrs(true, pred, blocklen):
	nblocks = int(true.shape[0] / blocklen)
	true = true[:blocklen*nblocks]
	block_index = np.random.choice(range(nblocks), nblocks)
	index = []
	for i in block_index:
		start, end = i*blocklen, (i+1)*blocklen
		index.extend(range(start, end))
	pred_perm = pred[index]
	nvox = true.shape[1]
	corrs = np.nan_to_num(mcorr(true, pred_perm))
	return corrs

def permutation_test(true, pred, blocklen, nperms):
	start_time = time.time()
	pool = ThreadPool(processes=10)
	perm_rsqs = pool.map(
		lambda perm: get_permuted_corrs(true, pred, blocklen), range(nperms))
	pool.close()
	end_time = time.time()
	print((end_time - start_time) / 60)
	perm_rsqs = np.array(perm_rsqs).astype(np.float32)
	real_rsqs = np.nan_to_num(mcorr(true, pred))
	pvals = (real_rsqs <= perm_rsqs).mean(0)
	return np.array(pvals), perm_rsqs, real_rsqs

def run_permutation_test(zPresp, pred, blocklen, nperms, mode='', thres=0.001):
	assert zPresp.shape == pred.shape, print(zPresp.shape, pred.shape)

	start_time = time.time()
	ntr, nvox = zPresp.shape
	partlen = nvox
	pvals, perm_rsqs, real_rsqs = [[] for _ in range(3)]

	for start in range(0, nvox, partlen):
		print(start, start+partlen)
		pv, pr, rs = permutation_test(zPresp[:, start:start+partlen], pred[:, start:start+partlen],
									  blocklen, nperms)
		pvals.append(pv)
		perm_rsqs.append(pr)
		real_rsqs.append(rs)
	pvals, perm_rsqs, real_rsqs = np.hstack(pvals), np.hstack(perm_rsqs), np.hstack(real_rsqs)

	assert pvals.shape[0] == nvox, (pvals.shape[0], nvox)
	assert perm_rsqs.shape[0] == nperms, (perm_rsqs.shape[0], nperms)
	assert perm_rsqs.shape[1] == nvox, (perm_rsqs.shape[1], nvox)
	assert real_rsqs.shape[0] == nvox, (real_rsqs.shape[0], nvox)

	cci.upload_raw_array(os.path.join(save_location, '%spvals'%mode), pvals)
	cci.upload_raw_array(os.path.join(save_location, '%sperm_rsqs'%mode), perm_rsqs)
	cci.upload_raw_array(os.path.join(save_location, '%sreal_rsqs'%mode), real_rsqs)
	print((time.time() - start_time)/60)
	
	pID, pN = fdr_correct(pvals, thres)
	cci.upload_raw_array(os.path.join(save_location, '%sgood_voxels'%mode), (pvals <= pN))
	cci.upload_raw_array(os.path.join(save_location, '%spN_thres'%mode), np.array([pN, thres], dtype=np.float32))
	return