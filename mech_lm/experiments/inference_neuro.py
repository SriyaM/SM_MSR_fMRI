from collections import defaultdict
from transformers import AutoTokenizer
from tqdm import tqdm
from os.path import join
import joblib
import logging
import numpy as np
import random
import torch
import os
import pickle

import argparse
import alm.config
from alm.mechlm import InfiniGram, InfiniGramModel

# Use this file to generate embeddings using the infinigram models and save them before returning to neuro mapping

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=join(alm.config.SAVE_DIR_ROOT, "mechlm-neuro", "infer"),
        help="directory for saving",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default='infini-gram',
        choices=['infini-gram', 'infini-gram-w-incontext', 'incontext-infini-gram'],
        help="type of model to inference",
    )
    parser.add_argument(
        "--tokenizer_checkpoint",
        type=str,
        default="gpt2",
        help="checkpoint for tokenizer",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="hf_openwebtext_gpt2",
        help="checkpoint for infini-gram",
    )

    return parser.parse_args()

def get_infini_embeddings(story, input_list, save_dir, model_type, lm=None, tokenizer=None, seed=1, tokenizer_checkpoint="gpt2", checkpoint="hf_openwebtext_gpt2"):

    if lm == None:

        model_kwargs = dict(
            context_length=np.inf,
            random_state=seed,
        )
    
        # Set seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        # Set up saving
        r = defaultdict(list)
        
        # Update r with the function's parameters
        r.update({
            "input": input,
            "model_type": model_type,
            "seed": seed,
            "tokenizer_checkpoint": tokenizer_checkpoint,
            "checkpoint": checkpoint
        })

        os.makedirs(save_dir, exist_ok=True)
        with open(join(save_dir, 'args.txt'), 'w') as f:
            for k, v in r.items():
                f.write(f'{k}: {v}\n')

        # Set up logging
        logger = logging.getLogger()
        logging.basicConfig(
            filename=join(save_dir, 'inference.log'),
            level=logging.INFO,
            format='%(asctime)s %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S%p'
        )

        logger.info(f'Save Dir: {save_dir}')

        if tokenizer_checkpoint == 'llama2':
            tokenizer_checkpoint = 'meta-llama/Llama-2-7b-hf'
        elif tokenizer_checkpoint == 'llama3':
            tokenizer_checkpoint = 'meta-llama/Meta-Llama-3-8B'

        logger.info('Start building a model...')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint, add_bos_token=False, add_eos_token=False, token=alm.config.TOKEN_HF)
        use_incontext = False
        if model_type in ['infini-gram', 'infini-gram-w-incontext']:
            if model_type == 'infini-gram-w-incontext':
                use_incontext=True
            checkpoint_path = join(alm.config.INFINIGRAM_INDEX_PATH, checkpoint)
            lm = InfiniGram(
                load_to_ram=True,
                tokenizer=tokenizer,
                infinigram_checkpoint=checkpoint_path,
                **model_kwargs
            )
        logger.info('Done building a model...')

    vectors = []

    use_incontext = False
    if model_type in ['infini-gram', 'infini-gram-w-incontext']:
        if model_type == 'infini-gram-w-incontext':
            use_incontext=True
    
    for i, word in enumerate(input_list):
        segment = input_list[max(0, i+1-1024):i+1]
        input_str = "".join(segment)
    
        token_ids = tokenizer(input_str)['input_ids']
        if model_type in ['incontext-infini-gram']:
            lm = InfiniGramModel.from_data(documents_tkn=token_ids[:-1], tokenizer=tokenizer)
            prob_next_distr = lm.predict_prob(np.array(token_ids))
            next_token_probs = prob_next_distr.distr
        else:
            next_token_probs, others = lm.predict_prob(token_ids, use_incontext=use_incontext, return_others=True)
        vectors.append(next_token_probs)

    
    return np.array(vectors), lm, tokenizer

def setup_model(save_dir, model_type, seed=1, tokenizer_checkpoint="gpt2", checkpoint="hf_openwebtext_gpt2"):
    lm = None
    tokenizer = None

    # Set seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Set up saving directory
    os.makedirs(save_dir, exist_ok=True)
    args_dict = {
        "model_type": model_type,
        "seed": seed,
        "tokenizer_checkpoint": tokenizer_checkpoint,
        "checkpoint": checkpoint
    }

    # Save arguments for reference
    with open(os.path.join(save_dir, 'args.txt'), 'w') as file:
        for key, value in args_dict.items():
            file.write(f'{key}: {value}\n')

    # Set up logging
    logger = logging.getLogger()
    logging.basicConfig(
        filename=os.path.join(save_dir, 'inference.log'),
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p'
    )

    logger.info(f'Setup directory: {save_dir}')

    # Initialize tokenizer
    if tokenizer_checkpoint == 'llama2':
        tokenizer_checkpoint = 'meta-llama/Llama-2-7b-hf'
    elif tokenizer_checkpoint == 'llama3':
        tokenizer_checkpoint = 'meta-llama/Meta-Llama-3-8B'
    
    
    # Start model setup
    logger.info('Start building a model...')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint, add_bos_token=False, add_eos_token=False)
    model_kwargs = dict(
        context_length=np.inf,
        random_state=seed,
    )
    print("loading the lm")
    if model_type in ['infini-gram', 'infini-gram-w-incontext']:
        checkpoint_path = join(alm.config.INFINIGRAM_INDEX_PATH, checkpoint)
        lm = InfiniGram(
            load_to_ram=True,
            tokenizer=tokenizer,
            infinigram_checkpoint=checkpoint_path,
            **model_kwargs
        )
    logger.info('Done building a model...')

    return lm, tokenizer

def generate_embeddings(input_list, lm, tokenizer, lm_b, tokenizer_b, save_dir, story, model_type):
    vectors = []
    og_use_incontext = model_type.endswith('w-incontext')
    og_model = model_type
    og_lm = lm
    og_tokenizer = tokenizer

    for i, word in enumerate(input_list):
        segment = input_list[max(0, i + 1 - 1024):i + 1]
    
        input_str = "".join(segment)
        token_ids = tokenizer(input_str)['input_ids']

        # Ensure token_ids does not exceed the model's maximum sequence length
        max_len = min(len(token_ids), 1020)  # Adjust this based on your model's limit
        token_ids = token_ids[:max_len]
    
        if len(segment) < 3:
            lm = lm_b
            tokenizer = tokenizer_b
            model_type = 'infini-gram'
            use_incontext = False
        input_str = "".join(segment)
    
        token_ids = np.array(tokenizer(input_str)['input_ids'])
        #token_ids = np.array(tokenizer(input_str)['input_ids'], dtype=np.int32)
        # print(f"Token IDs length: {len(segment)}")
        # print(f"Model Type: {model_type}")
        # print(f"Use In-Context: {use_incontext}")
        # print(f"Token IDs type: {token_ids.dtype}")  # Debugging line

        if model_type in ['incontext-infini-gram']:
            lm = InfiniGramModel.from_data(documents_tkn=token_ids[:-1], tokenizer=tokenizer)
            prob_next_distr = lm.predict_prob(np.array(token_ids))
            next_token_probs = prob_next_distr.distr
        else:
            next_token_probs, others = lm.predict_prob(token_ids, use_incontext=use_incontext, return_others=True)
        if len(segment) < 3:
            lm = og_lm
            tokenizer = og_tokenizer
            model_type = og_model
            use_incontext = og_use_incontext
        vectors.append(next_token_probs)
    
    os.makedirs(os.path.join(save_dir, story), exist_ok=True)
    
    # Save the vectors to a pickle file
    pickle_file_path = os.path.join(save_dir, story, 'vectors.pkl')
    with open(pickle_file_path, 'wb') as pickle_file:
        pickle.dump(np.array(vectors), pickle_file)
    
    return np.array(vectors)
