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
import alm2.config
from alm2.mechlm import InfiniGram, InfiniGramModel, IncontextFuzzyLM


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
        choices=['infini-gram', 'infini-gram-w-incontext', 'incontext-infini-gram', 'incontext-fuzzy'],
        help="type of model to inference",
    )
    parser.add_argument(
        "--tokenizer_checkpoint",
        type=str,
        default="gpt2",
        help="checkpoint for tokenizer",
        choices=['gpt2', 'llama2']
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="hf_openwebtext_gpt2",
        help="checkpoint for infini-gram",
    )
    parser.add_argument(
        "--fuzzy_tokenizer",
        type=str,
        default="llama2",
        help="checkpoint for fuzzy matching",
        choices=['gpt2', 'llama2'],
    )
    parser.add_argument(
        "--fuzzy_lm_name",
        type=str,
        default="llama2",
        help="checkpoint for fuzzy matching (e.g., gpt2, llama2, path to model checkpoint)",
    )

    return parser.parse_args()

# Change the checkpoint for with context and the plain infinigram versions
def setup_model(save_dir, model_type, seed=1, tokenizer_checkpoint="gpt2", checkpoint="hf_openwebtext_gpt2", DATA_DIR=""):
    lm = None
    tokenizer = None

    # Set seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    fuzzy_tokenizer = tokenizer_checkpoint
    fuzzy_lm_name = tokenizer_checkpoint
    if tokenizer_checkpoint == "gpt2_dist":
        fuzzy_tokenizer = "gpt2"
        fuzzy_lm_name = DATA_DIR[:-5] + "/infini_ind/dist_gpt/ckpt-last-00.pt"
        tokenizer_checkpoint = "gpt2"
    if tokenizer_checkpoint == "llama2_dist":
        fuzzy_tokenizer = "llama2"
        fuzzy_lm_name = DATA_DIR[:-5] + "/infini_ind/dist_llama/ckpt-last-00.pt"
        tokenizer_checkpoint = "llama2"
        
    

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
    if fuzzy_tokenizer == 'llama2':
        fuzzy_tokenizer = 'meta-llama/Llama-2-7b-hf'
    elif fuzzy_tokenizer == 'llama3':
        fuzzy_tokenizer = 'meta-llama/Meta-Llama-3-8B'
    
    
    # Start model setup
    logger.info('Start building a model...')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint, add_bos_token=False, add_eos_token=False)
    model_kwargs = dict(
        context_length=np.inf,
        random_state=seed,
    )
    print("loading the lm")
    if model_type in ['infini-gram', 'infini-gram-w-incontext']:
        checkpoint_path = join(alm2.config.INFINIGRAM_INDEX_PATH, checkpoint)
        lm = InfiniGram(
            load_to_ram=True,
            tokenizer=tokenizer,
            infinigram_checkpoint=checkpoint_path,
            **model_kwargs
        )
    elif model_type in ['incontext-fuzzy']:
        fuzzy_tokenizer = AutoTokenizer.from_pretrained(fuzzy_tokenizer, add_bos_token=False, add_eos_token=False, token=alm2.config.TOKEN_HF)
        fuzzy_tokenizer.pad_token_id = fuzzy_tokenizer.eos_token_id
        if fuzzy_lm_name == 'llama2':
            fuzzy_lm_name = 'meta-llama/Llama-2-7b-hf'
        elif fuzzy_lm_name == 'llama3':
            fuzzy_lm_name = 'meta-llama/Meta-Llama-3-8B'
        elif fuzzy_lm_name.endswith('.pt'):
            model_kwargs['context_length'] = 32
        lm = IncontextFuzzyLM(
            tokenizer=tokenizer,
            fuzzy_tokenizer=fuzzy_tokenizer,
            fuzzy_lm_name=fuzzy_lm_name,
            **model_kwargs
        )
    logger.info('Done building a model...')

    return lm, tokenizer

def generate_embeddings(input_list, lm, tokenizer, lm_b, tokenizer_b, save_dir, story, model_type):
    vectors = []
    og_use_incontext = model_type.endswith('w-incontext')
    use_incontext = og_use_incontext
    og_model = model_type
    og_lm = lm
    og_tokenizer = tokenizer

    for i, word in enumerate(input_list):
        segment = input_list[max(0, i + 1 - 1024):i + 1]
    
        input_str = "".join(segment)
        token_ids = tokenizer(input_str)['input_ids']
        token_ids = torch.tensor(token_ids)

       # Ensure token_ids does not exceed the model's maximum sequence length
        max_len = min(len(token_ids), 1024)  # Adjust this based on your model's limit
        token_ids = token_ids[:max_len]
        if len(segment) < 3:
            lm = lm_b
            tokenizer = tokenizer_b
            model_type = 'infini-gram'
            use_incontext = False
        input_str = "".join(segment)

        #token_ids = np.array(tokenizer(input_str)['input_ids'], dtype=np.int32)
        # print(f"Token IDs length: {len(segment)}")
        # print(f"Model Type: {model_type}")
        # print(f"Use In-Context: {use_incontext}")
        # print(f"Token IDs type: {token_ids.dtype}")  # Debugging line

        if model_type in ['incontext-infini-gram']:
            lm = InfiniGramModel.from_data(documents_tkn=token_ids[:-1], tokenizer=tokenizer)
            prob_next_distr = lm.predict_prob(np.array(token_ids))
            next_token_probs = prob_next_distr.distr
        elif model_type in ['incontext-fuzzy']:
            # insert topk=10, for topk
            next_token_probs, others = lm.predict_prob(token_ids, incontext_mode='fuzzy', return_others=True)
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

# if __name__ == '__main__':
#     # hyperparams ######################
#     args = get_args()

#     model_kwargs = dict(
#         context_length=np.inf,
#         random_state=args.seed,
#     )

#     # set seed
#     np.random.seed(args.seed)
#     random.seed(args.seed)
#     torch.manual_seed(args.seed)

#     # set up saving
#     r = defaultdict(list)
#     r.update(vars(args))

#     os.makedirs(args.save_dir, exist_ok=True)
#     with open(join(args.save_dir, 'args.txt'), 'w') as f:
#         for k, v in vars(args).items():
#             f.write(f'{k}: {v}\n')

#     # set up logging
#     logger = logging.getLogger()
#     logging.basicConfig(filename=join(args.save_dir, 'inference.log'), level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S%p')
#     stream_handler = logging.StreamHandler()
#     stream_handler.setLevel(logging.INFO)
#     logger.addHandler(stream_handler)

#     logger.info(f'Save Dir: {args.save_dir}')

#     if args.tokenizer_checkpoint == 'llama2':
#         args.tokenizer_checkpoint = 'meta-llama/Llama-2-7b-hf'
#     elif args.tokenizer_checkpoint == 'llama3':
#         args.tokenizer_checkpoint = 'meta-llama/Meta-Llama-3-8B'
#     if args.fuzzy_tokenizer == 'llama2':
#         args.fuzzy_tokenizer = 'meta-llama/Llama-2-7b-hf'
#     elif args.fuzzy_tokenizer == 'llama3':
#         args.fuzzy_tokenizer = 'meta-llama/Meta-Llama-3-8B'
    
#     logger.info('Start building a model...')
#     tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_checkpoint, add_bos_token=False, add_eos_token=False, token=alm.config.TOKEN_HF)
#     use_incontext = False
#     if args.model_type in ['infini-gram', 'infini-gram-w-incontext']:
#         if args.model_type == 'infini-gram-w-incontext':
#             use_incontext=True
#         checkpoint_path = join(alm.config.INFINIGRAM_INDEX_PATH, args.checkpoint)
#         lm = InfiniGram(
#             load_to_ram=True,
#             tokenizer=tokenizer,
#             infinigram_checkpoint=checkpoint_path,
#             **model_kwargs
#         )
#     elif args.model_type in ['incontext-fuzzy']:
#         fuzzy_tokenizer = AutoTokenizer.from_pretrained(args.fuzzy_tokenizer, add_bos_token=False, add_eos_token=False, token=alm.config.TOKEN_HF)
#         fuzzy_tokenizer.pad_token_id = fuzzy_tokenizer.eos_token_id
#         if args.fuzzy_lm_name == 'llama2':
#             args.fuzzy_lm_name = 'meta-llama/Llama-2-7b-hf'
#         elif args.fuzzy_lm_name == 'llama3':
#             args.fuzzy_lm_name = 'meta-llama/Meta-Llama-3-8B'
#         elif args.fuzzy_lm_name.endswith('.pt'):
#             model_kwargs['context_length'] = 32
#         lm = IncontextFuzzyLM(
#             tokenizer=tokenizer,
#             fuzzy_tokenizer=fuzzy_tokenizer,
#             fuzzy_lm_name=args.fuzzy_lm_name,
#             **model_kwargs
#         )
#     # TODO: change input string
#     input_str = "Hi, Sriya. Hi,"
#     token_ids = tokenizer(input_str)['input_ids']
#     if args.model_type in ['incontext-infini-gram']:
#         lm = InfiniGramModel.from_data(documents_tkn=token_ids[:-1], tokenizer=tokenizer)
#         prob_next_distr = lm.predict_prob(np.array(token_ids))
#         all_count = prob_next_distr.count
#         next_token_probs = prob_next_distr.distr
#         others = {'prompt_cnt': all_count, 'suffix_len': prob_next_distr.effective_n, 'sparse': prob_next_distr.sparse}
#     elif args.model_type in ['incontext-fuzzy']:
#         next_token_probs, others = lm.predict_prob(token_ids, incontext_mode='fuzzy', return_others=True)
#     else:
#         next_token_probs, others = lm.predict_prob(token_ids, use_incontext=use_incontext, return_others=True)

#     print(next_token_probs)