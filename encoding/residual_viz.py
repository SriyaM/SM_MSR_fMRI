
import os
import sys
from matplotlib.pyplot import figure
from encoding_utils import *
import numpy as np
#import cortex
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from feature_spaces import *
import pickle

import torch
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel, BertForMaskedLM, LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import matplotlib.patches as mpatches
from scipy.stats import ttest_rel, wilcoxon


ALLSTORIES = ['wheretheressmoke', 'haveyoumethimyet', 'alternateithicatom', 'souls', 'avatar', 'legacy', 'odetostepfather', 'undertheinfluence', 'howtodraw', 'myfirstdaywiththeyankees', 'naked', 'life', 'stagefright', 'tildeath', 'fromboyhoodtofatherhood', 'sloth', 'exorcism', 'inamoment', 'theclosetthatateeverything', 'adventuresinsayingyes', 'buck', 'swimmingwithastronauts', 'thatthingonmyarm', 'eyespy', 'itsabox', 'hangtime']

CUR_STORY = 'wheretheressmoke'

if CUR_STORY == 'haveyoumethimyet':
    STORY_IND = 1
else:
    STORY_IND = 0
    

def save_features():
    allstories = ["wheretheressmoke"]
    k = 64
    features = ["llama_w_prefix_same", "llama_w_prefix_att", "llama", "llama_w_prefix_means"]

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

    for feature in features:
        print("extracting" + feature)
        downsampled_feat = get_feature_space(feature, allstories, 0, k, model, tokenizer)
        name = "/blob_data/" + feature + '.pkl'
        with open(name, 'wb') as f:
            pickle.dump(downsampled_feat, f)





def save_features_infini():
    allstories = [CUR_STORY]
    x = 900
    features = ["infinigram_p", "infinigram_w_cont_p", "incont_infinigram_p"]

    for feature in features:
        print("extracting" + feature)
        downsampled_feat = get_feature_space(feature, allstories, x, 0)
        name = "/home/t-smantena/internblobdl/infini_features/story2/" + feature + str(x) + '.pkl'
        with open(name, 'wb') as f:
            pickle.dump(downsampled_feat, f)




def zscore(data):
    return (data - np.mean(data)) / np.std(data)


def plot_residual(OUTPUT, emb, save):

    test_stories = [CUR_STORY]
    zPresp = get_response(test_stories, 'UTS03')

    print("load")
    data = np.load(OUTPUT + 'weights.npz')
    wt = data[data.files[0]]

    with open(emb, 'rb') as f:
        feature = pickle.load(f)

	# Delayed stimulus
    delPstim = apply_zscore_and_hrf(test_stories, feature, 5, 4)
    
    # First let's refresh ourselves on the shapes of these matrices
    print ("zPresp has shape: ", zPresp.shape)
    print ("wt has shape: ", wt.shape)
    print ("delPstim has shape: ", delPstim.shape)

    pred = np.dot(delPstim, wt)

    print ("pred has shape: ", pred.shape)

    f = figure(figsize=(15,5))
    ax = f.add_subplot(1,1,1)

    # NEED TO PICK THE VOXEL WITH THE TOP CORR
    
    print("load")
    data = np.load(OUTPUT + 'corrs.npz')
    corr = data[data.files[0]]
    selvox = np.argmax(corr)
    print("done")

    # SHOULD BE DONE PLOTTING

    # CALC RESID

    # Calculate residuals for all voxels
    residuals = zPresp - pred
    # Average residuals across all voxels at each time point
    residuals = np.mean(residuals, axis=1)

    residuals = np.abs(residuals)

    # Plot residuals
    fig_res, ax_res = plt.subplots(figsize=(15,5))
    ax_res.plot(residuals, 'b', label="Residuals")
    ax_res.set_title("Residuals of Prediction")
    ax_res.set_xlabel("Time (fMRI time points)")
    ax_res.legend()

    # Optionally show figures if running in an environment without automatic rendering
    plt.show()

    wordseqs = get_story_wordseqs(test_stories)
    words = wordseqs[test_stories[0]].data
    word_times = wordseqs[test_stories[0]].data_times
    tr_times = wordseqs[test_stories[0]].tr_times

    print("TR", len(tr_times), tr_times)
    print("RESID_LEN", len(residuals))

    chunks = np.split(words, wordseqs[test_stories[0]].split_inds)

    zeros_at_start = np.zeros(10)  # Creates a NumPy array of 10 zeros
    extended_list = np.concatenate([zeros_at_start, residuals])  # Concatenates zeros to the beginning

    # Add zeros to the end
    zeros_at_end = np.zeros(5)  # Creates a NumPy array of 5 zeros
    residuals = np.concatenate([extended_list, zeros_at_end])
    print(residuals)
    print("NEW RESID LEN", len(residuals))

    word_list = []
    residual_list = []
    
    for i, chunk in enumerate(chunks):
        for word in chunk:
            word_list.append(word)
            residual_list.append(residuals[i])
        
    ret_words = word_list
    ret_resid = residual_list
    
    # Normalize residuals for color mapping
    norm = plt.Normalize(min(residual_list), max(residual_list))
    colors = plt.cm.coolwarm(norm(residual_list))  # Using the coolwarm colormap

    # Define number of rows and columns for words
    num_rows = 90
    words_per_row = len(word_list) // num_rows

    plt.figure(figsize=(20, 26))  # Increase figure size to better accommodate words
    ax = plt.gca()

    # Adjust text positions and plot in multiple lines
    for i, (word_list, color) in enumerate(zip(word_list, colors)):
        row = i // words_per_row
        col = i % words_per_row
        x_position = col / words_per_row
        y_position = 1 - (row / num_rows)
        ax.text(x_position, y_position, word_list, color='black', fontsize=12,
        ha='center', va='center', transform=ax.transAxes,
        bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.1'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.axis('off')
    plt.tight_layout()  # Adjust layout to make room for all elements
    plt.show()

    plt.savefig(save)

    return (ret_resid, ret_words)

def word_plot(residual_list, word_list, save):
    # Normalize residuals for color mapping
    norm = plt.Normalize(min(residual_list), max(residual_list))
    cmap = plt.cm.coolwarm
    colors = cmap(norm(residual_list))  # Using the coolwarm colormap

    # Define number of rows and columns for words
    num_rows = 63
    words_per_row = len(word_list) // num_rows

    fig, ax = plt.subplots(figsize=(20, 20))  # Increase figure size to better accommodate words

    # Adjust text positions and plot in multiple lines
    for i, (word, color) in enumerate(zip(word_list, colors)):
        row = i // words_per_row
        col = i % words_per_row
        x_position = col / words_per_row
        y_position = 1 - (row / num_rows)
        ax.text(x_position, y_position, word, color='black', fontsize=12,
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.1'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.axis('off')
    plt.tight_layout()  # Adjust layout to make room for all elements

    # Create a ScalarMappable with the same normalization and colormap
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # You can safely ignore this line for modern Matplotlib versions

    # Create colorbar
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
    cbar.set_label('Residual Values')

    # Optionally save the plot
    if save:
        plt.savefig(save)

    plt.show()

def plot_best(same, att, base, word_list, save):
    # Define a list of light colors with their corresponding names
    color_names = ['Infinigram', 'Infinigram With Context', 'Incontext Infinigram', 'No Significant Difference']
    colors = [ 'lightgreen', 'lightcoral', 'lightblue', 'white']  # Light colors for 0, 1, 2, 3, and no significant difference

    # Compute global differences and their statistics
    global_differences = [abs(s - a) for s, a in zip(same, att)] + [abs(s - b) for s, b in zip(same, base)] + [abs(a - b) for a, b in zip(att, base)]
    global_mean_diff = np.mean(global_differences)
    global_std_dev = np.std(global_differences)

    # Generate val_list where each value determines the color of the corresponding word
    val_list = []
    for i in range(len(same)):
        res = [same[i], att[i], base[i]]
        min_val = min(res)
        ind = res.index(min_val)
        differences = [x - min_val for x in res]

        other_diffs = [x for j, x in enumerate(differences) if j != ind]  # Exclude the minimum itself
        avg_diff = sum(other_diffs) / len(other_diffs)

        # Check if the average of the differences is significantly greater than global average
        if avg_diff > global_mean_diff + (global_std_dev/3):
            val_list.append(ind)  # Significantly better
        else:
            val_list.append(3)  # No significant difference
                
    # Define number of rows and columns for words
    num_rows = 85
    words_per_row = len(word_list) // num_rows

    fig, ax = plt.subplots(figsize=(20, 30))  # Increased figure height

    for i, word in enumerate(word_list):
        row = i // words_per_row
        col = i % words_per_row
        x_position = col / words_per_row
        y_position = 1 - (row / num_rows)
        color_index = val_list[i]
        ax.text(x_position, y_position, word, color='black', fontsize=12,
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(facecolor=colors[color_index], edgecolor='none', boxstyle='round,pad=0.1'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.axis('off')

    legend_patches = [mpatches.Patch(color=colors[i], label=color_names[i]) for i in range(len(colors))]
    legend = ax.legend(handles=legend_patches, title='Legend', loc='upper left', bbox_to_anchor=(1, 1), fontsize='large')

    plt.title('Word Plot with Color-Coded Values', pad=20)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if save:
        plt.savefig(save)

    plt.show()

    # Initialize lists to store phrases
    list_same, list_att, list_base = [], [], []
    temp_phrase = word_list[0]

    # Initialize current best category that is significantly better
    current_best = val_list[0]

    # Loop through val_list and word_list
    for i in range(1, len(val_list)):
        if val_list[i] == current_best:
            temp_phrase += " " + word_list[i]
        else:
            if current_best == 0:
                list_same.append(temp_phrase)
            elif current_best == 1:
                list_att.append(temp_phrase)
            elif current_best == 2:
                list_base.append(temp_phrase)

            # Reset for new phrase
            current_best = val_list[i]
            temp_phrase = word_list[i]

    # Append the last phrase to the appropriate list
    if current_best == 0:
        list_same.append(temp_phrase)
    elif current_best == 1:
        list_att.append(temp_phrase)
    elif current_best == 2:
        list_base.append(temp_phrase)

    # Print or return the lists
    print(color_names[0], list_same)
    print(color_names[1], list_att)
    print(color_names[2], list_base)








def plot_best_top_30(same, att, base, word_list, save):
    # Define a list of light colors with their corresponding names
    color_names = ['Infinigram', 'Infinigram With Context', 'Incontext Infinigram', 'No Significant Difference']
    colors = [ 'lightgreen', 'lightcoral', 'lightblue', 'white']  # Light colors for 0, 1, 2, 3, and no significant difference
    
    wordseqs = get_story_wordseqs(ALLSTORIES)
    words = wordseqs[ALLSTORIES[STORY_IND]].data
    tr_times = wordseqs[ALLSTORIES[STORY_IND]].tr_times

    print("TR", len(tr_times), tr_times)

    chunks = np.split(range(0, len(words)), wordseqs[ALLSTORIES[STORY_IND]].split_inds)
    
    # Compute chunk-level differences, skip empty chunks
    chunk_differences_same = []
    chunk_differences_att = []
    chunk_differences_base = []
    valid_chunks = []

    for chunk in chunks:
        if len(chunk) > 0:  # Check if the chunk is not empty
            valid_chunks.append(chunk)
            chunk_differences_same.append(np.mean([same[i] - ((att[i] + base[i]) / 2) for i in range(chunk[0], chunk[-1]+1)]))
            chunk_differences_att.append(np.mean([att[i] - ((same[i] + base[i]) / 2) for i in range(chunk[0], chunk[-1]+1)]))
            chunk_differences_base.append(np.mean([base[i] - ((same[i] + att[i]) / 2) for i in range(chunk[0], chunk[-1]+1)]))

    # Get the top 30 chunks for each category
    top_30_same_chunks = np.argsort(chunk_differences_same)[-30:]
    top_30_att_chunks = np.argsort(chunk_differences_att)[-30:]
    top_30_base_chunks = np.argsort(chunk_differences_base)[-30:]

    val_list = [3] * len(word_list)  # Initialize all to 'No Significant Difference'

    # Mark the chunks in each category, ensuring the difference is positive
    for chunk_idx in top_30_same_chunks:
        if chunk_differences_same[chunk_idx] > 0:
            for i in range(valid_chunks[chunk_idx][0], valid_chunks[chunk_idx][-1]+1):
                val_list[i] = 0

    for chunk_idx in top_30_att_chunks:
        if chunk_differences_att[chunk_idx] > 0:
            for i in range(valid_chunks[chunk_idx][0], valid_chunks[chunk_idx][-1]+1):
                val_list[i] = 1

    for chunk_idx in top_30_base_chunks:
        if chunk_differences_base[chunk_idx] > 0:
            for i in range(valid_chunks[chunk_idx][0], valid_chunks[chunk_idx][-1]+1):
                val_list[i] = 2
                
    # Define number of rows and columns for words
    num_rows = 85
    words_per_row = len(word_list) // num_rows

    fig, ax = plt.subplots(figsize=(20, 30))  # Increased figure height

    for i, word in enumerate(word_list):
        row = i // words_per_row
        col = i % words_per_row
        x_position = col / words_per_row
        y_position = 1 - (row / num_rows)
        color_index = val_list[i]
        ax.text(x_position, y_position, word, color='black', fontsize=12,
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(facecolor=colors[color_index], edgecolor='none', boxstyle='round,pad=0.1'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.axis('off')

    legend_patches = [mpatches.Patch(color=colors[i], label=color_names[i]) for i in range(len(colors))]
    legend = ax.legend(handles=legend_patches, title='Legend', loc='upper left', bbox_to_anchor=(1, 1), fontsize='large')

    plt.title('Word Plot with Color-Coded Values', pad=20)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if save:
        plt.savefig(save)

    plt.show()

    # Initialize lists to store phrases
    list_same, list_att, list_base = [], [], []
    temp_phrase = word_list[0]

    # Initialize current best category that is significantly better
    current_best = val_list[0]

    # Loop through val_list and word_list
    for i in range(1, len(val_list)):
        if val_list[i] == current_best:
            temp_phrase += " " + word_list[i]
        else:
            if current_best == 0:
                list_same.append(temp_phrase)
            elif current_best == 1:
                list_att.append(temp_phrase)
            elif current_best == 2:
                list_base.append(temp_phrase)

            # Reset for new phrase
            current_best = val_list[i]
            temp_phrase = word_list[i]

    # Append the last phrase to the appropriate list
    if current_best == 0:
        list_same.append(temp_phrase)
    elif current_best == 1:
        list_att.append(temp_phrase)
    elif current_best == 2:
        list_base.append(temp_phrase)

    # Print or return the lists
    print(color_names[0], list_same)
    print(color_names[1], list_att)
    print(color_names[2], list_base)










def plot_top_pred(PATH1, PATH2, PATH3, emb1, emb2, emb3, sig, save="infini_900"):
    test_stories = [CUR_STORY]

    print("load")
    data = np.load(PATH1 + 'weights.npz')
    wt1 = data[data.files[0]]

    with open(emb1, 'rb') as f:
        feature1 = pickle.load(f)

	# Delayed stimulus
    delPstim1 = apply_zscore_and_hrf(test_stories, feature1, 5, 4)

    pred1 = np.dot(delPstim1, wt1)

    data = np.load(PATH2 + 'weights.npz')
    wt2 = data[data.files[0]]

    with open(emb2, 'rb') as f:
        feature2 = pickle.load(f)

	# Delayed stimulus
    delPstim2 = apply_zscore_and_hrf(test_stories, feature2, 5, 4)

    pred2 = np.dot(delPstim2, wt2)

    data = np.load(PATH3 + 'weights.npz')
    wt3 = data[data.files[0]]

    with open(emb3, 'rb') as f:
        feature3 = pickle.load(f)

	# Delayed stimulus
    delPstim3 = apply_zscore_and_hrf(test_stories, feature3, 5, 4)

    pred3 = np.dot(delPstim3, wt3)

    # All the predictoins are loaded
    
    wordseqs = get_story_wordseqs(test_stories)
    words = wordseqs[test_stories[0]].data
    tr_times = wordseqs[test_stories[0]].tr_times

    print("TR", len(tr_times), tr_times)

    chunks = np.split(words, wordseqs[test_stories[0]].split_inds)

    print("load")
    data = np.load(PATH1 + 'corrs.npz')
    corr = data[data.files[0]]
    selvox = np.argmax(corr)
    print("done")

    # pred1 = np.mean(pred1, axis=1)
    # pred2 = np.mean(pred2, axis=1)
    # pred3 = np.mean(pred3, axis=1)

    pred1 = pred1[:, selvox]
    pred2 = pred2[:, selvox]
    pred3 = pred3[:, selvox]

    zeros_at_start = np.zeros(10)  # Creates a NumPy array of 10 zeros
    # Add zeros to the end
    zeros_at_end = np.zeros(5)  # Creates a NumPy array of 5 zeros

    extended_list = np.concatenate([zeros_at_start, pred1])
    pred1 = np.concatenate([extended_list, zeros_at_end])

    extended_list = np.concatenate([zeros_at_start, pred2])
    pred2 = np.concatenate([extended_list, zeros_at_end])

    extended_list = np.concatenate([zeros_at_start, pred3])
    pred3 = np.concatenate([extended_list, zeros_at_end])

    word_list = []
    pred1_list = []
    pred2_list = []
    pred3_list = []

    for i, chunk in enumerate(chunks):
        for word in chunk:
            word_list.append(word)
            pred1_list.append(pred1[i])
            pred2_list.append(pred2[i])
            pred3_list.append(pred3[i])
    
    color_names = ['Repeat Phrase', 'Repeat Attended-to Sections', 'No Prefix', 'No Significant Difference']
    colors = [ 'lightgreen', 'lightcoral', 'lightblue', 'white']  # Light colors for 0, 1, 2, 3, and no significant difference

    combined_list = pred1_list + pred2_list + pred3_list

    combined_array = np.array(combined_list)

    std_dev = np.std(combined_array)

    if sig:
        color_names = ['Repeat Phrase', 'Repeat Attended-to Sections', 'No Prefix', 'No Significant Difference']
        colors = ['lightgreen', 'lightcoral', 'lightblue', 'white']  # Light colors for 0, 1, 2, 3, and no significant difference

        # Compute global differences and their statistics
        global_differences = [abs(p1 - p2) for p1, p2 in zip(pred1_list, pred2_list)] + \
                            [abs(p1 - p3) for p1, p3 in zip(pred1_list, pred3_list)] + \
                            [abs(p2 - p3) for p2, p3 in zip(pred2_list, pred3_list)]
        global_mean_diff = np.mean(global_differences)
        global_std_dev = np.std(global_differences)

        # Generate val_list where each value determines the color of the corresponding word
        val_list = []
        for i in range(len(pred1_list)):
            res = [pred1_list[i], pred2_list[i], pred3_list[i]]
            min_val = min(res)
            ind = res.index(min_val)
            differences = [x - min_val for x in res]

            other_diffs = [x for j, x in enumerate(differences) if j != ind]  # Exclude the minimum itself
            avg_diff = sum(other_diffs) / len(other_diffs)

            # Check if the average of the differences is significantly greater than the global average
            if avg_diff > global_mean_diff + global_std_dev:
                val_list.append(ind)  # Significantly better
            else:
                val_list.append(3)  # No significant difference
    else:
        #color_names = ['Repeat Phrase', 'Repeat Attended-to Sections', 'No Prefix']
        color_names = ['Infinigram', 'Infinigram With Context', 'Incontext infinigram']
        colors = [ 'lightgreen', 'lightcoral', 'lightblue']  # Light colors for 0, 1, 2, 3, and no significant difference
        # Generate val_list where each value determines the color of the corresponding word
        val_list = []
        for i in range(len(pred1_list)):
            pre = [pred1_list[i], pred2_list[i], pred3_list[i]]
            pre = [abs(p) for p in pre]
            max_val = max(pre)
            ind = pre.index(max_val)

            val_list.append(ind)
    
    # Define number of rows and columns for words
    num_rows = 85
    words_per_row = len(word_list) // num_rows

    fig, ax = plt.subplots(figsize=(20, 30))  # Increased figure height

    for i, word in enumerate(word_list):
        row = i // words_per_row
        col = i % words_per_row
        x_position = col / words_per_row
        y_position = 1 - (row / num_rows)
        color_index = val_list[i]
        ax.text(x_position, y_position, word, color='black', fontsize=12,
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(facecolor=colors[color_index], edgecolor='none', boxstyle='round,pad=0.1'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.axis('off')

    legend_patches = [mpatches.Patch(color=colors[i], label=color_names[i]) for i in range(len(colors))]
    legend = ax.legend(handles=legend_patches, title='Legend', loc='upper left', bbox_to_anchor=(1, 1), fontsize='large')

    plt.title('Word Plot with Color-Coded Values', pad=20)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(save)

    plt.show()

    # Initialize lists to store phrases
    list_same, list_att, list_base = [], [], []
    temp_phrase = word_list[0]

    # Initialize current best category that is significantly better
    current_best = val_list[0]

    # Loop through val_list and word_list
    for i in range(1, len(val_list)):
        if val_list[i] == current_best:
            temp_phrase += " " + word_list[i]
        else:
            if current_best == 0:
                list_same.append(temp_phrase)
            elif current_best == 1:
                list_att.append(temp_phrase)
            elif current_best == 2:
                list_base.append(temp_phrase)

            # Reset for new phrase
            current_best = val_list[i]
            temp_phrase = word_list[i]

    # Append the last phrase to the appropriate list
    if current_best == 0:
        list_same.append(temp_phrase)
    elif current_best == 1:
        list_att.append(temp_phrase)
    elif current_best == 2:
        list_base.append(temp_phrase)

    # Print or return the lists
    print("Same:", list_same)
    print("Att:", list_att)
    print("Base:", list_base)

emb_dim = str(900)

PATH1 = "/home/t-smantena/internblobdl/results/pca_tr_infini_900x_0.8k/UTS03/"
PATH2 = "/home/t-smantena/internblobdl/results/pca_tr_infini_w_cont_900x_0.8k/UTS03/"
PATH3 = "/home/t-smantena/internblobdl/results/pca_tr_incont_infini_900x_0.8k/UTS03/"

emb1 = "/home/t-smantena/internblobdl/results/pca_tr_infini_900x_0.8k/emb/vecs.pkl"
emb2 = "/home/t-smantena/internblobdl/results/pca_tr_infini_w_cont_900x_0.8k/emb/vecs.pkl"
emb3 = "/home/t-smantena/internblobdl/results/pca_tr_incont_infini_900x_0.8k/emb/vecs.pkl"    

inf_resid, word_list = plot_residual(PATH1, emb1, "inf_avg")
inf_w_cont_resid, word_list = plot_residual(PATH2, emb2, "inf_w_cont_avg")
incont_inf_resid, word_list = plot_residual(PATH3, emb3, "incont_inf_avg")

plot_best(inf_resid, inf_w_cont_resid, incont_inf_resid, word_list, "inf_best" + emb_dim)

# plot_top_pred(PATH1, PATH2, PATH3, emb1, emb2, emb3, True)


# save_features_infini()

# same_resid, word_list = plot_residual("/home/t-smantena/internblobdl/results/llama_w_prefix_same_64k/UTS03/", "/home/t-smantena/internblobdl/llama_w_prefix_same.pkl", "same_map_avg")
# att_resid, word_list = plot_residual("/home/t-smantena/internblobdl/results/llama_w_prefix_att_32k/UTS03/", "/home/t-smantena/internblobdl/llama_w_prefix_att.pkl", "att_map_avg")
# base_resid, word_list = plot_residual("/home/t-smantena/internblobdl/results/llama_64k/UTS03/", "/home/t-smantena/internblobdl/llama.pkl", "base_map_avg")

# plot_best(same_resid, att_resid, base_resid, word_list, "best")

# same_resid = np.array(same_resid)
# att_resid = np.array(att_resid)
# base_resid = np.array(base_resid)

# print(same_resid[10:40])

# same_att_diff = same_resid - att_resid

# att_base_diff = att_resid - base_resid

# base_same_diff = base_resid - same_resid

# print("base and diff sum", np.sum(base_same_diff))

# same_att_diff = same_att_diff.tolist()
# att_base_diff = att_base_diff.tolist()
# base_same_diff = base_same_diff.tolist()


# word_plot(same_att_diff, word_list, "same_att_diff")
# word_plot(att_base_diff, word_list, "att_base_diff")
# word_plot(base_same_diff, word_list, "base_same_diff")

# Need to change out all stories indices and the plot_best story name when swithcing out the stories