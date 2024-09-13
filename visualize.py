import numpy as np
#import cortex
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


# Sample data repeated to increase word count
words = ['hello', 'world', 'this', 'is', 'a', 'test'] * 10
residuals = [0.1, 0.2, 0.3, -0.4, 0.5, -0.6] * 10  # Example residual values

# Normalize residuals for color mapping
norm = plt.Normalize(min(residuals), max(residuals))
colors = plt.cm.coolwarm(norm(residuals))  # Using the coolwarm colormap

# Define number of rows and columns for words
num_rows = 2
words_per_row = len(words) // num_rows

plt.figure(figsize=(20, 5))  # Increase figure size to better accommodate words
ax = plt.gca()

# Adjust text positions and plot in multiple lines
for i, (word, color) in enumerate(zip(words, colors)):
    row = i // words_per_row
    col = i % words_per_row
    x_position = col / words_per_row
    y_position = 1 - (row / num_rows)
    ax.text(x_position, y_position, word, color=color, fontsize=12, ha='center', transform=ax.transAxes)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.axis('off')
plt.tight_layout()  # Adjust layout to make room for all elements
plt.show()

plt.savefig("Example_output")


def get_overall_and_top_mean(OUTPUT_DIR):

    print(OUTPUT_DIR)

    # fuzzy_induction_head 
    #fuzzy_induction_head_temp_2 
    # fuzzy_induction_head_temp_4
    #  eng1000_w_fuzzy_distribution

    #data = np.load(OUTPUT_DIR + 'bscorrs.npz')
    #bscorrs = data[data.files[0]]

    print("load")
    data = np.load(OUTPUT_DIR + 'corrs.npz')
    corr = data[data.files[0]]
    print("done")


    """
    data = np.load(OUTPUT_DIR + 'valinds.npz')
    valinds = data[data.files[0]]

    data = np.load(OUTPUT_DIR + 'valphas.npz')
    alphas = data[data.files[0]]

    data = np.load(OUTPUT_DIR + 'weights.npz')
    wt = data[data.files[0]]"""

    abs_corr = np.abs(corr)
    
    text = "mean: " + str(np.mean(abs_corr)) + \
        "\nstd: " + str(np.std(abs_corr)) + \
        "\nmin: " + str(np.min(abs_corr)) + \
        "\nmax: " + str(np.max(abs_corr))
    
    # Calculate the mean of the top 5% of correlations
    top_5_percent_threshold = np.percentile(abs_corr, 95)
    top_5_percent_mean = np.mean(abs_corr[abs_corr >= top_5_percent_threshold])
    
    text += "\nmean of top 5%: " + str(top_5_percent_mean)
    
    print(text)


def alphas_viz(OUTPUT_DIR):
    data = np.load(OUTPUT_DIR + 'valphas.npz')
    alphas = data[data.files[0]]


    # Plot the contents in a histogram
    plt.figure(figsize=(10, 6))
    plt.hist(alphas, bins=30, alpha=0.75, color='blue', edgecolor='black')
    plt.title('Histogram of Alphas Induction')
    plt.xlabel('Alpha Values')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    output_path = 'histogram_of_alphas.png'
    plt.savefig(output_path)


def get_combined_mean(OUTPUT_DIR1, OUTPUT_DIR2):
    """Calculate the overall average and top 5% mean of absolute correlations from two directories."""

    print(f"Loading data from {OUTPUT_DIR1} and {OUTPUT_DIR2}")

    # Load correlation data from both directories
    data1 = np.load(OUTPUT_DIR1 + 'corrs.npz')
    corr1 = data1[data1.files[0]]
    data2 = np.load(OUTPUT_DIR2 + 'corrs.npz')
    corr2 = data2[data2.files[0]]

    print("Data loaded successfully")

    # Combine the correlation data
    combined_corr = np.concatenate((corr1, corr2), axis=None)

    # Calculate the absolute values of the correlations
    abs_combined_corr = np.abs(combined_corr)

    # Calculate statistics
    overall_mean = np.mean(abs_combined_corr)
    overall_std = np.std(abs_combined_corr)
    overall_min = np.min(abs_combined_corr)
    overall_max = np.max(abs_combined_corr)

    # Calculate the mean of the top 5% of correlations
    top_5_percent_threshold = np.percentile(abs_combined_corr, 95)
    top_5_percent_mean = np.mean(abs_combined_corr[abs_combined_corr >= top_5_percent_threshold])

    # Prepare the output text
    text = (
        f"Overall mean: {overall_mean}\n"
        f"Overall std: {overall_std}\n"
        f"Overall min: {overall_min}\n"
        f"Overall max: {overall_max}\n"
        f"Mean of top 5%: {top_5_percent_mean}"
    )

    print(text)

get_overall_and_top_mean("/home/t-smantena/internblobdl/results/pca_tr_incont_infini_10_900x_1.0k/UTS03/")

get_overall_and_top_mean("/home/t-smantena/internblobdl/results/eng1000_10_/UTS03/")

get_overall_and_top_mean("/home/t-smantena/internblobdl/results/pca_tr_incont_infini_10_900x_1.0k/UTS03/")

get_overall_and_top_mean("/home/t-smantena/internblobdl/results/pca_tr_incont_fuzzy_llama_900x_1.0k/UTS03/")

get_overall_and_top_mean("/home/t-smantena/internblobdl/results/pca_tr_random_1.0k/UTS03/")

get_overall_and_top_mean("/home/t-smantena/internblobdl/results/pca_tr_exact_1.0k/UTS03/")

get_overall_and_top_mean("/home/t-smantena/internblobdl/results/incont_infinigram_p_900x/UTS03/")

get_overall_and_top_mean("/home/t-smantena/internblobdl/results/incont_infinigram_p_450x/UTS03/")

get_overall_and_top_mean("/home/t-smantena/internblobdl/results/incont_infinigram_p_200x/UTS03/")



get_overall_and_top_mean("/home/t-smantena/internblobdl/results/eng_window_2048x_16k/UTS03/")

get_combined_mean(
    "/home/t-smantena/internblobdl/results/llama_ind_only_1024x_8k/UTS03/",
    "/home/t-smantena/internblobdl/results/llama_non_ind_only_1024x_8k/UTS03/"

    
)

get_combined_mean(
    "/home/t-smantena/internblobdl/results/llama_ind_only_1024x_32k/UTS03/",
    "/home/t-smantena/internblobdl/results/llama_non_ind_only_1024x_32k/UTS03/"
)





get_overall_and_top_mean("/home/t-smantena/internblobdl/results/llama_w_prefix_att_64k/UTS03/")












get_overall_and_top_mean("/home/t-smantena/internblobdl/results/llama_non_ind_only_1024x_16k/UTS03/")

get_overall_and_top_mean("/home/t-smantena/internblobdl/results/eng_f_ind_zero_b_64x_1k/UTS03/")

get_overall_and_top_mean("/home/t-smantena/internblobdl/results/eng_f_ind_zero_b_128x_1k/UTS03/")

get_overall_and_top_mean("/home/t-smantena/internblobdl/results/eng_f_ind_zero_b_512x_1k/UTS03/")

get_overall_and_top_mean("/home/t-smantena/internblobdl/results/eng_f_ind_zero_b_1024x_1k/UTS03/")

get_overall_and_top_mean("/home/t-smantena/internblobdl/results/eng_f_ind_zero_b_2048x_1k/UTS03/")








# get_overall_and_top_mean("/home/t-smantena/internblobdl/results/eng_k_window_avg_256x_8k/UTS03/")

# get_overall_and_top_mean("/home/t-smantena/internblobdl/results/eng_k_window_avg_512x_8k/UTS03/")

# get_overall_and_top_mean("/home/t-smantena/internblobdl/results/eng_k_window_avg_1024x_8k/UTS03/")

# get_overall_and_top_mean("/home/t-smantena/internblobdl/results/eng_k_window_avg_512x_16k/UTS03/")

# get_overall_and_top_mean("/home/t-smantena/internblobdl/results/eng_k_window_avg_1024_16k/UTS03/")

# get_overall_and_top_mean("/home/t-smantena/internblobdl/results/eng_k_window_avg_2048x_16k/UTS03/")

# get_overall_and_top_mean("/home/t-smantena/internblobdl/results/eng_k_window_avg_1024x_32k/UTS03/")

# get_overall_and_top_mean("/home/t-smantena/internblobdl/results/eng_k_window_avg_2048_32k/UTS03/")

get_overall_and_top_mean("/home/t-smantena/internblobdl/results/eng_k_window_avg_4096x_32k/UTS03/")



#print("2")
#get_overall_and_top_mean("/home/t-smantena/internblobdl/results/ind_1024_secs/UTS03/")


#print("4")
#get_overall_and_top_mean("/home/t-smantena/internblobdl/results/e_perp4/UTS03/")


# print("8")
# get_overall_and_top_mean("/home/t-smantena/deep-fMRI-dataset/results/llama_window_1024x_16k/UTS03/")


# print("16")
# get_overall_and_top_mean("/home/t-smantena/internblobdl/results/llama_k_window_avg_b_1024x_16k/UTS03/")

# print("32")
# get_overall_and_top_mean("/home/t-smantena/internblobdl/results/llama_k_window_avg_512x_16k/UTS03/")

# print("8")
# get_overall_and_top_mean("/home/t-smantena/internblobdl/results/llama_k_window_b_2048x_16k/UTS03/")

# print("128")
# get_overall_and_top_mean("/home/t-smantena/internblobdl/results/llama_window_avg_512x_8k/UTS03/")

# print("512")
# get_overall_and_top_mean("/home/t-smantena/internblobdl/results/llama_window_avg_b_1024x_8k/UTS03/")

# print("1024")
# get_overall_and_top_mean("/home/t-smantena/internblobdl/results/sim_128k_aw/UTS03/")
    

# Graphing the predicted and the real responses side by side
"""
f = plt.figure(figsize=(15,5))
ax = f.add_subplot(1,1,1)
pred = np.dot(delPstim, wt)

selvox = 20710 # a decent voxel

realresp = ax.plot(zPresp[:,selvox], 'k')[0]
predresp = ax.plot(pred[:,selvox], 'r')[0]

ax.set_xlim(0, 291)
ax.set_xlabel("Time (fMRI time points)")

ax.legend((realresp, predresp), ("Actual response", "Predicted response"))

voxcorrs = np.zeros((zPresp.shape[1],)) # create zero-filled array to hold correlations
for vi in range(zPresp.shape[1]):
    voxcorrs[vi] = np.corrcoef(zPresp[:,vi], pred[:,vi])[0,1]
print (voxcorrs)

# Plot histogram of correlations
f = plt.figure(figsize=(8,8))
ax = f.add_subplot(1,1,1)
ax.hist(voxcorrs, 100) # histogram correlations with 100 bins
ax.set_xlabel("Correlati
ax.set_ylabel("Num. voxels")

# Plot mosaic of correlations
corrvolume = np.zeros(mask.shape)
corrvolume[mask>0] = voxcorrs

f = plt.figure(figsize=(10,10))
cortex.mosaic(corrvolume, vmin=0, vmax=0.5, cmap=cm.hot)


f = plt.figure(figsize=(8,8))
ax = f.add_subplot(1,1,1)
ax.hist(voxcorrs, 100) # histogram correlations with 100 bins
ax.set_xlabel("Correlation")
ax.set_ylabel("Num. voxels")


# Plot correlations on cortex
corrvol = cortex.Volume(corr, "S1", "fullhead", mask=mask, vmin=0, vmax=0.5, cmap='hot')
cortex.webshow(corrvol, port=8889, open_browser=False)

# view the correlations as a flat map
cortex.quickshow(corrvol, with_rois=False, with_labels=False)"""


"""
# Then let's predict responses by taking the dot product of the weights and stim
pred = np.dot(delPstim, wt)
print ("pred has shape: ", pred.shape)

# Visualizes the predicted and actual responses on the same scale
f = figure(figsize=(15,5))
ax = f.add_subplot(1,1,1)

selvox = 20710 # a good voxel

realresp = ax.plot(zPresp[:,selvox], 'k')[0]
predresp = ax.plot(zscore(pred[:,selvox]), 'r')[0]

ax.set_xlim(0, 291)
ax.set_xlabel("Time (fMRI time points)")

ax.legend((realresp, predresp), ("Actual response", "Predicted response (scaled)"));

# Compute correlation between single predicted and actual response
# (np.corrcoef returns a correlation matrix; pull out the element [0,1] to get 
# correlation between the two vectors)
voxcorr = np.corrcoef(zPresp[:,selvox], pred[:,selvox])[0,1]
print ("Correlation between predicted and actual responses for voxel %d: %f" % (selvox, voxcorr))

# Computing the correlation for all voxels
voxcorrs = np.zeros((zPresp.shape[1],)) # create zero-filled array to hold correlations
for vi in range(zPresp.shape[1]):
    voxcorrs[vi] = np.corrcoef(zPresp[:,vi], pred[:,vi])[0,1]
print (voxcorrs)


# Plot mosaic of correlations
corrvolume = np.zeros(mask.shape)
corrvolume[mask>0] = voxcorrs

f = figure(figsize=(10,10))
cortex.mosaic(corrvolume, vmin=0, vmax=0.5, cmap=cm.hot);


# Plot correlations on cortex
import cortex
corrvol = cortex.Volume(corr, "S1", "fullhead", mask=mask, vmin=0, vmax=0.5, cmap='hot')
cortex.webshow(corrvol, port=8889, open_browser=False)

# view the correlations as a flat map
cortex.quickshow(corrvol, with_rois=False, with_labels=False)"""
