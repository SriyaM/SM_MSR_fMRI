import numpy as np
import pickle
import matplotlib.pyplot as plt

def get_top_1_percent_indices(OUTPUT_DIR):
    # Load the correlation data
    print("load")
    data = np.load(OUTPUT_DIR + 'corrs.npz')
    corr = data[data.files[0]]
    print("done")

    # Calculate the absolute values of the correlations
    abs_corr = np.abs(corr)

    # Calculate the threshold for the top 1% correlations
    top_1_percent_threshold = np.percentile(abs_corr, 99.5)

    # Get the indices of the top 1% performing voxels
    top_1_percent_indices = np.where(abs_corr >= top_1_percent_threshold)[0]

    return top_1_percent_indices

def extract_weights(PATH, ind):
    # Load the weights data
    data = np.load(PATH + 'weights.npz')
    weights = data[data.files[0]]

    print("WEIGHTS_SHAPE", weights.shape)

    # Ensure the weights array has the expected shape
    if weights.shape[1] != 95556:
        raise ValueError("The weights array does not have the expected second dimension of 95556.")

    # Extract the vectors at the specified indices
    extracted_vectors = weights[:, ind]

    # Number of parts to split each vector into
    num_parts = 4

    # Check if the first dimension is divisible by the number of parts
    if extracted_vectors.shape[0] % num_parts != 0:
        raise ValueError("The first dimension of the weights array is not divisible by 4.")

    # Calculate the size of each part
    part_size = extracted_vectors.shape[0] // num_parts

    # Split and average each vector
    averaged_vectors = []
    for vec in extracted_vectors.T:
        # Split into 4 equal parts
        split_vecs = np.split(vec, num_parts)
        # Average the parts
        averaged_vec = np.mean(split_vecs, axis=0)
        averaged_vectors.append(averaged_vec)

    # Convert the list to a numpy array
    averaged_vectors = np.array(averaged_vectors)

    # Average all the x/4 dimension vectors together
    final_averaged_vector = np.mean(averaged_vectors, axis=0)

    return final_averaged_vector

# Paths to data
PATH_1 = "/home/t-smantena/internblobdl/results/llama_window_512x_4k/UTS03/"
PATH_2 = "/home/t-smantena/internblobdl/results/llama_2x_4k/UTS03/"

# Get the top 1% performing voxels indices
top_1_ind = get_top_1_percent_indices(PATH_1)

# Extract weights
weights_ind = extract_weights(PATH_1, top_1_ind)
print("weights1")

weights_b = extract_weights(PATH_2, top_1_ind)
print("weights2")

# Save the weights to a pickle file
with open('weights_data.pkl', 'wb') as f:
    pickle.dump({'weights_ind': weights_ind, 'weights_b': weights_b}, f)

print("Weights data saved to weights_data.pkl")

def load_weights_from_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['weights_ind'], data['weights_b']

# Load weights
weights_ind, weights_b = load_weights_from_pickle('weights_data.pkl')

# Debugging prints to check data
print("Loaded weights_ind shape:", weights_ind.shape)
print("Loaded weights_b shape:", weights_b.shape)

def plot_weight_differences(weights_ind, weights_b):
    # Calculate the differences between the first and last 4096 indices
    first_4096_weights_ind = weights_ind[:4096]
    last_4096_weights_ind = weights_ind[-4096:]

    first_4096_weights_b = weights_b[:4096]
    last_4096_weights_b = weights_b[-4096:]

    # Debugging prints to check data
    print("First 4096 weights_ind:", first_4096_weights_ind)
    print("Last 4096 weights_ind:", last_4096_weights_ind)
    print("First 4096 weights_b:", first_4096_weights_b)
    print("Last 4096 weights_b:", last_4096_weights_b)

    # Calculate the differences
    diff_weights_ind = first_4096_weights_ind - last_4096_weights_ind
    diff_weights_b = first_4096_weights_b - last_4096_weights_b

    # Debugging prints to check differences
    print("Difference weights_ind:", diff_weights_ind)
    print("Difference weights_b:", diff_weights_b)

    # Plot the differences
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.bar(range(len(diff_weights_ind)), diff_weights_ind, alpha=0.75, color='blue')
    plt.title('Difference in Weights (First 4096 - Last 4096) from induction')
    plt.xlabel('Index')
    plt.ylabel('Difference in Weight Value')

    plt.subplot(1, 2, 2)
    plt.bar(range(len(diff_weights_b)), diff_weights_b, alpha=0.75, color='green')
    plt.title('Difference in Weights (First 4096 - Last 4096) from repeated')
    plt.xlabel('Index')
    plt.ylabel('Difference in Weight Value')

    plt.tight_layout()

    # Save the plots to a file
    plt.savefig('w_diff.png')

    # Optionally, also show the plot
    plt.show()

# Plot the differences
plot_weight_differences(weights_ind, weights_b)