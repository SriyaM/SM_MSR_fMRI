import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, IncrementalPCA

# Assuming X_embeddings is your input data with shape (n_samples, 50000)

CUR_MODEL = "incont_dist_gpt"

TRAIN_STORIES = ['alternateithicatom', 'souls', 'avatar', 'legacy', 'odetostepfather', 'undertheinfluence', 'howtodraw', 'myfirstdaywiththeyankees', 'naked', 'life', 'stagefright', 'tildeath', 'fromboyhoodtofatherhood', 'sloth', 'exorcism', 'inamoment', 'theclosetthatateeverything', 'adventuresinsayingyes', 'buck', 'swimmingwithastronauts', 'thatthingonmyarm', 'eyespy', 'itsabox', 'hangtime']

TEST_STORIES = ['wheretheressmoke', 'haveyoumethimyet']

allstories = list(set(TRAIN_STORIES) | set(TEST_STORIES))

# Output directory -> pickle_file_path = os.path.join(save_dir, story, 'vectors.pkl') which has an array of vectors

for story in allstories:
    model = CUR_MODEL + "_"
    pickle_file_path = os.path.join("/home/t-smantena/internblobdl/results", model, "UTS03", story, "vectors.pkl")
    
    with open(pickle_file_path, 'rb') as file:
        # Load the NumPy array from the pickle file
        X_embeddings = pickle.load(file)
        print(story, "array before PCA", X_embeddings.shape)

    # Count number of NaN values before replacing them
    nan_count = np.isnan(X_embeddings).sum()
    print(f"Number of NaN values in {story} before replacement: {nan_count}")
    
    # Replace NaN values with zero
    X_embeddings = np.nan_to_num(X_embeddings, nan=0.0)
    
    # Verify that NaN values have been replaced
    nan_after = np.isnan(X_embeddings).sum()
    print(f"Number of NaN values in {story} after replacement: {nan_after}")

    n_components = 900  # Target dimensionality

    # Using IncrementalPCA for memory efficiency
    ipca = IncrementalPCA(n_components=n_components, batch_size=1000)
    X_ipca = ipca.fit_transform(X_embeddings)
    print(f"{story} array after PCA", X_ipca.shape)

    save_dir = os.path.join("/home/t-smantena/internblobdl/infini_pca_emb900", CUR_MODEL, story)
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, "vectors.pkl")
    print(f"Saving PCA transformed embeddings to {save_file}")
    
    with open(save_file, 'wb') as file:
        pickle.dump(X_ipca, file)


# Need to test the amount of time that it takes for a single story
# Need the format of the output to match the embeddings that I feed using the features function
# Run the inference and be preprared to present the output

# colors = ["navy", "turquoise", "darkorange"]

# for X_transformed, title in [(X_ipca, "Incremental PCA"), (X_pca, "PCA")]:
#     plt.figure(figsize=(8, 8))
#     for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
#         plt.scatter(
#             X_transformed[y == i, 0],
#             X_transformed[y == i, 1],
#             color=color,
#             lw=2,
#             label=target_name,
#         )

#     if "Incremental" in title:
#         err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
#         plt.title(title + " of iris dataset\nMean absolute unsigned error %.6f" % err)
#     else:
#         plt.title(title + " of iris dataset")
#     plt.legend(loc="best", shadow=False, scatterpoints=1)
#     plt.axis([-4, 4, -1.5, 1.5])

# plt.show()