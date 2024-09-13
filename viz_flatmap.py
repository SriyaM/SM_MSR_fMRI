import cortex
# Plot one slice of the mask that was used to select the voxels

from matplotlib.pyplot import figure, cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
from os.path import dirname
import numpy as np
from matplotlib.cm import ScalarMappable
import tables
import seaborn as sns

print(cortex.database.default_filestore)


OUTPUT_DIR = "/home/t-smantena/deep-fMRI-dataset/results/eng1000/UTS03/"

print("load")
data = np.load(OUTPUT_DIR + 'corrs.npz')
voxcorrs = data[data.files[0]]
print("done")

# def _save_flatmap(vals, subject, fname_save, clab=None, with_rois=False, cmap='RdBu', with_borders=False):
#     vabs = max(np.abs(vals))

#     # cmap = sns.diverging_palette(12, 210, as_cmap=True)
#     # cmap = sns.diverging_palette(16, 240, as_cmap=True)

#     vol = cortex.Volume(
#         vals, 'UT' + subject, xfmname=f'UT{subject}_auto', vmin=-vabs, vmax=vabs, cmap=cmap)

#     cortex.quickshow(vol,
#                      with_rois=with_rois,
#                      with_labels=False,
#                      with_borders=with_borders,
#                      with_colorbar=clab == None,  # if not None, save separate cbar
#                      )
#     os.makedirs(dirname(fname_save), exist_ok=True)
#     plt.savefig(fname_save)
#     plt.close()

#     # save cbar
#     norm = Normalize(vmin=-vabs, vmax=vabs)
#     # need to invert this to match above
#     sm = ScalarMappable(norm=norm, cmap=cmap)
#     sm.set_array([])
#     fig, ax = plt.subplots(figsize=(5, 0.35))
#     cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
#     if clab:
#         cbar.set_label(clab, fontsize='x-large')
#         plt.savefig(fname_save.replace('flatmap.pdf',
#                     'cbar.pdf'), bbox_inches='tight')
#     plt.close()

vabs = max(np.abs(voxcorrs))

cmap = sns.diverging_palette(12, 210, as_cmap=True)
cmap = sns.diverging_palette(16, 240, as_cmap=True)

vol = cortex.Volume(voxcorrs, 'S1', xfmname='UTS_auto', vmin=-vabs, vmax=vabs, cmap=cmap)

# resptf = tables.open_file("/home/t-smantena/internblobdl/data/fmri-responses.hf5")
# zRresp = resptf.root.zRresp.read()
# zPresp = resptf.root.zPresp.read()
# mask = resptf.root.mask.read()

# # Plot the mask slice
# f = plt.figure()
# ax = f.add_subplot(1, 1, 1)
# ax.matshow(mask[16], interpolation="nearest", cmap=cm.gray)  # show the 17th slice of the mask
# plt.savefig("mask_slice.png")  # Save the figure
# print("Mask slice saved as 'mask_slice.png'")

# # Plot the mosaic
# f = plt.figure(figsize=(10, 10))
# cortex.mosaic(mask, cmap=cm.gray, interpolation="nearest")
# plt.savefig("mask_mosaic.png")  # Save the mosaic
# print("Mosaic saved as 'mask_mosaic.png'")

# print(len(mask))
# print(voxcorrs.shape)
# # Initialize corrvolume with the same shape as mask
# corrvolume = np.zeros(mask.shape)

# # Ensure voxcorrs has the same shape as corrvolume
# # (voxcorrs should be the same size as corrvolume)
# assert voxcorrs.shape == corrvolume.shape, "voxcorrs and corrvolume must have the same shape."

# # Assign values from voxcorrs to corrvolume only at the positions where mask > 0
# corrvolume[mask > 0] = voxcorrs[mask > 0]

# f = plt.figure(figsize=(10, 10))
# cortex.mosaic(corrvolume, vmin=0, vmax=0.5, cmap=cm.hot)  # Plot the mosaic of correlations
# plt.savefig("corr_mosaic.png")  # Save the mosaic of correlations
# print("Correlation mosaic saved as 'corr_mosaic.png'")

# # Display the last figure if in an interactive environment
# plt.show()

# corrvolume = np.zeros(mask.shape)

# print(corrvolume[mask > 0].shape)  # Shape of selected region in corrvolume
# print(voxcorrs.shape)              # Shape of voxcorrs

# corrvolume[mask>0] = voxcorrs

# f = figure(figsize=(10,10))
# cortex.mosaic(corrvolume, vmin=0, vmax=0.5, cmap=cm.hot)

# corrvol = cortex.Volume(voxcorrs, "S1", "fullhead", mask=mask, vmin=0, vmax=0.5, cmap='hot')

# # cortex.webshow(corrvol, port=8889, open_browser=False)

# # server_ip = "10.8.161.240"  # Or use "172.17.0.1" if needed
# # port = 8889  # Replace with the correct port if necessary

# # print(f"Click here for viewer: http://{server_ip}:{port}")

# cortex.quickshow(corrvol, with_rois=False, with_labels=False)

