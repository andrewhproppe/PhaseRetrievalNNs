import pickle
import torch
import matplotlib.pyplot as plt
from utils import norm_to_phase
from QIML.visualization.AP_figs_funcs import *
import matplotlib
matplotlib.use('TkAgg')

def combine_batches(filename, nbatch, rootpath="../data/predictions/"):
    # Initialize lists to store the variables
    phi_list, mse_list, ssim_list = [], [], []
    # Combine batches
    for i in range(nbatch):
        try:
            phi_temp, mse_temp, ssim_temp = pickle.load(open(f"{rootpath}{filename}_batch{i}.pickle", "rb"))
        except:
            phi_temp = pickle.load(open(f"{rootpath}{filename}_batch{i}.pickle", "rb"))
            mse_temp, ssim_temp = torch.tensor([0]), torch.tensor([0])
        phi_list.append(phi_temp)
        mse_list.append(mse_temp)
        ssim_list.append(ssim_temp)

    # Concatenate the lists to create the final tensors
    phi = torch.cat(phi_list, dim=0)
    mse = torch.cat(mse_list, dim=0)
    ssim = torch.cat(ssim_list, dim=0)

    return phi, mse, ssim


nbatch = 4
batch_size = 250
nframes    = 32

svd_phi, svd_mse, svd_ssim = combine_batches(
    f"SVD_bs{batch_size}_nf{nframes}",
    nbatch
)

nn_phi, nn_mse, nn_ssim = combine_batches(
    f"NN_bs{batch_size}_nf{nframes}",
    nbatch
)

true_phi, _, _ = combine_batches(
    f"True_bs{batch_size}_nf{nframes}",
    nbatch
)

# raise RuntimeError

# Histogram of errors
nbins = 200
nn_mse_histo = torch.histogram(nn_mse, bins=nbins)
svd_mse_histo = torch.histogram(svd_mse, bins=nbins)
fig, ax = plt.subplots(1, 2, figsize=(6, 2), dpi=150)
ax[0].bar(nn_mse_histo[1][:-1], nn_mse_histo[0], width=nn_mse_histo[1][1]-nn_mse_histo[1][0], linewidth=0.1, edgecolor='black')
ax[0].set_title("SRN3D")
ax[0].set_yscale('log')
ax[1].bar(svd_mse_histo[1][:-1], svd_mse_histo[0], width=svd_mse_histo[1][1]-svd_mse_histo[1][0], linewidth=0.1, edgecolor='black')
ax[1].set_yscale('log')
ax[1].set_title("SVD")
dress_fig(tight=True, xlabel='MSE', ylabel='Counts')

# # Plot results
# Y_true = norm_to_phase(true_phi)
#
# idx = 0
# cmap = 'hsv'
# fig, ax = plt.subplots(1, 3, figsize=(6, 2), dpi=150)
# ax[0].imshow(Y_true[idx], cmap=cmap)
# ax[0].set_title("True")
# ax[1].imshow(svd_phi[idx], cmap=cmap)
# ax[1].set_title("SVD")
# ax[2].imshow(nn_phi[idx], cmap=cmap)
# ax[2].set_title("SRN3D")
# dress_fig(tight=True, xlabel='$\it{x}$', ylabel='$\it{y}$')