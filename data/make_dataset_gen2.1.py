from PRNN.utils import get_system_and_backend
get_system_and_backend()

import h5py
import time
import numpy as np
import random
import torch
import os
import torchvision.transforms

from tqdm import tqdm
from data.utils import poisson_sampling_batch, make_E_fields, rgb_to_phase, crop_and_resize
from PRNN.pipeline.image_data import make_interferogram_frames, scale_interferogram_frames
from PRNN.pipeline.PhaseImages import frames_to_svd_torch
from PRNN.visualization.figure_utils import *
from PRNN.visualization.visualize import plot_frames

### PARAMETERS ###
ndata       = 8000 # number of different training frame sets to include in a data set
val_split   = 0.
nx          = 64 # X pixels
ny          = nx # Y pixels
nframes     = 32*2
nbar_signal = (1e2, 1e5)
nbar_bkgrnd = (0, 0)
color_balance = [0.6, 0.2, 0.2]
crop_frac   = 1
sigma_X     = 5
sigma_Y     = 5
vis         = 1
poisson     = True
svd         = True
save        = True
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device      = 'cpu'

# Save info
basepath = "raw/"
# filepath = 'flowers_n%i_npix%i_20231221.h5' % (ndata, nx)
# filepath = 'flowers_pruned_n%i_npix%i_Eigen_20240110_test12.h5' % (ndata, nx)
filepath = 'flowers102_n%i_npix%i_Eigen_20240119.h5' % (ndata, nx)
# filepath = 'mnist_n%i_npix%i.h5' % (ndata, nx)

masks_folder = 'flowers102'
# masks_folder = 'flowers_more_pruned'
filenames = os.listdir(os.path.join('masks', masks_folder))
# filenames.sort()
random.seed(666)
random.shuffle(filenames)

E1, E2 = make_E_fields(nx, ny, sigma_X, sigma_Y, device)

# Reserve the first 90% of files for training data, 10% for validation data
nfiles = int(len(filenames))
ntrain = int(nfiles * (1 - val_split))
ndata_train = int(ndata * (1 - val_split))
ndata_val = ndata - ndata_train
train_indices = list(range(0, ntrain))
val_indices = list(range(ntrain, len(filenames)))

#### Data generation loop for training set ####
truths_data = torch.zeros((ndata, nx, ny), device='cpu')
inputs_data = torch.zeros((ndata, nframes, nx, ny), device='cpu')
if svd:
    svd_data    = torch.zeros((ndata, 2, nx, ny), device='cpu')

""" Two loops are used to generate training and validation data. Because we have a finite number of images,
the images are re-used with some random re-sizing and cropping. The separate loops are used to avoid similar
crops of the training images ending up in the validation data. """

signal_levels = []

tic1 = time.time()

# raise RuntimeError

for d in tqdm(range(0, ndata_train), desc='Generating training images..'):
    # idx = random.randint(0, len(train_indices) - 1)
    idx = d
    mask = filenames[train_indices[idx]]
    filename = os.path.join('masks', masks_folder, mask)

    y = rgb_to_phase(filename, color_balance=color_balance)
    y = crop_and_resize(y, nx, ny, crop_frac=crop_frac)
    y = torch.tensor(y).to(device)

    signal_level = torch.randint(low=int(nbar_signal[0]), high=int(nbar_signal[1]) + 1, size=(1,), device=device)
    bkgrnd_level = torch.randint(low=int(nbar_bkgrnd[0]), high=int(nbar_bkgrnd[1]) + 1, size=(1,), device=device)

    x = make_interferogram_frames(y, E1, E2, vis, nframes, device)
    x = scale_interferogram_frames(x, signal_level, bkgrnd_level)

    truths_data[d, :, :] = y
    inputs_data[d, :, :, :] = x

    signal_levels.append(signal_level.cpu())

for d in tqdm(range(0, ndata_val), desc='Generating validation images..'):
    idx = random.randint(0, len(val_indices) - 1)
    mask = filenames[val_indices[idx]]
    filename = os.path.join('masks', masks_folder, mask)

    y = rgb_to_phase(filename, color_balance=color_balance)
    y = crop_and_resize(y, nx, ny, crop_frac=crop_frac)
    y = torch.tensor(y).to(device)

    signal_level = torch.randint(low=int(nbar_signal[0]), high=int(nbar_signal[1]) + 1, size=(1,), device=device)
    bkgrnd_level = torch.randint(low=int(nbar_bkgrnd[0]), high=int(nbar_bkgrnd[1]) + 1, size=(1,), device=device)

    x = make_interferogram_frames(y, E1, E2, vis, nframes, device)
    x = scale_interferogram_frames(x, signal_level, bkgrnd_level)

    truths_data[ndata_train + d, :, :] = y
    inputs_data[ndata_train + d, :, :, :] = x

print(f"Time elapsed for data generation: {time.time()-tic1:.4f} sec", end='\n')

# Poisson sampling
if poisson:
    tic = time.time()
    poisson_batch_size = 500
    inputs_data = poisson_sampling_batch(inputs_data, 500, device)
    print(f"Time elapsed for Poisson sampling mini-batch: {time.time()-tic:.4f} sec", end='\n')

# Calculate SVD
if svd:
    print('Calculating SVDs..', end='\n')
    for d, x in tqdm(enumerate(inputs_data)):
        # Calculate SVD from 32 random frames
        xsubset = x[torch.randperm(x.shape[0])][0:32]
        # phi1, phi2 = frames_to_svd(xsubset)
        phi1, phi2, zsin, zcos = frames_to_svd_torch(xsubset, device=device)

        phi1, phi2 = phi1.cpu(), phi2.cpu()
        zsin, zcos = zsin.cpu(), zcos.cpu()

        del xsubset

        # svd_data[d, 0, :, :] = phi1
        # svd_data[d, 1, :, :] = phi2
        svd_data[d, 0, :, :] = zsin
        svd_data[d, 1, :, :] = zcos

print('Total time = %.4f sec' % (time.time()-tic1))

#
# # Plotting to verify
# xtest = inputs_data[60, 0:9, :, :]
# plot_frames(xtest, 3, 3, cmap='gray')

# raise RuntimeError

# Move data to cpu
truths_data = truths_data.cpu()
inputs_data = inputs_data.cpu()
if svd:
    svd_data    = svd_data.cpu()

# Make header for logging
header_dict = {
    "masks": masks_folder,
    "val_split": val_split,
    "nbar_signal": nbar_signal,
    "nbar_bkgrnd": nbar_bkgrnd,
    "nframes": nframes,
    "color_balance": color_balance,
    "sigma_X": sigma_X,
    "sigma_Y": sigma_Y,
    "crop_scale": crop_frac
}

# Save to h5
if save:
    """ Save the data to .h5 file """
    with h5py.File(basepath+filepath, "a") as h5_data:
        h5_data["truths"] = np.array(truths_data)
        h5_data["inputs"] = np.array(inputs_data)
        h5_data["E1"] = np.array(E1.cpu())
        h5_data["E2"] = np.array(E2.cpu())
        h5_data["vis"] = np.array([vis], dtype=np.float32)
        h5_data["header"] = str(header_dict)
        if svd:
            h5_data["svd"] = svd_data

    h5_data.close()
    print('Saved to %s' % (basepath + filepath))