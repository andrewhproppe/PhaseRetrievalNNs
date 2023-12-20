from PRNN.utils import get_system_and_backend
get_system_and_backend()

import h5py
import time
import numpy as np
import random
import torch
import os
import torchvision.transforms

from matplotlib import pyplot as plt
from tqdm import tqdm
from data.utils import rgb_to_phase, plantnet300K_image_paths
from PRNN.pipeline.PhaseImages import frames_to_svd_torch

def image_to_interferograms(filename, nframes, nbar_signal, nbar_bkgrnd, device):

    y = rgb_to_phase(filename, color_balance=[0.6, 0.2, 0.2])

    y = torch.tensor(y).to(device)

    # ycrop = crop_and_resize(y, nx, ny)
    y = random_resize_crop(y.unsqueeze(0)).squeeze(0)

    # generate array of phi values
    phi = torch.rand(nframes) * 2 * torch.pi - torch.pi

    # make nframe copies of original phase mask
    phase_mask = y.repeat(nframes, 1, 1)

    # add phi to each copy (#UUUUU 20231212)
    phase_mask = phase_mask + phi.unsqueeze(-1).unsqueeze(-1).to(device)

    # keep phase mask values is between 0 and 2pi
    phase_mask = phase_mask % (2 * torch.pi)

    # make detected intensity
    x = (
            torch.abs(E1) ** 2
            + torch.abs(E2) ** 2
            + 2 * vis * torch.abs(E1) * torch.abs(E2) * torch.cos(phase_mask)
    )

    # normalize by mean of sum of frames
    x = x / torch.mean(torch.sum(x, axis=(-2, -1)))

    # scale to nbar total counts each frame
    x = x * torch.randint(low=int(nbar_signal[0]), high=int(nbar_signal[1]) + 1, size=(1,), device=device)

    # add flat background
    x = x + torch.randint(low=int(nbar_bkgrnd[0]), high=int(nbar_bkgrnd[1]) + 1, size=(1,), device=device) / (nx*ny)

    return x, y

def poisson_sampling_batch(X, poisson_batch_size):
    while poisson_batch_size > 0:
        try:
            for i in tqdm(range(0, len(X), poisson_batch_size), desc='Poisson sampling mini-batches..'):
                poisson_minibatch = torch.poisson(X[i:i+poisson_batch_size].to(device))
                X[i:i+poisson_batch_size] = poisson_minibatch.cpu()
            break  # break out of the loop if successful
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA out of memory with poisson_batch_size = {poisson_batch_size}. Reducing batch size.")
                poisson_batch_size = max(poisson_batch_size // 2, 1)
            else:
                raise  # propagate other errors

    return X

def make_E_fields(nx, ny, sigma_X, sigma_Y):
    x = np.linspace(-5, 5, nx)
    y = np.linspace(-5, 5, ny)
    X, Y = np.meshgrid(x, y)
    E1 = np.exp(-(X)**2/(2*sigma_X**2) - (Y)**2/(2*sigma_Y**2)) # signal amplitude
    E2 = np.exp(-(X)**2/(2*sigma_X**2) - (Y)**2/(2*sigma_Y**2)) # reference amplitude
    E1 = E1.astype(np.float32) # data must be in float32 for pytorch
    E2 = E2.astype(np.float32)
    E1 = torch.tensor(E1).to(device)
    E2 = torch.tensor(E2).to(device)
    return E1, E2

### PARAMETERS ###
ndata       = 100 # number of different training frame sets to include in a data set
val_split   = 0.1
nx          = 128 # X pixels
ny          = nx # Y pixels
nframes     = 32*2
nbar_signal = (1e2, 1e5)
nbar_bkgrnd = (0, 0)
sigma_X     = 5
sigma_Y     = 5
vis         = 1
svd         = False
save        = True
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Save info
basepath = "raw/"
filepath = 'TEST_flowers_n%i_npix%i_SVD_20231216.h5' % (ndata, nx)
# filepath = 'testtt'
# filepath = 'plantnet_n%i_npix%i_SVD_20231216.h5' % (ndata, nx)
# filepath = 'mnist_n%i_npix%i.h5' % (ndata, nx)

masks_folder = 'flowers102'
filenames = os.listdir(os.path.join('masks', masks_folder))
random.seed(666)
random.shuffle(filenames)

E1, E2 = make_E_fields(nx, ny, sigma_X, sigma_Y)
random_resize_crop = torchvision.transforms.RandomResizedCrop([nx, ny], scale=(0.95, 1.0), ratio=(1.0, 1.0))

# Reserve the first 90% of files for training data, 10% for validation data
nfiles = int(len(filenames))
ntrain = int(nfiles * (1 - val_split))
ndata_train = int(ndata * (1 - val_split))
ndata_val = int(ndata * val_split)
nval   = int(nfiles - ntrain)
train_indices = list(range(0, ntrain))
val_indices = list(range(ntrain, len(filenames)))

#### Data generation loop for training set ####
truths_data = torch.zeros((ndata, nx, ny), device='cpu')
inputs_data = torch.zeros((ndata, nframes, nx, ny), device='cpu')
svd_data    = torch.zeros((ndata, 2, nx, ny), device='cpu')

""" Two loops are used to generate training and validation data. Because we have a finite number of images,
the images are re-used with some random re-sizing and cropping. The separate loops are used to avoid similar
crops of the training images ending up in the validation data. """

tic1 = time.time()

for d in tqdm(range(0, ndata_train), desc='Generating training images..'):
    idx = random.randint(0, len(train_indices))
    mask = filenames[train_indices[idx]]
    filename = os.path.join('masks', masks_folder, mask)
    x, y = image_to_interferograms(filename, nframes, nbar_signal, nbar_bkgrnd, device)
    truths_data[d, :, :] = y
    inputs_data[d, :, :, :] = x

for d in tqdm(range(0, ndata_val), desc='Generating validation images..'):
    idx = random.randint(0, len(val_indices))
    mask = filenames[val_indices[idx]]
    filename = os.path.join('masks', masks_folder, mask)
    x, y = image_to_interferograms(filename, nframes, nbar_signal, nbar_bkgrnd, device)
    truths_data[ndata_train + d, :, :] = y
    inputs_data[ndata_train + d, :, :, :] = x

print(f"Time elapsed for data generation: {time.time()-tic1:.4f} sec", end='\n')

# Poisson sampling
tic = time.time()
poisson_batch_size = 500
inputs_data = poisson_sampling_batch(inputs_data, 500)
print(f"Time elapsed for Poisson sampling mini-batch: {time.time()-tic:.4f} sec", end='\n')

if svd:
    print('Calculating SVDs..', end='\n')
    for d, x in tqdm(enumerate(inputs_data)):
        # Calculate SVD from 32 random frames
        xsubset = x[torch.randperm(x.shape[0])][0:32]
        # phi1, phi2 = frames_to_svd(xsubset)
        phi1, phi2 = frames_to_svd_torch(xsubset, device=device)
        phi1, phi2 = phi1.cpu(), phi2.cpu()
        del xsubset
        svd_data[d, 0, :, :] = phi1
        svd_data[d, 1, :, :] = phi2

print('Total time = %.4f sec' % (time.time()-tic1))

# Move data to cpu
truths_data = truths_data.cpu()
inputs_data = inputs_data.cpu()
svd_data    = svd_data.cpu()

if save:
    """ Save the data to .h5 file """
    with h5py.File(basepath+filepath, "a") as h5_data:
        h5_data["truths"] = np.array(truths_data)
        h5_data["inputs"] = np.array(inputs_data)
        h5_data["E1"] = np.array(E1.cpu())
        h5_data["E2"] = np.array(E2.cpu())
        h5_data["vis"] = np.array([vis], dtype=np.float32)
        if svd:
            h5_data["svd"] = svd_data

    h5_data.close()
    print('Saved to %s' % (basepath + filepath))