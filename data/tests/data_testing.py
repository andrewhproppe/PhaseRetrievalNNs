from PRNN.utils import get_system_and_backend
get_system_and_backend()

import h5py
import numpy as np
import random
import torch
import os
import torchvision.transforms

from matplotlib import pyplot as plt
from tqdm import tqdm
from data.utils import random_rotate_image, random_roll_image, convertGreyscaleImgToPhase, rgb_to_phase, crop_and_resize
from PRNN.pipeline.PhaseImages import frames_to_svd, frames_to_svd_torch

### PARAMETERS ###
ndata   = 1000 # number of different training frame sets to include in a data set
nx      = 64 # X pixels
ny      = nx # Y pixels
nframes = 32*2
nbar_signal = (1e2, 1e5)
nbar_bkgrnd = (0, 0)
sigma_X = 5
sigma_Y = 5
vis     = 1
svd     = True
save    = False

# masks_folder = 'mnist'
masks_folder = 'flowers_more'
filenames = os.listdir(os.path.join('../masks', masks_folder))
filenames.sort()

### DEFINE ARRAYS ###
x = np.linspace(-5, 5, nx)
y = np.linspace(-5, 5, ny)
X, Y = np.meshgrid(x, y)

# the beam profile generation could also be moved to the data generation loop, so that different beam profiles are used in the training. Same for visibility
E1 = np.exp(-(X)**2/(2*sigma_X**2) - (Y)**2/(2*sigma_Y**2)) # signal amplitude
E2 = np.exp(-(X)**2/(2*sigma_X**2) - (Y)**2/(2*sigma_Y**2)) # reference amplitude
E1 = E1.astype(np.float32) # data must be in float32 for pytorch
E2 = E2.astype(np.float32)

E1 = torch.tensor(E1)
E2 = torch.tensor(E2)

""" Data generation loop """
truths_data = np.zeros((ndata, nx, ny), dtype=np.float32)
inputs_data = np.zeros((ndata, nframes, nx, ny), dtype=np.float32)
svd_data    = np.zeros((ndata, 2, nx, ny), dtype=np.float32)


import time

random_resize_crop = torchvision.transforms.RandomResizedCrop([nx, ny], scale=(0.75, 1.0), ratio=(1.0, 1.0))

for d in tqdm(range(0, ndata)):

    idx = random.randint(0, len(filenames)-1)
    # idx = d
    mask = filenames[idx]
    filename = os.path.join('../masks', masks_folder, mask)

    y = rgb_to_phase(filename, color_balance=[0.6, 0.2, 0.2])

    y = torch.tensor(y)

    # ycrop = crop_and_resize(y, nx, ny)
    y = random_resize_crop(y.unsqueeze(0)).squeeze(0)

    # generate array of phi values
    phi = torch.rand(nframes) * 2 * torch.pi - torch.pi

    # make nframe copies of original phase mask
    phase_mask = y.repeat(nframes, 1, 1)

    # add phi to each copy (#UUUUU 20231212)
    phase_mask = phase_mask + phi.unsqueeze(-1).unsqueeze(-1)

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
    x = x * torch.randint(low=int(nbar_signal[0]), high=int(nbar_signal[1]) + 1, size=(1,))

    # add flat background
    x = x + torch.randint(low=int(nbar_bkgrnd[0]), high=int(nbar_bkgrnd[1]) + 1, size=(1,)) / (nx*ny)

    truths_data[d, :, :] = y
    inputs_data[d, :, :, :] = x

# raise RuntimeError

# Poisson and SVD in loop
X = torch.tensor(inputs_data.copy()).to(device='cuda')

tic = time.time()
for x in X:
    x = torch.poisson(x)
print(f"Time elapsed for looped Poisson: {time.time()-tic:.5f} sec")

# Poisson and SVD in batch
tic = time.time()
X_poisson = torch.poisson(X)
print(f"Time elapsed for batched Poisson: {time.time()-tic:.5f} sec")

#
# tic = time.time()
# for x in X:
#     _, _ = frames_to_svd_torch(x, device='cuda')
# print(f"Time elapsed for looped SVD: {time.time()-tic:.5f} sec")
#
# def frames_to_svd_torch_batched(X, device):
#     xflat = torch.flatten(X, start_dim=2).to(device)
#     batch_size, nframes, Nx, Ny = X.shape
#     U, S, Vh = torch.linalg.svd(xflat)
#     zsin = torch.reshape(Vh[:, 1, :], (batch_size, Nx, Ny))
#     zcos = torch.reshape(Vh[:, 2, :], (batch_size, Nx, Ny))
#     z1 = zcos + 1j * zsin
#     z2 = zsin + 1j * zcos
#     phi1 = torch.angle(z1).cpu()
#     phi2 = torch.angle(z2).cpu()
#     return phi1, phi2
#
# tic = time.time()
# _, _ = frames_to_svd_torch_batched(X, device='cuda')
# print(f"Time elapsed for batched SVD: {time.time()-tic:.5f} sec")

