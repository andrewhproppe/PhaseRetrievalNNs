import h5py
import numpy as np
import random
import os

from matplotlib import pyplot as plt
from tqdm import tqdm
from data.utils import random_rotate_image, random_roll_image, convertGreyscaleImgToPhase, rgb_to_phase, crop_and_resize
from PRNN.utils import get_system_and_backend
get_system_and_backend()

### PARAMETERS ###
ndata   = 10000 # number of different training frame sets to include in a data set
nx      = 64 # X pixels
ny      = nx # Y pixels
sigma_X = 5
sigma_Y = 5
vis     = 1
save    = False

# masks_folder = 'mnist'
masks_folder = 'flowers'
filenames = os.listdir(os.path.join('masks', masks_folder))
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

""" Data generation loop """
truths_data = np.zeros((ndata, nx, ny), dtype=np.float32)
for d in tqdm(range(0, ndata)):
    # idx = random.randint(0, len(filenames)-1)
    idx = d
    mask = filenames[idx]
    filename = os.path.join('masks', masks_folder, mask)
    phase_mask = rgb_to_phase(filename, color_balance=[0.6, 0.2, 0.2])
    phase_mask = crop_and_resize(phase_mask, nx, ny)
    truths_data[d, :, :] = phase_mask


if save:
    """ Save the data to .h5 file """
    basepath = "raw/"
    filepath = 'flowers_n%i_npix%i.h5' % (ndata, nx)
    # filepath = 'mnist_n%i_npix%i.h5' % (ndata, nx)

    with h5py.File(basepath+filepath, "a") as h5_data:
        h5_data["truths"] = truths_data
        h5_data["inputs"] = []
        h5_data["E1"] = np.array([E1])
        h5_data["E2"] = np.array([E2])
        h5_data["vis"] = np.array([vis], dtype=np.float32)
