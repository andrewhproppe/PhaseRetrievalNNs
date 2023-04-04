import h5py
import numpy as np
import random
import os

from matplotlib import pyplot as plt
from tqdm import tqdm
from data.utils import random_rotate_image, random_roll_image, convertGreyscaleImgToPhase
from QIML.utils import get_system_and_backend
get_system_and_backend()

### PARAMETERS ###
ndata   = 3000 # number of different training frame sets to include in a data set
nx      = 32 # X pixels
ny      = 32 # Y pixels
sigma_X = 5
sigma_Y = 5
vis     = 1
flat_background = 0.1
# png training images should in a folder called masks_nhl (in same directory as script)
# masks_folder = '../masks_nhl'
# masks_folder = '../masks'
masks_folder = '../masks_mnist'
filenames = os.listdir(masks_folder)

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
    idx = random.randint(0, len(filenames)-1)
    mask = filenames[idx]
    filename = os.path.join(masks_folder, mask)
    phase_mask = convertGreyscaleImgToPhase(filename, nx, ny)
    phase_mask = random_rotate_image(phase_mask)
    phase_mask = random_roll_image(phase_mask)
    phase_mask = phase_mask + flat_background*np.max(phase_mask)
    truths_data[d, :, :] = phase_mask # frames seem to always be inverted compared to the original image

""" Save the data to .h5 file """
basepath = ""
filepath = 'QIML_mnist_data_n%i_npix%i.h5' % (ndata, nx)

with h5py.File(basepath+filepath, "a") as h5_data:
    h5_data["truths"] = truths_data
    h5_data["inputs"] = []
    h5_data["E1"] = np.array([E1])
    h5_data["E2"] = np.array([E2])
    h5_data["vis"] = np.array([vis], dtype=np.float32)