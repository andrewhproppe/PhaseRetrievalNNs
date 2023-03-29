import h5py
import numpy as np
import random
import imageio
import matplotlib as mpl

from matplotlib import pyplot as plt
from os import walk
from numpy import exp, pi
from skimage.transform import resize
from tqdm import tqdm
from QIML.utils import random_rotate_image, random_roll_image, convertGreyscaleImgToPhase


ndata   = 1000 # number of different training frame sets to include in a data set
nx      = 64 # X pixels
ny      = 64 # Y pixels
sigma_X = 100
sigma_Y = 100
vis     = 1

filenames = next(walk('../masks'), (None, None, []))[2] # directory of phase mask .png files
filenames.pop(0)

x = np.linspace(-5, 5, nx)
y = np.linspace(-5, 5, ny)
X, Y = np.meshgrid(x, y)

# the beam profile generation could also be moved to the data generation loop, so that different beam profiles are used in the training. Same for visibility
E1 = exp(-(X)**2/(2*sigma_X**2) - (Y)**2/(2*sigma_Y**2))
E2 = exp(-(X)**2/(2*sigma_X**2) - (Y)**2/(2*sigma_Y**2))

""" Data generation loop """
truths_data = np.zeros((ndata, nx, ny), dtype=np.float32)
for d in tqdm(range(0, ndata)):
    idx = random.randint(0, len(filenames)-1)
    mask = filenames[idx]
    phase_mask = np.fliplr(np.flip(convertGreyscaleImgToPhase('../masks/'+mask, nx, ny)))
    phase_mask = random_rotate_image(phase_mask)
    phase_mask = random_roll_image(phase_mask)
    truths_data[d, :, :] = phase_mask # frames seem to always be inverted compared to the original image

""" Save the data to .h5 file """
basepath = ""
filepath = 'QIML_poisson_data_n%i_npix%i.h5' % (ndata, nx)

with h5py.File(basepath+filepath, "a") as h5_data:
    h5_data["truths"] = truths_data
    h5_data["inputs"] = []
    h5_data["E1"] = np.array([E1])
    h5_data["E2"] = np.array([E2])
    h5_data["vis"] = np.array([vis])