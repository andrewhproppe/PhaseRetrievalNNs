import h5py
import numpy as np
import random
import os

from matplotlib import pyplot as plt
from tqdm import tqdm
from data.utils import random_rotate_image, random_roll_image, convertGreyscaleImgToPhase
from PRNN.utils import get_system_and_backend
get_system_and_backend()

### PARAMETERS ###
ndata   = 1000 # number of different training frame sets to include in a data set
nx      = 32 # X pixels
ny      = 32 # Y pixels
sigma_X = 5
sigma_Y = 5
vis     = 1
flat_background = 0.1
# png training images should in a folder called masks_nhl (in same directory as script)
# masks_folder = '../masks_nhl'
# masks_folder = 'masks'
# masks_folder = '../emojis'
# masks_folder = '../masks'
masks_folder = 'mnist'
filenames = os.listdir(os.path.join('../masks', masks_folder))

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
    filename = os.path.join('../masks', masks_folder, mask)
    phase_mask = convertGreyscaleImgToPhase(filename, nx, ny)
    phase_mask = random_rotate_image(phase_mask)
    # phase_mask = random_roll_image(phase_mask)
    phase_mask = phase_mask + flat_background*np.max(phase_mask)
    truths_data[d, :, :] = phase_mask # frames seem to always be inverted compared to the original image

""" Save the data to .h5 file """
basepath = "raw/"
filepath = 'QIML_mnist_data_n%i_npix%i.h5' % (ndata, nx)

with h5py.File(basepath+filepath, "a") as h5_data:
    h5_data["truths"] = truths_data
    h5_data["inputs"] = []
    h5_data["E1"] = np.array([E1])
    h5_data["E2"] = np.array([E2])
    h5_data["vis"] = np.array([vis], dtype=np.float32)



# """ Make Poisson sampled frames through only broadcasted operations. Seems about 30% faster on CPU """
# import torch
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# E1 = torch.tensor(E1)
# E2 = torch.tensor(E2)
# def random_phi_frames(phase_mask, nframes, nbar):
#     phi = torch.rand(nframes) * 2 * torch.pi  # generate array of phi values
#     phase_masks = phase_mask.repeat(nframes, 1, 1).to(device)  # make nframe copies of original phase mask
#     phase = phase_masks + phi.unsqueeze(-1).unsqueeze(-1).to(device)  # add phi to each copy
#     I = torch.abs(E1)**2 + torch.abs(E2)**2 + 2*vis*torch.abs(E1)*torch.abs(E2)*torch.cos(phase)  # make detected intensity
#     I_maxima = torch.sum(I, axis=(-2, -1)).unsqueeze(-1).unsqueeze(-1)  # get maximum intensity of each frame
#     I = I*nbar/I_maxima  # scale to nbar total counts each frame
#     return torch.poisson(I)  # Poisson sample each pixel of each frame
#
# f = random_phi_frames(torch.tensor(phase_mask), 32, 100)
# f_sum = torch.sum(f, dim=0)
# f_acos = torch.acos(f)
# f_acos_sum = torch.sum(f_acos, dim=0)
# # f_sum = torch.sum(f, dim=0)
# # f_acos = torch.sum(torch.acos(f), dim=0)
# # nrow = 5
# # ncol = 5
# # fig, axs = plt.subplots(nrow, ncol)
# #
# # frame = phase_mask
# # frame_scaled = frame / np.sum(frame)*100
# # for i, ax in enumerate(fig.axes):
# #     frame = np.random.poisson(frame_scaled)
# #     # frame = np.arccos(frame)
# #     ax.imshow(frame)
# #     ax.set_aspect('equal', 'box')
# #     ax.axis('off')
# #
# # plt.tight_layout()