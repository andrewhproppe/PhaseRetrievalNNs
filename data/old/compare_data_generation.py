import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, pi
from skimage.transform import resize
import random
import imageio
import os
from data.utils import convertGreyscaleImgToPhase, random_roll_image, random_rotate_image


ndata = 1 # number of different training frame sets to include in a data set
nx = 64 # X pixels
ny = 64 # Y pixels
nframes = 64 # number of frames (coherence times?) per frame set
nbar = 1e4 # average number of photons
sigma_X = 100
sigma_Y = 100

x = np.linspace(-5,5,nx)
y = np.linspace(-5,5,ny)
X, Y = np.meshgrid(x,y)
E1 = exp(-(X)**2 / (2*sigma_X**2) - (Y)**2 / (2*sigma_Y**2))
E2 = exp(-(X)**2 / (2*sigma_X**2) - (Y)**2 / (2*sigma_Y**2))
vis = 1

masks_folder = 'masks_nhl'
# masks_folder = 'masks'
filenames = os.listdir('../'+masks_folder)

outcome_list = []
for j in range(nx):
    for k in range(ny):
        outcome_list.append((x[j],y[k]))
outcome_list = np.array(outcome_list).copy() # create a array of possible indices

idx = random.randint(0, len(filenames) - 1)
mask = filenames[idx]
# phase_mask = np.fliplr(np.flip(convertGreyscaleImgToPhase('../' + masks_folder + '/' + mask, nx, ny)))
phase_mask = convertGreyscaleImgToPhase('../' + masks_folder + '/' + mask, nx, ny)
# phase_mask = random_rotate_image(phase_mask)
# phase_mask = random_roll_image(phase_mask)


samplesDict = {}
numEvents = np.random.poisson(nbar)
phi = np.random.rand(1)[0] * 2 * pi # random phase offset
phase = phase_mask + phi
I = abs(E1)**2 + abs(E2)**2 + 2 * vis * abs(E1)*abs(E2)*np.cos(phase)

""" Method 1 """
I_prob = I / np.sum(I) # normalize s.t. this is now a prob distb
I_prob_1d = I_prob.flatten()
samples = np.random.choice(np.linspace(0,ny*nx-1,ny*nx, dtype='int'), size=numEvents, p=I_prob_1d)
x_outcomes = outcome_list[samples][:, 0]
y_outcomes = outcome_list[samples][:, 1]
samplesDict[0] = np.transpose(np.array([x_outcomes, y_outcomes]))
frames_histo = np.histogram2d(samplesDict[0][:, 0], samplesDict[0][:, 1], [nx, ny], [[-5, 5], [-5, 5]])[0]

""" Method 2 """
I_scaled = I*nbar/np.max(I)
I_poisson = np.random.poisson(I)

fig, ax = plt.subplots(ncols=2, nrows=1, dpi=150, figsize=(6, 3))
ax[0].imshow(frames_histo)
ax[1].imshow(I_poisson)