import h5py
import numpy as np
from os import walk
import matplotlib.pyplot as plt
from numpy import exp, pi
from skimage.transform import resize
import random
import imageio
from tqdm import tqdm
from PIL import Image
import matplotlib as mpl
mpl.use("TkAgg")  # this forces a non-X server backend


def generateSamples(phase_mask):

    sampledFrames = np.zeros((num_frames, nx, ny)) # container for histograms of each frame
    
    for k in range(num_frames):
        
        numEvents = np.random.poisson(nbar)
        phi = np.random.rand(1)[0] * 2 * pi # random phase offset
        phase = phase_mask + phi
             
        I = abs(E1)**2 + abs(E2)**2 + 2 * vis * abs(E1)*abs(E2)*np.cos(phase)
        I_prob = I / np.sum(I) # normalize s.t. this is now a prob distb
        I_prob_1d = I_prob.flatten()
        
        samples = np.random.choice(np.linspace(0,nx*ny-1,nx*ny, dtype='int'), size=numEvents, p=I_prob_1d) 
        
        x_outcomes = outcome_list[samples][:, 0]
        y_outcomes = outcome_list[samples][:, 1]
        
        histogram_outcomes,_,_ = np.histogram2d(x_outcomes, y_outcomes, [nx,ny], [[y_min,y_max],[x_min,x_max]])
        
        sampledFrames[k, :, :] = histogram_outcomes
        
    return sampledFrames

def random_rotate_image(arr):
    """ Randomly apply up/down and left/right flips to input image """
    if random.random() >= 0.5:
        arr = np.flipud(arr)
    if random.random() >= 0.5:
        arr = np.fliplr(arr)
    return arr

def random_roll_image(arr):
    """ Randomy permute rows or columns of input image """
    shift = random.randint(0, arr.shape[0])
    axis = round(random.random())
    return np.roll(arr, shift, axis)

### PARAMETERS ###

ndata = 1500 # number of different training frame sets to include in a data set
nx = 32 # X pixels
ny = 32 # Y pixels
num_frames = 32 # number of frames (coherence times?) per frame set
vis = 1 # visibility of interference
nbar = 1e3 # average number of photons
sigma_X = 100
sigma_Y = 100


### DEFINE ARRAYS ###
x_min, x_max = -5, 5
y_min, y_max = -5, 5
x = np.linspace(x_min,x_max,nx)
y = np.linspace(y_min,y_max,ny)
X, Y = np.meshgrid(x,y)

E1 = exp(-(X)**2 / (2*sigma_X**2) - (Y)**2 / (2*sigma_Y**2)) # signal amplitude
E2 = exp(-(X)**2 / (2*sigma_X**2) - (Y)**2 / (2*sigma_Y**2)) # reference amplitude

outcome_list = []
for j in range(nx):
    for k in range(ny):
        outcome_list.append((x[j],y[k]))
outcome_list = np.array(outcome_list).copy() # create a array of possible indices

""" Data generation loop """

# png training images should in a folder called masks_nhl (in same directory as script)
filenames = next(walk('../masks_nhl'), (None, None, []))[2] # directory of phase mask .png files

inputs_data = np.zeros((ndata, num_frames, nx, ny))
truths_data = np.zeros((ndata, nx, ny))

for d in tqdm(range(0, ndata)):
    
    idx = random.randint(0, len(filenames)-1)
    mask = filenames[idx]
    
    filename = '../masks_nhl/'+mask
    image = np.array(Image.open(filename).convert('L')) # load as greyscale
    phase_mask = image * 1.0 / 255 * 2 * np.pi # convert to phase
    phase_mask = np.fliplr(np.flip(phase_mask)) # flip so that upright
    phase_mask = resize(phase_mask, [nx, ny]) # resize. mask_y is num rows, mask_x is num cols    

    phase_mask = random_rotate_image(phase_mask)
    phase_mask = random_roll_image(phase_mask)

    sampledFrames = generateSamples(phase_mask)

    inputs_data[d, :, :, :] = sampledFrames
    truths_data[d, :, :] = phase_mask # frames seem to always be inverted compared to the original image

    # # Plotting to verify
    # fig, ax = plt.subplots(nrows=1, ncols=2)
    # ax[0].imshow(phase_mask)
    # ax[1].imshow(frames[0])


# Save the data to .h5 file
basepath = ""
filepath = 'QIML_3logos_n%i_nbar%i_nframes%i_npix%i.h5' % (ndata, nbar, num_frames, nx)

with h5py.File(basepath+filepath, "a") as h5_data:
    h5_data["inputs"] = inputs_data
    h5_data["truths"] = truths_data
