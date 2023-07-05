import random
import numpy as np
import torch

from imageio import imread
from skimage.transform import resize

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


def rgb2gray(rgb, color_balance=None):
    if color_balance is None:
        color_balance = [0.2989, 0.5870, 0.1140]
    return np.dot(rgb[..., :3], color_balance)


def convertGreyscaleImgToPhase(img_filename, mask_x, mask_y, color_balance=None):
    '''Loads an image and returns a phase mask of size [mask_x, mask_y].

    Converts greyscale [0,255] to phase values [0, 2*pi]

    '''

    image = imread(img_filename)

    if len(image.shape) > 2:
        image = rgb2gray(image, color_balance)

    phase_mask = image/255*2*np.pi
    phase_mask = resize(phase_mask, [mask_y, mask_x])  # mask_y is num rows, mask_x is num cols

    return phase_mask


def rgb_to_phase(img_filename, color_balance=None):
    '''Loads an image and returns a phase mask of size [mask_x, mask_y].

    Converts greyscale [0,255] to phase values [0, 2*pi]

    '''

    image = imread(img_filename)

    if len(image.shape) > 2:
        image = rgb2gray(image, color_balance)

    phase_mask = image/255*2*np.pi

    return phase_mask

def crop_and_resize(phase_mask, mask_x, mask_y):
    ''' Ρesizes a phase mask to size [mask_x, mask_y]. May add crop option later '''
    return resize(phase_mask, [mask_y, mask_x])  # mask_y is num rows, mask_x is num cols

def get_batch_from_dataset(data, batch_size):
    data.setup()
    # Loop to generate a batch of data taken from dataset
    for i in range(0, batch_size):
        if i == 0:
            X, _ = data.train_set.__getitem__(0)
            X = X.unsqueeze(0)
        else:
            Xtemp, _ = data.train_set.__getitem__(0)
            Xtemp = Xtemp.unsqueeze(0)
            X = torch.cat((X, Xtemp), dim=0)

    return X