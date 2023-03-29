import torch
import random
import numpy as np
from pathlib import Path
from imageio import imread
from skimage.transform import resize


install_path = Path(__file__)
top = install_path.parents[1].absolute()

paths = {
    "raw": top.joinpath("data/raw"),
    "processed": top.joinpath("data/processed"),
    "models": top.joinpath("models"),
    "notebooks": top.joinpath("notebooks"),
    "scripts": top.joinpath("scripts")
}


def get_encoded_size(data, model):
    data.setup()
    # Loop to generate a batch of data taken from dataset
    for i in range(0, 12):
        if i == 0:
            X, _ = data.train_set.__getitem__(0)
            X = X.unsqueeze(0)
        else:
            Xtemp, _ = data.train_set.__getitem__(0)
            Xtemp = Xtemp.unsqueeze(0)
            X = torch.cat((X, Xtemp), dim=0)

    # some shape tests before trying to actually train
    z, res = model.encoder(X.unsqueeze(1))
    # out = model(X)[0]
    return z, res

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


def convertGreyscaleImgToPhase(img_filename, mask_x, mask_y):
    '''Loads an image and returns a phase mask of size [mask_x, mask_y].

    Converts greyscale [0,255] to phase values [0, 2*pi]

    '''

    image = imread(img_filename)
    image = image[:, :, 2]  # convert to baw

    phase_mask = image/255*2*np.pi
    phase_mask = resize(phase_mask, [mask_y, mask_x])  # mask_y is num rows, mask_x is num cols

    return phase_mask