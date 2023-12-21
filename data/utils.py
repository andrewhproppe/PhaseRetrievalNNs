import os
import random

import h5py
import numpy as np
import torch

from imageio import imread
from skimage.transform import resize
from torchvision.transforms import RandomCrop
from tqdm import tqdm


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

def crop_and_resize(phase_mask, mask_x, mask_y, crop_frac=0.8, make_square=True):
    """
    Crops a phase mask image based on a fraction of its original size (crop_frac; if = 1, then no crop),
    then resizes to [mask_x, mask_y].
    """
    crop_size = int(min(phase_mask.shape)*crop_frac)
    cropped_phase_mask = RandomCrop(crop_size)(torch.tensor(phase_mask)).numpy()
    return resize(cropped_phase_mask, [mask_y, mask_x])  # mask_y is num rows, mask_x is num cols



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


def get_from_h5(data_fname, device, idx_start, idx_stop=None):
    idx_stop = idx_start+1 if idx_stop is None else idx_stop
    with h5py.File(data_fname, "r") as f:
        y = torch.tensor(f["truths"][idx_start:idx_stop, :]).float().to(device)
        E1 = torch.tensor(f["E1"][:]).float().to(device)
        E2 = torch.tensor(f["E2"][:]).float().to(device)
        vis = torch.tensor(f["vis"][:]).float().to(device)
    return y, E1, E2, vis


def plantnet300K_image_paths(ndata):
    root_directory = "masks/plantnet_300K/images_train"
    image_paths = []
    for subdir, dirs, files in os.walk(root_directory):
        for file in files:
            # Check if the file is an image (you can extend this check based on your image file extensions)
            if file.endswith(".jpg") or file.endswith(".png"):
                image_path = os.path.join(subdir, file)
                image_paths.append(image_path)
                if len(image_paths) > ndata - 1:
                    break
        if len(image_paths) > ndata - 1:
            break

    return image_paths


def image_to_interferograms(filename, E1, E2, vis, nx, ny, nframes, nbar_signal, nbar_bkgrnd, color_balance, random_crop_layer, device):

    y = rgb_to_phase(filename, color_balance=color_balance)

    y = torch.tensor(y).to(device)

    # ycrop = crop_and_resize(y, nx, ny)
    y = random_crop_layer(y.unsqueeze(0)).squeeze(0)

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


def poisson_sampling_batch(X, poisson_batch_size, device):
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


def make_E_fields(nx, ny, sigma_X, sigma_Y, device):
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
