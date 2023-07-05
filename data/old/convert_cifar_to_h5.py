import h5py
import numpy as np
import os

from PIL import Image
from matplotlib import pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

dir = 'cifar-10-batches-py'
fname = 'data_batch_1'
data = unpickle(os.path.join(dir, fname))
images = data[b'data']
images_color = np.reshape(images, (images.shape[0], 3, 32, 32))

# r, g, and b are 512x512 float arrays with values >= 0 and < 1.
arr = images_color[25, :, :, :]

def cifar_to_greyscale(arr):
    arr = np.transpose(arr, axes=(1, 2, 0))
    rgb_arr = np.array(arr, 'uint8')
    return np.array(Image.fromarray(rgb_arr).convert('L'))

test = cifar_to_greyscale(arr)
plt.imshow(test)