from functools import lru_cache
from typing import Tuple, Type, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Compose, ToTensor
import torchvision.transforms.functional as tvf
import pytorch_lightning as pl
import random

from QIML.pipeline import transforms
from QIML.utils import paths

import platform
import matplotlib as mpl

if platform.system() == "Linux":
    mpl.use("TkAgg")


class QI_H5Dataset(Dataset):
    def __init__(self, filepath: str, seed: int = 10236, **kwargs):
        super().__init__()
        self._filepath = filepath
        self.image_transform = transforms.image_transform_pipeline()
        self.truth_transform = transforms.truth_transform_pipeline()
        self.input_transform = transforms.input_transform_pipeline()
        self.input_transform_fourier = transforms.input_transform_pipeline(submin=False)

    @property
    def filepath(self) -> str:
        return self._filepath

    @property
    def num_params(self) -> int:
        return len([key for key in self.data.keys() if "parameter" in key])

    @lru_cache()
    def __len__(self) -> int:
        """
        Returns the total number of g2s in the dataset.
        Because the array is n-dimensional, the length is
        given as the product of the first four dimensions.

        Returns
        -------
        int
            Number of g2s in the dataset
        """
        inputs_shape = self.data["truths"].shape
        return inputs_shape[0]

    @property
    @lru_cache()
    def indices(self) -> np.ndarray:
        return np.arange(len(self))

    @property
    def data(self):
        return h5py.File(self.filepath, "r")

    @property
    @lru_cache()
    def inputs(self) -> np.ndarray:
        return self.data["inputs"]

    @property
    @lru_cache()
    def truths(self) -> np.ndarray:
        return self.data["truths"]

    def __getitem__(self, index: int) -> Tuple[Type[torch.Tensor]]:
        """
        Returns a randomly chosen phase mask (truth) with noisy frames (inputs).

        Parameters
        ----------
        index : int
            Not used; passed by a `DataLoader`

        Returns
        -------
        x : torch.Tensor
            Noisy frames
        y : torch.Tensor
            Noise-free phase mask
        """

        x = self.inputs[index]
        y = self.truths[index]

        x = self.input_transform(x)
        y = self.truth_transform(y)

        return x, y


class QI_H5Dataset_Poisson(QI_H5Dataset):
    def __init__(self, filepath: str, seed: int = 10236, **kwargs):
        super().__init__(filepath, seed, **kwargs)

        # To grab **kwargs
        self.nframes = None
        self.nbar = None
        self.flat_background = (0.0, 0, 0)
        self.corr_matrix = None
        self.fourier = None
        self.randomize = True
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, index: int) -> Tuple[Type[torch.Tensor]]:
        """
        Returns a randomly chosen phase mask (truth) with noisy frames (inputs).

        Parameters
        ----------
        index : int
            Not used; passed by a `DataLoader`

        Returns
        -------
        x : torch.Tensor
            Noisy frames
        y : torch.Tensor
            Noise-free phase mask
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        y = torch.tensor(self.truths[index]).to(device)
        E1 = torch.tensor(self.E1[0]).to(device)
        E2 = torch.tensor(self.E2[0]).to(device)
        vis = torch.tensor(self.vis[0]).to(device)

        """ Add background and random transformation to y """
        # Is this the right place to add background?
        # y = (y+self.flat_background*y.max())*y.max()/(y.max()*(1+self.flat_background)) # add a flat background as a fraction of the max mask value

        if self.randomize:
            y = self.image_transform(y)  # apply random h and v flips
            angle = random.choice([-90, 0, 90])
            # rotate by a random multiple of 90˚
            y = tvf.rotate(y.unsqueeze(0), float(angle)).squeeze(0)

        nbar = torch.randint(
            low=int(self.nbar[0]), high=int(self.nbar[1]) + 1, size=(1,)
        ).to(device)

        bckgrnd = (
            torch.FloatTensor(1)
            .uniform_(self.flat_background[0], self.flat_background[1])
            .to(device)
        )

        """ Make Poisson sampled frames through only broadcasted operations. Seems about 30% faster on CPU """
        phi = torch.rand(self.nframes) * 2 * torch.pi  # generate array of phi values
        # make nframe copies of original phase mask
        phase_mask = y.repeat(self.nframes, 1, 1).to(device)
        # add phi to each copy
        phase_mask = phase_mask + phi.unsqueeze(-1).unsqueeze(-1).to(device)
        # make detected intensity
        x = (
            torch.abs(E1) ** 2
            + torch.abs(E2) ** 2
            + 2 * vis * torch.abs(E1) * torch.abs(E2) * torch.cos(phase_mask)
        )
        # get maximum intensity of each frame and reshape to broadcast
        x_maxima = torch.sum(x, axis=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
        # scale to nbar total counts each frame
        x = x * nbar / x_maxima
        # add flat background
        x = x + bckgrnd
        # Poisson sample each pixel of each frame
        x = torch.poisson(x)

        if self.corr_matrix:
            # x = x - x.mean(dim=0)
            xflat = torch.flatten(x, start_dim=1)
            x = torch.matmul(torch.transpose(xflat, 0, 1), xflat)

        x = self.input_transform(x).to(torch.device("cpu"))
        y = self.truth_transform(y).to(torch.device("cpu"))

        if self.fourier:
            xf = torch.fft.fft2(x, dim=(-2, -1))
            xf = torch.fft.fftshift(xf, dim=(-2, -1))
            xf = torch.abs(xf)
            xf = self.input_transform(xf).to(torch.device("cpu"))
            x = torch.concat(
                (x.unsqueeze(0), xf.unsqueeze(0)), dim=0
            )  # create channel dimension and concat x with real and imaginary fft parts
            del xf

            # xr = self.input_transform_fourier(xf.real).to(torch.device('cpu'))
            # xi = self.input_transform_fourier(xf.imag).to(torch.device('cpu'))
            # x = torch.concat((x.unsqueeze(0), xr.unsqueeze(0), xi.unsqueeze(0)), dim=0) # create channel dimension and concat x with real and imaginary fft parts
            # del xf, xr, xi

        return x, y

    @property
    @lru_cache()
    def E1(self) -> np.ndarray:
        return self.data["E1"]

    @property
    @lru_cache()
    def E2(self) -> np.ndarray:
        return self.data["E2"]

    @property
    @lru_cache()
    def vis(self) -> np.ndarray:
        return self.data["vis"]


class QIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        h5_path: Union[None, str] = None,
        batch_size: int = 64,
        seed: int = 120516,
        num_workers=0,
        **kwargs
    ):
        super().__init__()
        # by default run with the devset
        if not h5_path:
            h5_path = "QI_devset.h5"
        self.h5_path = paths.get("raw").joinpath(h5_path)
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.data_kwargs = kwargs

    def setup(self, stage: Union[str, None] = None):
        full_dataset = QI_H5Dataset_Poisson(self.h5_path, **self.data_kwargs)
        # use 10% of the data set a test set
        test_size = int(len(full_dataset) * 0.2)
        self.train_set, self.val_set = random_split(
            full_dataset,
            [len(full_dataset) - test_size, test_size],
            # torch.Generator().manual_seed(self.seed), #FFFFF
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            # collate_fn=transforms.pad_collate_func,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            # collate_fn=transforms.pad_collate_func,
        )


def get_test_batch(
    batch_size: int = 32, h5_path: Union[None, str] = None, seed: int = 120516
) -> Tuple[torch.Tensor]:
    """
    Convenience function to grab a batch of validation data using the same
    pipeline as in the training process.

    Parameters
    ----------
    batch_size : int, optional
        Number of g2s to grab, by default 32
    h5_path : Union[None, str], optional
        Name of the HDF5 file containing g2s, by default None
    seed : int, optional
        random seed, by default 120516, same as training
    noise : bool, optional
        Whether to use the noisy data set, by default True

    Returns
    -------
    Tuple[torch.Tensor]
        3-tuple of data
    """
    target_module = QIDataModule
    data = target_module(h5_path, batch_size, seed)
    data.setup()
    return next(iter(data.val_dataloader()))


def plot_frames(frames, nrows=4, ncols=None, figsize=(4, 4), dpi=150, cmap="gray"):
    if ncols is None:
        ncols = nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=150)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(frames[i], cmap=cmap)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# Testing
if __name__ == "__main__":

    import time
    from matplotlib import pyplot as plt

    data_fname = "flowers_n600_npix64.h5"
    data = QIDataModule(
        data_fname,
        batch_size=50,
        num_workers=0,
        nbar=(1e3, 1e4),
        nframes=64,
        shuffle=True,
        randomize=True,
        flat_background=(0.2, 2),
    )
    data.setup()
    batch = next(iter(data.train_dataloader()))

    y = batch[1][0].numpy()
    test = batch[0][0].numpy()
    plot_frames(test, nrows=4, figsize=(4, 4), dpi=150, cmap="viridis")
    # start = time.time()
    # (x, y) = data.train_set.__getitem__(1)
    # print(f'Time: {time.time() - start}')
    # print('fin')
