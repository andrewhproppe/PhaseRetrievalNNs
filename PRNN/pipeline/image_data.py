from PRNN.models.submodels_gen2 import FramesToEigenvalues
from PRNN.utils import get_system_and_backend
get_system_and_backend()

from functools import lru_cache
from typing import Tuple, Type, Union

import os
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import torchvision.transforms.functional as tvf
import pytorch_lightning as pl
import random

from PRNN.pipeline import transforms
from PRNN.utils import paths


def make_interferogram_frames(y, E1, E2, vis, nbar_signal, nbar_bkgrnd, npixels, nframes, device):
    # generate array of phi values
    phi = torch.rand(nframes) * 2 * torch.pi - torch.pi

    # make nframe copies of original phase mask
    phase_mask = y.repeat(nframes, 1, 1).to(device)

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
    x = x * nbar_signal

    # add flat background
    x = x + nbar_bkgrnd / npixels

    # Poisson sample each pixel of each frame
    x = torch.poisson(x)

    return x


class FrameDataset(Dataset):
    def __init__(self, filepath: str, seed: int = 10236, **kwargs):
        super().__init__()
        self._filepath = filepath

        # To grab **kwargs
        self.nframes = 32
        self.nbar_signal = (1e3, 1.1e3)
        self.nbar_bkgrnd = (0, 0)
        self.corr_matrix = None
        self.fourier = None
        self.randomize = True
        self.experimental = False
        self.premade = False
        self.device = 'cpu'
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.image_transform = transforms.image_transform_pipeline()
        self.truth_transform = transforms.truth_transform_pipeline(submin=False)
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

    def randomize_inputs(self, x, y, p=None, nframes=32, shuffle=True):
        # Apply a random rotation
        angle = torch.randint(0, 3, (1,)).item() * 90 - 90
        y = tvf.rotate(y.unsqueeze(0), float(angle)).squeeze(0)
        x = tvf.rotate(x, float(angle))
        if p is not None:
            p = tvf.rotate(p, float(angle))

        # Apply random horizontal and vertical flips
        if torch.rand(1).item() > 0.5:
            x = tvf.hflip(x)
            y = tvf.hflip(y)
            if p is not None:
                p = tvf.hflip(p)
        if torch.rand(1).item() > 0.5:
            x = tvf.vflip(x)
            y = tvf.vflip(y)
            if p is not None:
                p = tvf.vflip(p)

        # Shuffle the order of the frames
        if shuffle:
            idx = torch.randperm(nframes)
            x = x[idx]

        return (x, y, p) if p is not None else (x, y)

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
        y = torch.tensor(self.truths[index]).to(self.device)

        # For experimentally measured sets of frames
        if self.experimental:
            x = torch.tensor(self.inputs[index]).to(self.device)
            x, y = self.randomize_inputs(x, y)

        # For simulated sets of frames
        else:
            E1 = torch.tensor(self.E1[0]).to(self.device)
            E2 = torch.tensor(self.E2[0]).to(self.device)
            vis = torch.tensor(self.vis[0]).to(self.device)

            # For interferograms made outside of training loop
            if self.premade:
                x = torch.tensor(self.inputs[index]).to(self.device)
                # 64 frames are loaded, and 32 are randomly selected
                x, y = self.randomize_inputs(x, y, nframes=self.nframes, shuffle=True)

            # Create random interferograms within training loop
            else:
                """ Add background and random transformation to y """
                if self.randomize:
                    y = self.image_transform(y)  # apply random h and v flips
                    angle = random.choice([-90, 0, 90])
                    # rotate by a random multiple of 90˚
                    y = tvf.rotate(y.unsqueeze(0), float(angle)).squeeze(0)

                nbar_signal = torch.randint(
                    low=int(self.nbar_signal[0]), high=int(self.nbar_signal[1]) + 1, size=(1,)
                ).to(self.device)

                nbar_bkgrnd = torch.randint(
                    low=int(self.nbar_bkgrnd[0]), high=int(self.nbar_bkgrnd[1]) + 1, size=(1,)
                ).to(self.device)

                npixels = y.shape[-1] * y.shape[-2]

                x = make_interferogram_frames(
                    y,
                    E1,
                    E2,
                    vis,
                    nbar_signal,
                    nbar_bkgrnd,
                    npixels,
                    self.nframes,
                    self.device,
                )

                if self.corr_matrix:
                    # x = x - x.mean(dim=0)
                    xflat = torch.flatten(x, start_dim=1)
                    x = torch.matmul(torch.transpose(xflat, 0, 1), xflat)

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

        # x = self.input_transform(x).to(torch.device("cpu"))
        x = self.input_transform(x)
        # y = self.truth_transform(y).to(torch.device("cpu"))
        y = self.truth_transform(y)

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


class HybridDataset(FrameDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prior_transform = transforms.svd_transform_pipeline(minmax=(-1, 1))

    @property
    @lru_cache()
    def svds(self) -> np.ndarray:
        return self.data["svd"]

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
        y = torch.tensor(self.truths[index]).to(self.device)
        x = torch.tensor(self.inputs[index]).to(self.device)
        p = torch.tensor(self.svds[index]).to(self.device)

        x, y, p = self.randomize_inputs(x=x, y=y, p=p, nframes=32, shuffle=True)

        x = self.input_transform(x)
        y = self.truth_transform(y)
        p = self.prior_transform(p)

        return x, y, p


class SVDDataset(FrameDataset):
    def __init__(self, filepath: str, seed: int = 10236, **kwargs):
        super().__init__(filepath, seed, **kwargs)

        self.input_transform = transforms.svd_transform_pipeline(minmax=(-1, 1))

    @property
    @lru_cache()
    def svds(self) -> np.ndarray:
        return self.data["svd"]

    def __getitem__(self, index: int) -> Tuple[Type[torch.Tensor]]:
        """
        Returns a randomly chosen phase mask (truth) with SVD reconstructions (svd).

        Parameters
        ----------
        index : int
            Not used; passed by a `DataLoader`

        Returns
        -------
        x : torch.Tensor
            SVD reconstructions
        y : torch.Tensor
            Noise-free phase mask
        """

        y = torch.tensor(self.truths[index]).to(self.device)
        x = torch.tensor(self.svds[index]).to(self.device)

        x, y = self.randomize_inputs(x, y, shuffle=False)

        x = self.input_transform(x)
        y = self.truth_transform(y)

        return x, y


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        h5_path: Union[None, str] = None,
        batch_size: int = 64,
        seed: int = 120516,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        val_size: float = 0.1,
        split_type: str = 'fixed',
        data_type: str = 'frames',
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
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.val_size = val_size
        self.split_type = split_type
        self.data_type = data_type
        self.data_kwargs = kwargs

        data_module_info = {
            "h5_path": h5_path,
            "batch_size": self.batch_size,
        }
        self.data_module_info = {**data_module_info, **self.data_kwargs}

        self.check_h5_path()

    def check_h5_path(self):
        """ Useful to run on instantiating the module class so that wandb, pytorch etc.
         don't initialize before they realize the data file doesn't exist. """
        if not os.path.exists(self.h5_path):
            raise RuntimeError('Unable to find h5 file path.')

    def setup(self, stage: Union[str, None] = None):
        if self.data_type == 'frames':
            full_dataset = FrameDataset(self.h5_path, **self.data_kwargs)
        elif self.data_type == 'hybrid':
            full_dataset = HybridDataset(self.h5_path, **self.data_kwargs)

        ntotal = int(len(full_dataset))
        ntrain = int(ntotal * (1 - self.val_size))
        nval   = ntotal - ntrain

        if self.split_type == 'fixed':
            self.train_set = Subset(full_dataset, range(0, ntrain))
            self.val_set = Subset(full_dataset, range(ntrain, ntotal))

        elif self.split_type == 'random':
            self.train_set, self.val_set = random_split(
                full_dataset,
                [ntrain, nval],
                # torch.Generator().manual_seed(self.seed), #FFFFF
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
            # collate_fn=transforms.pad_collate_func,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
            # collate_fn=transforms.pad_collate_func,
        )


class SVDDataModule(pl.LightningDataModule):
    def __init__(
        self,
        h5_path: Union[None, str] = None,
        batch_size: int = 64,
        seed: int = 120516,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        split_type: str = 'fixed',
        val_size: float = 0.1,
        **kwargs
    ):
        super().__init__()

        self.h5_path = paths.get("raw").joinpath(h5_path)
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.val_size = val_size
        self.split_type = split_type
        self.data_kwargs = kwargs

        data_module_info = {
            "h5_path": h5_path,
            "batch_size": self.batch_size,
        }
        self.data_module_info = {**data_module_info, **self.data_kwargs}

        self.check_h5_path()

    def check_h5_path(self):
        """ Useful to run on instantiating the module class so that wandb, pytorch etc.
         don't initialize before they realize the data file doesn't exist. """
        if not os.path.exists(self.h5_path):
            raise RuntimeError('Unable to find h5 file path.')

    def setup(self, stage: Union[str, None] = None):
        full_dataset = SVDDataset(self.h5_path, **self.data_kwargs)

        ntotal = int(len(full_dataset))
        ntrain = int(ntotal * (1 - self.val_size))
        nval   = ntotal - ntrain

        if self.split_type == 'fixed':
            self.train_set = Subset(full_dataset, range(0, ntrain))
            self.val_set = Subset(full_dataset, range(ntrain, ntotal))

        elif self.split_type == 'random':
            self.train_set, self.val_set = random_split(
                full_dataset,
                [ntrain, nval],
                # torch.Generator().manual_seed(self.seed), #FFFFF
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
            # collate_fn=transforms.pad_collate_func,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
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
    target_module = ImageDataModule
    data = target_module(h5_path, batch_size, seed)
    data.setup()
    return next(iter(data.val_dataloader()))


def summon_batch(batch_size=4, nframes=32, fname= "flowers_n5000_npix64.h5", nbar_signal=(1e2, 1e5), nbar_bkgrnd=(0, 0)):
    data = ImageDataModule(
        fname,
        batch_size=batch_size,
        nbar_signal=nbar_signal,
        nbar_bkgrnd=nbar_bkgrnd,
        nframes=nframes,
        device='cpu'
    )
    data.setup()
    X, Y = next(iter(data.train_dataloader()))
    return X, Y


# Testing
if __name__ == "__main__":
    # data_fname = "flowers_n5000_npix64.h5"
    data_fname = "flowers_n100_npix64_SVD_20231214.h5"
    # data_fname = "flowers_n512_npix64_20231221.h5"

    data = ImageDataModule(
        data_fname,
        batch_size=32,
        nbar_signal=(1e3, 1.1e3),
        nbar_bkgrnd=(0, 0),
        num_workers=0,
        shuffle=True,
        premade=True,
        split_type='random',
        data_type='hybrid'
    )

    data.setup()
    X, Y, P = next(iter(data.train_dataloader()))

    x = X[0]
    y = Y[0]

    # nbar_signal = (1e3, 1e4)
    # nbar_bkgrnd = (0, 1)
    #
    # frame_to_eigen = FramesToEigenvalues(nbar_signal, nbar_bkgrnd)
    # import time
    # tic = time.time()
    # Y = frame_to_eigen(X.to('cuda')).cpu()
    # print('frame_to_eigen:', time.time() - tic)

    #
    # # Normalize by mean of sum of frames
    # X = X / torch.mean(torch.sum(X, axis=(-2, -1)))
    #
    # # Scale to nbar total counts each frame
    # signal_levels = torch.randint(low=int(nbar_signal[0]), high=int(nbar_signal[1]) + 1, size=(X.shape[0],), device=X.device)
    # X = X * signal_levels.view(X.shape[0], 1, 1, 1)
    #
    # # Add flat background to all frames
    # bkgrnd_levels = torch.randint(low=int(nbar_bkgrnd[0]), high=int(nbar_bkgrnd[1]) + 1, size=(X.shape[0],), device=X.device) / (X.shape[-2] * X.shape[-1])
    # X = X + bkgrnd_levels.view(X.shape[0], 1, 1, 1)
    #
    # # Poisson sampling
    # X = torch.poisson(X)
    #
    # # SVD
    # xflat = torch.flatten(X, start_dim=2)
    # batch_size, nframes, Nx, Ny = X.shape
    # U, S, Vh = torch.linalg.svd(xflat)
    # zsin = torch.reshape(Vh[:, 1, :], (batch_size, Nx, Ny))
    # zcos = torch.reshape(Vh[:, 2, :], (batch_size, Nx, Ny))


    # test = batch[0][0].numpy()
    # plot_frames(x, nrows=4, figsize=(4, 4), dpi=150, cmap="viridis")
    # start = time.time()
    # (x, y) = data.train_set.__getitem__(1)
    # print(f'Time: {time.time() - start}')
    # print('fin')