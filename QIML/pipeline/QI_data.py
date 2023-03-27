from functools import lru_cache
from typing import Tuple, Type, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
# from torchvision.transforms import Compose, ToTensor
import pytorch_lightning as pl

from QIML.pipeline import transforms
from QIML.utils import paths


class H5Dataset(Dataset):
    def __init__(self, filepath: str, seed: int = 10236, **kwargs):
        super().__init__()
        self._filepath = filepath
        self.truth_transform = transforms.truth_transform_pipeline()
        self.input_transform = transforms.input_transform_pipeline()

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
        inputs_shape = self.data["inputs"].shape
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



class QIDataModule(pl.LightningDataModule):
    def __init__(
        self, h5_path: Union[None, str] = None, batch_size: int = 64, seed: int = 120516, num_workers=0, **kwargs
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
        full_dataset = H5Dataset(self.h5_path, **self.data_kwargs)
        # use 10% of the data set a test set
        test_size = int(len(full_dataset) * 0.2)
        self.train_set, self.val_set = random_split(
            full_dataset,
            [len(full_dataset) - test_size, test_size],
            torch.Generator().manual_seed(self.seed),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
            # collate_fn=transforms.pad_collate_func,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers
            # collate_fn=transforms.pad_collate_func,
        )


def get_test_batch(batch_size: int = 32, h5_path: Union[None, str] = None, seed: int = 120516) -> Tuple[torch.Tensor]:
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

# Testing
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from torch import nn
    # data_fname = 'image_data_n10_nbar10000_nframes16_npix32.h5'
    data_fname = 'QI_devset.h5'

    data = QIDataModule(data_fname, batch_size=10)
    data.setup()
    (x, y) = data.train_set.__getitem__(1)

    print('fin')
    # input = torch.randn(20, 1, 100, 140)
    # from QIML.models.base import Conv2DAutoEncoder
    # ae = Conv2DAutoEncoder(kernel1=15, kernel2=3)
    # output = ae.forward(input)[0]