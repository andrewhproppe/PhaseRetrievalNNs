import numpy as np

import torch
from matplotlib import pyplot as plt

# from QIML.models.utils import SSIM
# from torch.nn import MSELoss
# from QIML.models.QI_models import SRN3D_v3
from QIML.pipeline.transforms import input_transform_pipeline

# from utils import compute_svd_loss, compute_model_loss, save_pickle_with_auto_increment

from QIML.visualization.visualize import plot_frames
import h5py

# import matplotlib as mpl
# mpl.use("TkAgg")

if __name__ == "__main__":
    # Load experimental data set
    root = "../data/expt"
    date = "20230808"
    data_fname = "raw_frames_0.01ms.npy"
    bg_fname = "bg_frames_0.01ms.npy"
    data = np.load(f"{root}/{date}/{data_fname}").astype(np.float32)
    bg = np.load(f"{root}/{date}/{bg_fname}").astype(np.float32)
    # transforms = input_transform_pipeline()
    # x_expt = transforms(torch.tensor(data[0, :, :, :]))
    idx = 30
    x_expt = data[idx, :, :, :]
    bg_expt = bg[idx, :, :, :]
    # plot_frames(x_expt, nrows=4, figsize=(4, 4), dpi=150, cmap="viridis")
    # plot_frames(bg_expt, nrows=3, figsize=(4, 4), dpi=150, cmap="viridis")
    # plot_frames(x_expt - bg_expt, nrows=3, figsize=(4, 4), dpi=150, cmap="viridis")
    del data, bg

    # raise RuntimeError

    # Simulate experimental data
    path = "../data/raw/flowers_n5000_npix64.h5"
    masks = h5py.File(path, "r")["truths"][0:100, :, :]
    E1 = torch.Tensor(h5py.File(path, "r")["E1"][0])
    E2 = E1
    vis = 1
    # nbar_signal = 0.1e5
    # nbar_bkgrnd = 1.3e6
    nbar_signal = 2e3
    nbar_bkgrnd = 1.3e1
    npixels = 64 * 64

    nframes = 32
    y = torch.Tensor(masks[idx, :, :])
    phi = torch.rand(32) * 2 * torch.pi  # generate array of phi values
    phase_mask = y.repeat(nframes, 1, 1)
    phase_mask = phase_mask + phi.unsqueeze(-1).unsqueeze(-1)
    x = (
        torch.abs(E1) ** 2
        + torch.abs(E2) ** 2
        + 2 * vis * torch.abs(E1) * torch.abs(E2) * torch.cos(phase_mask)
    )
    x = x / x[0, :, :].sum()

    x = x * nbar_signal
    x = x + nbar_bkgrnd / npixels

    plot_frames(phase_mask, nrows=4, figsize=(4, 4), dpi=150, cmap="viridis")

    plot_frames(x, nrows=4, figsize=(4, 4), dpi=150, cmap="viridis")

    x = torch.poisson(x)
    plot_frames(x, nrows=4, figsize=(4, 4), dpi=150, cmap="viridis")
