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
    data_fname = "raw_frames_0.1ms.npy"
    bg_fname = "bg_frames_0.1ms.npy"
    data = np.load(f"{root}/{date}/{data_fname}").astype(np.float32)
    bg = np.load(f"{root}/{date}/{bg_fname}").astype(np.float32)
    # transforms = input_transform_pipeline()
    # x_expt = transforms(torch.tensor(data[0, :, :, :]))
    x_expt = data[0, :, :, :]
    bg_expt = bg[0, :, :, :]
    plot_frames(x_expt, nrows=3, figsize=(4, 4), dpi=150, cmap="viridis")
    # plot_frames(bg_expt, nrows=3, figsize=(4, 4), dpi=150, cmap="viridis")
    # plot_frames(x_expt - bg_expt, nrows=3, figsize=(4, 4), dpi=150, cmap="viridis")
    del data, bg

    # raise RuntimeError

    # Simulate experimental data
    path = "/Users/andrewproppe/Library/CloudStorage/GoogleDrive-andrew.proppe@gmail.com/My Drive/python/QuantumImagingML/data/raw/flowers_n5000_npix64.h5"
    masks = h5py.File(path, "r")["truths"][0:100, :, :]
    E1 = torch.Tensor(h5py.File(path, "r")["E1"][0])
    E2 = E1
    vis = 1
    nbar_signal = 1e5
    nbar_bkgrnd = 1e6
    npixels = 64 * 64

    nframes = 32
    y = torch.Tensor(masks[0, :, :])
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
    x = torch.poisson(x)
    plot_frames(x, nrows=3, figsize=(4, 4), dpi=150, cmap="viridis")
    raise RuntimeError

    factor = 20
    bckgrnd = 0.5
    x = torch.poisson(bckgrnd * factor + factor * x)
    x = transforms(x)
    plt.imshow(x[1, :, :])

    # xmin = torch.amin(x, dim=(1, 2))
    # x = x - xmin.unsqueeze(-1).unsqueeze(-1)

    #
    # x_maxima = torch.sum(x, axis=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
    # scale = nbar / x[0, :, :].sum()
    # x = x * scale
    # x = x + 5 * scale
    # x = torch.poisson(x)

    plot_frames(x, nrows=5, figsize=(4, 4), dpi=150, cmap="viridis")
    # plt.imshow(torch.poisson(300 + 100 * y / y.max()))

    raise RuntimeError

    transforms = input_transform_pipeline()
    idx = 0
    x = transforms(torch.tensor(data[idx, :, :, :]).unsqueeze(0))
    # bckgrnd = torch.mean(x, dim=1)
    # x -= bckgrnd.unsqueeze(1)

    # Load trained model and set to eval
    model = SRN3D_v3.load_from_checkpoint("../trained_models/SRN3Dv3_optim.ckpt").cuda()
    model.eval()

    y, z = model(x.cuda())
    y = y.squeeze(0).cpu().detach().numpy()

    cmap = "viridis"
    fig, ax = plt.subplots(1, 2, figsize=(6, 2), dpi=150)
    ax[0].imshow(x[0, 0, :, :], cmap=cmap)
    ax[1].imshow(y[:, :], cmap=cmap)
