import numpy as np

import torch
from matplotlib import pyplot as plt

from QIML.models.utils import SSIM
from torch.nn import MSELoss
from QIML.models.QI_models import SRN3D_v3
from QIML.pipeline.transforms import input_transform_pipeline
from utils import norm_to_phase, phase_to_norm, frames_to_svd

from utils import compute_svd_loss, compute_model_loss, save_pickle_with_auto_increment

from QIML.visualization.visualize import plot_frames
from QIML.visualization.AP_figs_funcs import *
import h5py


def load_expt_data(acq_time, date, idx=None, root="../data/expt"):
    data_fname = f"raw_frames_{acq_time}ms.npy"
    bg_fname = f"bg_frames_{acq_time}ms.npy"
    y_true_fname = f"theory_phase_{acq_time}ms.npy"
    data = np.load(f"{root}/{date}/{data_fname}").astype(np.float32)
    bg = np.load(f"{root}/{date}/{bg_fname}").astype(np.float32)
    y_true = np.load(f"{root}/{date}/{y_true_fname}").astype(np.float32)
    if idx is not None:
        data = data[idx, :, :, :]
        bg = bg[idx, :, :, :]
        y_true = y_true[idx, :, :]
    return data, y_true, bg


def pick_phi(phi1, phi2, y_true):
    l1 = torch.nn.L1Loss()(phi1, torch.Tensor(y_true))
    l2 = torch.nn.L1Loss()(phi2, torch.Tensor(y_true))
    phi, l = (phi1, l1) if l1 < l2 else (phi2, l2)
    return phi


if __name__ == "__main__":
    # Load experimental data set
    x, y_true, bkgrnd = load_expt_data(acq_time=0.1, idx=10, date="20230808")
    y_true -= torch.pi

    transforms = input_transform_pipeline()
    x = transforms(torch.tensor(x))

    # Load trained model and set to eval
    model = SRN3D_v3.load_from_checkpoint("../trained_models/SRN3D_bg1.ckpt")
    model.eval()

    # Get model prediction
    y, z = model(x.unsqueeze(0))
    y = y.squeeze(0).cpu().detach().numpy()
    y = norm_to_phase(y)

    # Get SVD prediction
    phi1, phi2 = frames_to_svd(norm_to_phase(x))
    phi = pick_phi(phi1, phi2, y_true)

    set_font_size(8)
    cmap = "hsv"
    fig, ax = plt.subplots(1, 4, figsize=(8, 4), dpi=150)
    ax = ax.flatten()
    ax[0].imshow(x[0, :, :], cmap=cmap)
    ax[0].set_title("Input")
    ax[1].imshow(y_true[:, :], cmap=cmap)
    ax[1].set_title("True")
    ax[2].imshow(y[:, :], cmap=cmap)
    ax[2].set_title("NN")
    ax[3].imshow(phi[:, :], cmap=cmap)
    ax[3].set_title("SVD")
    dress_fig(legend=False)
