import numpy as np
import time

import torch
import pickle
from matplotlib import pyplot as plt

from QIML.models.utils import SSIM
from torch.nn import MSELoss
from QIML.models.QI_models import SRN3D_v3
from QIML.pipeline.transforms import input_transform_pipeline
from utils import norm_to_phase, phase_to_norm, frames_to_svd

from QIML.visualization.visualize import plot_frames
from QIML.visualization.AP_figs_funcs import *
from tqdm import tqdm


class PhaseImages:
    def __init__(
        self,
        acq_time: float,
        date: str,
    ):
        self.acq_time = acq_time
        self.date = date
        self.data = None
        self.bkgd = None
        self.y_true = None
        self.y_nn = None
        self.y_svd = None
        self.nn_mse = None
        self.nn_ssim = None
        self.svd_mse = None
        self.svd_ssim = None
        self.mse = MSELoss()
        self.ssim = SSIM()
        self.transforms = input_transform_pipeline()

    def load_expt_data(self, idx=None, root="../data/expt"):
        acq_time = self.acq_time
        date = self.date
        data_fname = f"raw_frames_{acq_time}ms.npy"
        bg_fname = f"bg_frames_{acq_time}ms.npy"
        y_true_fname = f"theory_phase_{acq_time}ms.npy"
        svd_fname = f"SVD_phase_{acq_time}ms.npy"
        data = np.load(f"{root}/{date}/{data_fname}").astype(np.float32)
        bg = np.load(f"{root}/{date}/{bg_fname}").astype(np.float32)
        y_true = np.load(f"{root}/{date}/{y_true_fname}").astype(np.float32)
        svd = np.load(f"{root}/{date}/{svd_fname}").astype(np.float32)
        if idx is not None:
            data = data[:idx, :, :, :]
            bg = bg[:idx, :, :, :]
            y_true = y_true[:idx, :, :]
            svd = svd[:idx, :, :]
        self.data = torch.tensor(data)
        self.bkgd = torch.tensor(bg)
        self.y_true = torch.tensor(y_true)
        self.y_svd = torch.tensor(svd)

    def phase_to_norm(self):
        self.y_true /= 2 * torch.pi
        self.y_svd /= 2 * torch.pi

    def model_reconstructions(self, model):
        tic = time.time()
        # with torch.no_grad():
        #     self.y_nn, _ = model(self.transforms(self.data))
        reconstructions = []

        with torch.no_grad():
            for i, x in enumerate(tqdm(self.data)):
                x = self.transforms(x)
                y_true = self.y_true[i, :, :]
                y_nn, _ = model(x.unsqueeze(0))
                y_nn = y_nn.squeeze(0).cpu().detach()
                # y_nn = norm_to_phase(y_nn)

                loss1 = torch.nn.L1Loss()(torch.Tensor(y_true), torch.Tensor(y_nn))
                loss2 = torch.nn.L1Loss()(torch.Tensor(y_true), torch.Tensor(-y_nn))
                if loss2 < loss1:
                    y_nn = -y_nn
                reconstructions.append(y_nn)

        self.y_nn = torch.stack(reconstructions, dim=0)

        print(f"Time elapsed: {time.time()-tic:.2f} s")

    def compute_losses(self):
        self.nn_mse = []
        self.nn_ssim = []
        self.svd_mse = []
        self.svd_ssim = []
        for y_nn, y_true, y_svd in zip(self.y_nn, self.y_true, self.y_svd):
            self.nn_mse.append(self.mse(y_nn, y_true))
            self.nn_ssim.append(
                1
                - self.ssim(
                    y_nn.unsqueeze(0).unsqueeze(0), y_true.unsqueeze(0).unsqueeze(0)
                )
            )
            self.svd_mse.append(self.mse(y_svd, y_true))
            self.svd_ssim.append(
                1
                - self.ssim(
                    y_svd.unsqueeze(0).unsqueeze(0), y_true.unsqueeze(0).unsqueeze(0)
                )
            )

    def plot_phase_images(
        self, idx=None, cmap="twilight_shifted", figsize=(8, 4), dpi=150
    ):
        fig, ax = plt.subplots(1, 4, figsize=figsize, dpi=dpi)
        ax = ax.flatten()
        ax[0].imshow((self.data[idx, 0, :, :]), cmap=cmap)
        ax[0].set_title(f"Input ({self.acq_time} ms)")
        ax[1].imshow(self.y_true[idx, :, :], cmap=cmap)
        ax[1].set_title("True")
        ax[2].imshow(self.y_nn[idx, :, :], cmap=cmap)
        ax[2].set_title("NN")
        ax[3].imshow(self.y_svd[idx, :, :], cmap=cmap)
        ax[3].set_title("SVD")
        dress_fig(legend=False)


# Load trained model and set to eval
model = SRN3D_v3.load_from_checkpoint(
    "../trained_models/SRN3D_bg2.ckpt", map_location=torch.device("cpu")
).eval()

# Load experimental data set and SVD phase
PI = PhaseImages(acq_time=1, date="20230808")
PI.load_expt_data(idx=5)
PI.model_reconstructions(model)
PI.phase_to_norm()
PI.compute_losses()
PI.plot_phase_images(idx=0)


# # Save results as pickle files, to avoid recomputing every time for analysis
# fname = f"PhaseImages_{PI.acq_time}ms_{PI.date}.pickle"
# with open(fname, "wb") as file:
#     pickle.dump(PI, file)

#
