import time
import torch
from QIML.models.utils import SSIM
from torch.nn import MSELoss
from QIML.models.QI_models import SRN3D_v3
from QIML.pipeline.transforms import input_transform_pipeline
from QIML.utils import get_system_and_backend
from utils import norm_to_phase, phase_to_norm, frames_to_svd
from scipy.optimize import minimize

from QIML.visualization.visualize import plot_frames
from QIML.visualization.AP_figs_funcs import *
from tqdm import tqdm
get_system_and_backend()


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

    def load_expt_data(self, idx=None, root="../data/expt", background=False):
        acq_time = self.acq_time
        date = self.date
        data_fname = f"raw_frames_{acq_time}ms.npy"
        y_true_fname = f"theory_phase_{acq_time}ms.npy"
        svd_fname = f"SVD_phase_{acq_time}ms.npy"
        data = np.load(f"{root}/{date}/{data_fname}").astype(np.float32)
        y_true = np.load(f"{root}/{date}/{y_true_fname}").astype(np.float32)
        svd = np.load(f"{root}/{date}/{svd_fname}").astype(np.float32)

        # test = np.memmap(f"{root}/{date}/{data_fname}", dtype='float32', mode='r')
        # shape = (10, 32, 64, 64)
        # dataa = test[: shape[0] * np.prod(shape[1:])].reshape(shape)

        if idx is not None:
            data = data[:idx, :, :, :]
            y_true = y_true[:idx, :, :]
            svd = svd[:idx, :, :]

        if background:
            bg_fname = f"bg_frames_{acq_time}ms.npy"
            bg = np.load(f"{root}/{date}/{bg_fname}").astype(np.float32)
            if idx is not None:
                bg = bg[:idx, :, :, :]
            self.bkgd = torch.tensor(bg)

        self.data = torch.tensor(data)
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
            for i, x in enumerate(tqdm(self.data, desc="Computing reconstructions")):
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

        print(f"\nTime elapsed: {time.time()-tic:.2f} s")

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

    def optimize_global_phases(self, type="nn"):
        if type == "nn":
            phis = self.y_nn
        elif type == "svd":
            phis = self.y_svd
        else:
            raise ValueError(f"Invalid type: {type}")

        for i in tqdm(range(len(phis)), desc="Optimizing global phases"):
            theory_phase = norm_to_phase(np.array(self.y_true[i]))
            phi = norm_to_phase(np.array(phis[i]))

            def to_minimize(z, phi):

                error = (theory_phase - np.mod(phi - z, 2 * np.pi)).flatten()

                error2 = np.zeros(len(error))

                for k in range(len(error)):

                    if abs(-2 * np.pi - error[k]) < abs(error[k]):
                        error2[k] = error[k] + 2 * np.pi

                    elif abs(2 * np.pi - error[k]) < abs(error[k]):
                        error2[k] = error[k] - 2 * np.pi

                    else:
                        error2[k] = error[k]

                return np.sum(abs(error2))

            res1 = minimize(to_minimize, np.pi, args=phi)
            res2 = minimize(to_minimize, np.pi, args=-phi)

            norm_error = np.sum(
                np.mod(
                    abs(np.random.uniform(0, 2 * np.pi, phi.shape) - theory_phase),
                    2 * np.pi,
                )
            )

            if res2.fun < res1.fun:
                result = (
                    np.mod(-phi - res2.x[0], 2 * np.pi),
                    theory_phase,
                    res2.fun / norm_error,
                )

            else:
                result = (
                    np.mod(phi - res1.x[0], 2 * np.pi),
                    theory_phase,
                    res1.fun / norm_error,
                )

            self.y_nn[i] = phase_to_norm(torch.tensor(result[0]))

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
    checkpoint_path="../trained_models/SRN3D_expt.ckpt",
    # map_location=torch.device("cpu")
).eval()

# Load experimental data set and SVD phase
PI = PhaseImages(acq_time=0.1, date="20230829")
PI.load_expt_data(idx=1000)
PI.model_reconstructions(model)
PI.phase_to_norm()
PI.optimize_global_phases()
PI.compute_losses()
PI.plot_phase_images(idx=2)

# Histogram of errors
nbins = 200
min_error = min(min(PI.nn_mse, PI.svd_mse))
max_error = max(max(PI.nn_mse, PI.svd_mse))
bins = torch.linspace(min_error, max_error, nbins)
nn_mse_histo = torch.histogram(torch.tensor(PI.nn_mse), bins=bins)
svd_mse_histo = torch.histogram(torch.tensor(PI.svd_mse), bins=bins)
fig, ax = plt.subplots(1, 2, figsize=(6, 2), dpi=150)
ax[0].bar(nn_mse_histo[1][:-1], nn_mse_histo[0], width=nn_mse_histo[1][1]-nn_mse_histo[1][0], linewidth=0.1, edgecolor='black')
ax[0].set_title("SRN3D")
ax[0].set_yscale('log')
ax[1].bar(svd_mse_histo[1][:-1], svd_mse_histo[0], width=svd_mse_histo[1][1]-svd_mse_histo[1][0], linewidth=0.1, edgecolor='black')
ax[1].set_yscale('log')
ax[1].set_title("SVD")
dress_fig(tight=True, xlabel='MSE', ylabel='Counts')
# fig, ax = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
# ax = ax.flatten()
# ax[0].imshow(norm_to_phase(PI.y_true[0, :, :]), cmap="twilight_shifted")
# ax[0].set_title("True")
# ax[1].imshow(norm_to_phase(PI.y_nn[0, :, :]), cmap="twilight_shifted")
# ax[1].set_title("NN")
# ax[2].imshow(new_phi, cmap="twilight_shifted")
# ax[2].set_title("NN + global phase")

# # Save results as pickle files, to avoid recomputing every time for analysis
# fname = f"PhaseImages_{PI.acq_time}ms_{PI.date}.pickle"
# with open(fname, "wb") as file:
#     pickle.dump(PI, file)
