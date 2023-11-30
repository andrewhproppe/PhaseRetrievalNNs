import time

import numpy
import torch
import os
import pickle
import scipy.linalg

from PRNN.pipeline.transforms import input_transform_pipeline
from scipy.optimize import minimize
from PRNN.visualization.figure_utils import *
from PRNN.models.utils import SSIM
from torch.nn import MSELoss
from tqdm import tqdm

def frames_to_svd(x):
    xflat = torch.flatten(x, start_dim=1).numpy()
    Nx, Ny = x.shape[1:]
    U, S, Vh = scipy.linalg.svd(xflat)
    zsin = np.reshape(Vh[1, :], (Nx, Ny))
    zcos = np.reshape(Vh[2, :], (Nx, Ny))
    z1 = zcos + 1j * zsin
    z2 = zsin + 1j * zcos
    phi1 = np.angle(z1)
    phi2 = np.angle(z2)
    return torch.tensor(phi1), torch.tensor(phi2)


def norm_to_phase(x):
    # return x * 2 * torch.pi - torch.pi
    return x * 2 * torch.pi


def phase_to_norm(x):
    # return (x + torch.pi) / (2 * torch.pi)
    return (x) / (2 * torch.pi)


def compute_svd_loss(X, Y_true, loss=MSELoss(), ssim=SSIM()):
    phis, svd_mse, svd_ssim = [], [], []
    for x, y_true in tqdm(zip(X, Y_true)):
        phi1, phi2 = frames_to_svd(norm_to_phase(x))
        phi1, phi2 = torch.tensor(phi1), torch.tensor(phi2)
        l1, l2 = loss(phase_to_norm(phi1), y_true), loss(phase_to_norm(phi2), y_true)
        phi, l_mse = (phi1, l1) if l1 < l2 else (phi2, l2)
        l_ssim = 1 - ssim(phase_to_norm(phi)[None, None], y_true[None, None])
        phis.append(phi)
        svd_mse.append(l_mse)
        svd_ssim.append(l_ssim)
    return torch.stack(phis), torch.tensor(svd_mse), torch.tensor(svd_ssim)


def compute_model_loss(X, Y_true, model, loss=MSELoss(), ssim=SSIM()):
    with torch.no_grad():
        Y_pred, Z = model(X.cuda())

    nn_mse, nn_ssim = [], []
    for y, y_true in tqdm(zip(Y_pred, Y_true.cuda())):
        nn_mse.append(loss(y, y_true).cpu())
        nn_ssim.append((1 - ssim(y[None, None], y_true[None, None])).cpu())

    Y_pred = norm_to_phase(Y_pred.cpu())
    # Set all devices to cpu and make list of tensors to match svd output
    return Y_pred, torch.tensor(nn_mse), torch.tensor(nn_ssim)


def save_pickle_with_auto_increment(filename, data, rootpath="../data/predictions/"):
    batch_number = 0
    while True:
        full_filename = f"{rootpath}{filename}_batch{batch_number}.pickle"
        if not os.path.exists(full_filename):
            with open(full_filename, "wb") as file:
                pickle.dump(data, file)
            print(f"Data saved as {full_filename}")
            break
        batch_number += 1


def pick_phi(phi1, phi2, y_true):
    l1 = torch.nn.L1Loss()(phi1, torch.Tensor(y_true))
    l2 = torch.nn.L1Loss()(phi2, torch.Tensor(y_true))
    phi, l = (phi1, l1) if l1 < l2 else (phi2, l2)
    return phi


def get_tensor_memory_usage(tensor):
    """
    Get the memory usage of a PyTorch tensor in bytes.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        int: The memory usage of the tensor in bytes.
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor.")

    return tensor.element_size() * tensor.nelement()



class PhaseImages:
    def __init__(
            self,
            acq_time: float = None,
            date: str = None,
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

    def load_sim_data(self, X, Y):
        self.data = X
        self.y_true = Y

    def phase_to_norm(self):
        self.y_true /= 2 * torch.pi
        self.y_svd /= 2 * torch.pi

    def model_reconstructions(self, model):
        tic = time.time()
        reconstructions = []
        recon_times = []
        with torch.no_grad():
            for i, x in enumerate(tqdm(self.data, desc="Computing reconstructions")):
                recon_tic = time.time()
                x = self.transforms(x)
                x = x.to(model.device)
                y_true = self.y_true[i, :, :]
                y_nn, _ = model(x.unsqueeze(0))
                recon_times.append(time.time() - recon_tic)
                y_nn = y_nn.squeeze(0).cpu().detach()
                # y_nn = norm_to_phase(y_nn)
                loss1 = torch.nn.L1Loss()(torch.Tensor(y_true), torch.Tensor(y_nn))
                loss2 = torch.nn.L1Loss()(torch.Tensor(y_true), torch.Tensor(-y_nn))
                if loss2 < loss1:
                    y_nn = -y_nn
                reconstructions.append(y_nn)

        self.y_nn = torch.stack(reconstructions, dim=0)
        self.avg_nn_recon_time = np.mean(recon_times)
        print(f"\nTime elapsed: {time.time() - tic:.2f} s")
        print(f"Average reconstruction time: {self.avg_nn_recon_time:.4f} s")

    def svd_reconstructions(self):
        tic = time.time()
        phis = []
        recon_times = []
        for x, y_true in tqdm(zip(self.data, self.y_true)):
            recon_tic = time.time()
            phi1, phi2 = frames_to_svd(norm_to_phase(x))
            recon_times.append(time.time() - recon_tic)
            phi1, phi2 = torch.tensor(phi1), torch.tensor(phi2)
            l1, l2 = torch.nn.L1Loss()(phase_to_norm(phi1), y_true), torch.nn.L1Loss()(phase_to_norm(phi2), y_true)
            phi, l_mse = (phi1, l1) if l1 < l2 else (phi2, l2)
            phis.append(phase_to_norm(phi))

        self.y_svd = torch.stack(phis, dim=0)
        self.avg_svd_recon_time = np.mean(recon_times)

        print(f"\nTime elapsed: {time.time() - tic:.2f} s")
        print(f"Average reconstruction time: {self.avg_svd_recon_time:.4f} s")

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

            phis[i] = phase_to_norm(torch.tensor(result[0]))

        if type == "nn":
            self.y_nn = phis
        elif type == "svd":
            self.y_svd = phis

            # self.y_nn[i] = phase_to_norm(torch.tensor(result[0]))

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
        dress_fig(tight=True, xlabel='$\it{x}$', ylabel='$\it{y}$')

    def error_histograms(self, nbins=200, x_min='auto', x_max='auto'):
        min_error = min(min(self.nn_mse, self.svd_mse)) if x_min == 'auto' else x_min
        max_error = max(max(self.nn_mse, self.svd_mse)) if x_max == 'auto' else x_max
        bins = torch.linspace(min_error, max_error, nbins)
        self.nn_mse_histo = torch.histogram(torch.tensor(self.nn_mse), bins=bins)
        self.svd_mse_histo = torch.histogram(torch.tensor(self.svd_mse), bins=bins)

    def save(self, fname=None, root="../data/expt"):
        if fname is None:
            fname = f"PhaseImages_{self.acq_time}ms_{self.date}_{len(self.y_true)}n.pickle"
        with open(f"{root}/{fname}", "wb") as file:
            pickle.dump(self, file)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def optimize_global_phases(y: np.array, yhat: np.array) -> np.array:
    """
    Optimize the global phase of yhat to minimize the error between y and yhat.
    params:
        y: true phase (in units of rads)
        yhat: predicted phase (in units of rads)
    """
    def to_minimize(z, phi):

        error = (y - np.mod(phi - z, 2 * np.pi)).flatten()

        error2 = np.zeros(len(error))

        for k in range(len(error)):

            if abs(-2 * np.pi - error[k]) < abs(error[k]):
                error2[k] = error[k] + 2 * np.pi

            elif abs(2 * np.pi - error[k]) < abs(error[k]):
                error2[k] = error[k] - 2 * np.pi

            else:
                error2[k] = error[k]

        return np.sum(abs(error2))

    res1 = minimize(to_minimize, np.pi, args=yhat)
    res2 = minimize(to_minimize, np.pi, args=-yhat)

    norm_error = np.sum(
        np.mod(
            abs(np.random.uniform(0, 2 * np.pi, yhat.shape) - y),
            2 * np.pi,
        )
    )

    if res2.fun < res1.fun:
        result = (
            np.mod(-yhat - res2.x[0], 2 * np.pi),
            y,
            res2.fun / norm_error,
        )

    else:
        result = (
            np.mod(yhat - res1.x[0], 2 * np.pi),
            y,
            res1.fun / norm_error,
        )

    return result
