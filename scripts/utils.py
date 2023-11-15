import torch
import os
import pickle
import scipy.linalg

from QIML.visualization.figure_utils import *
from QIML.models.utils import SSIM
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
