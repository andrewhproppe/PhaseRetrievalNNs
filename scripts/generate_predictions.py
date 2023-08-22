import pickle
import time
import torch

from QIML.pipeline.QI_data import QIDataModule
from matplotlib import pyplot as plt
from QIML.models.utils import SSIM
from torch.nn import MSELoss
from QIML.models.QI_models import SRN3D_v3
from utils import compute_svd_loss, compute_model_loss, save_pickle_with_auto_increment


if __name__ == "__main__":
    # Load trained model and set to eval
    # model = SRN3D_v3.load_from_checkpoint("../trained_models/SRN3Dv3_optim.ckpt").cuda()
    model = SRN3D_v3.load_from_checkpoint(
        "../trained_models/SRN3D_bg4.ckpt", map_location=torch.device("cpu")
    )
    model.eval()

    # raise RuntimeError

    # Define loss functions
    mse = MSELoss()
    ssim = SSIM()

    # Load data set
    data_fname = "flowers_n5000_npix64.h5"
    batch_size = 250
    nframes = 32
    data = QIDataModule(
        data_fname,
        batch_size=batch_size,
        num_workers=0,
        nbar_signal=(0.1e5, 2e5),
        nbar_bkgrnd=(1e6, 1.3e6),
        nframes=nframes,
        shuffle=True,
        randomize=True,
    )
    # data = QIDataModule(data_fname, batch_size=batch_size, num_workers=0, nbar=(1e3, 2e3), nframes=nframes, shuffle=True, randomize=False)
    data.setup()

    # raise RuntimeError

    # Prepare batches of data
    batch = next(iter(data.val_dataloader()))
    X, Y_true = batch[0], batch[1]

    # Get SRN3D reconstructions and errors
    tic2 = time.time()
    nn_phi, nn_mse, nn_ssim = compute_model_loss(X, Y_true, model)
    print(f"Time elapsed: {time.time()-tic2:.2f} s")

    # Get SVD reconstructions and errors
    tic = time.time()
    svd_phi, svd_mse, svd_ssim = compute_svd_loss(X, Y_true)
    print(f"Time elapsed: {time.time()-tic:.2f} s")

    # Save results as pickle files, to avoid recomputing every time for analysis
    save_pickle_with_auto_increment(
        f"NN_bs{batch_size}_nf{nframes}", (nn_phi, nn_mse, nn_ssim)
    )

    save_pickle_with_auto_increment(
        f"SVD_bs{batch_size}_nf{nframes}", (svd_phi, svd_mse, svd_ssim)
    )

    save_pickle_with_auto_increment(f"True_bs{batch_size}_nf{nframes}", (Y_true))
