import torch
import numpy as np
import scipy.linalg
import time
from QIML.pipeline.QI_data import QIDataModule
from QIML.visualization.AP_figs_funcs import *
from matplotlib import pyplot as plt

if __name__ == '__main__':
    from QIML.models.QI_models import QI3Dto2DConvAE, SRN3D
    model = SRN3D.load_from_checkpoint("../trained_models/SRN3D.ckpt")

    data_fname = 'flowers_n5000_npix64.h5'
    data = QIDataModule(data_fname, batch_size=1, num_workers=0, nbar=(1e3, 2e3), nframes=64, shuffle=False, randomize=False)
    data.setup()
    batch = next(iter(data.val_dataloader()))
    X = batch[0]
    Y_true = batch[1]

    tic = time.time()
    with torch.no_grad():
        Y, Z = model(X)
    print(f"Time elapsed: {time.time()-tic:.2f} s")

    """ SVD """
    def frames_to_svd(X):
        xflat = torch.flatten(X.squeeze(0), start_dim=1).numpy()
        Nx, Ny = X.shape[2:]
        # raise ValueError("This is not working")
        U, S, Vh = scipy.linalg.svd(xflat)
        zsin = np.reshape(Vh[1, :], (Nx, Ny))
        zcos = np.reshape(Vh[2, :], (Nx, Ny))
        z = zcos + 1j * zsin
        phi = np.angle(z)
        return phi

    tic2 = time.time()
    phi = frames_to_svd(X)
    print(f"Time elapsed: {time.time()-tic2:.2f} s")



    fig, ax = plt.subplots(1, 3, figsize=(6, 2), dpi=150)
    ax[0].imshow(Y_true.squeeze(0), cmap='viridis')
    ax[0].set_title("True")
    ax[1].imshow(phi, cmap='viridis')
    ax[1].set_title("SVD")
    ax[2].imshow(Y.squeeze(0), cmap='viridis')
    ax[2].set_title("SRN3D")
    dress_fig(tight=True, xlabel='x pix.', ylabel='y pix.')