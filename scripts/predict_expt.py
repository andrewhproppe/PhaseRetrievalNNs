import numpy as np
import torch
from matplotlib import pyplot as plt
from QIML.models.utils import SSIM
from torch.nn import MSELoss
from QIML.models.QI_models import SRN3D_v3
from QIML.pipeline.transforms import input_transform_pipeline
from utils import compute_svd_loss, compute_model_loss, save_pickle_with_auto_increment

import matplotlib as mpl

mpl.use("TkAgg")

if __name__ == "__main__":
    # Load trained model and set to eval
    model = SRN3D_v3.load_from_checkpoint("../trained_models/SRN3Dv3_optim.ckpt").cuda()
    model.eval()

    # Load experimental data set
    root = "../data/expt"
    date = "20230804_32frames"
    fname = "expData_0.1ms_exp_2023_08_04.npy"
    data = np.load(f"{root}/{date}/{fname}").astype(np.float32)

    transforms = input_transform_pipeline()
    idx = 0
    x = transforms(torch.tensor(data[idx, :, :, :]).unsqueeze(0))
    # bckgrnd = torch.mean(x, dim=1)
    # x -= bckgrnd.unsqueeze(1)

    y, z = model(x.cuda())
    y = y.squeeze(0).cpu().detach().numpy()

    cmap = "viridis"
    fig, ax = plt.subplots(1, 2, figsize=(6, 2), dpi=150)
    ax[0].imshow(x[0, 0, :, :], cmap=cmap)
    ax[1].imshow(y[:, :], cmap=cmap)
