import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from torch import nn

from QIML.models.ode.ode import G2ODE
from QIML.models.ode.ode_models import MLPStack, Conv1DStack, SmallMLP
from QIML.pipeline.image_data import ODEDataModule

pl.seed_everything(42)

nchannels = 5 # number of g2s to use for solving initial value problem
dim       = 116
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# pl.seed_everything(21656)
# dm = ODEDataModule("pcfs_g2_2d_n128_wPrior.h5", batch_size=32, nchannels=nchannels, nsteps=(50, 75), add_noise=False)
dm = ODEDataModule("pcfs_g2_2d_n1_wPrior.h5", batch_size=1, nchannels=nchannels, nsteps=(15, 25), add_noise=False)

# model_kwargs = {"out_dim": 100, "depth": 3, "num_heads": 4, "activation": nn.SiLU}

# SmallMLP kwargs
model_kwargs = {
    "out_dim": dim,
    "nchannels": nchannels,
    "activation": nn.Tanh
}


# model_kwargs = {
#     "out_dim": dim,
#     "depth": 2,
#     "activation": nn.SiLU,
#     "residual": False,
#     "output_activation": nn.SiLU,
# }

# Conv kwargs
# model_kwargs = {
#     "out_dim": dim,
#     "depth": 4,
#     "input_channels":  channels,
#     "output_channels": 32,
#     "kernel": 3,
#     "stride": 2,
#     "activation": nn.SiLU,
#     "residual": True,
#     "output_activation": nn.SiLU,
#  }

ode = G2ODE(
    SmallMLP,
    # MLPStack,
    # Conv1DStack,
    model_kwargs,
    input_shape=[nchannels, dim],
    step='looped',
    prior=False,
    prior_dim=100,
    atol=1e-3,
    rtol=1e-3,
    lr=1e-3,
    plot_interval=1000,
)

logger = WandbLogger(
    entity="aproppe",
    save_dir='/Users/andrewproppe/Desktop/g2-pcfs_backup/wandb_garbage',
    project='g2-pcfs-nODE',
    log_model=False,
    save_code=False,
    offline=False,
)

trainer = pl.Trainer(
    max_epochs=1000,
    gpus=int(torch.cuda.is_available()),
    logger=logger,
    checkpoint_callback=False,
    enable_checkpointing=False,
)


trainer.fit(ode, datamodule=dm)

# testing
import numpy as np
from matplotlib import pyplot as plt

data = dm.train_set.__getitem__(0)
g2s  = data['input']
t_0 = g2s[:, :, 0:nchannels]
# t_0 = g2s[:, :, 0:1]
t_0 = torch.permute(t_0, (0, 2, 1))

timesteps = data.get("time")
min_steps = data['nsteps'][0]
max_steps = data['nsteps'][1]
num_steps = np.random.randint(min_steps, max_steps)
# num_steps = np.random.randint(5, 20)
nchannels = data['nchannels']
indices = torch.randperm(len(timesteps) - nchannels)[:num_steps].sort()[0]
ode_time = data.get("ode_time")

ode.eval()
with torch.no_grad():
    (time, output) = ode(t_0)

time = time.detach().numpy()
output = output.detach().numpy()

plt.plot(output[4, 0, 0, :])
    # pred_arr[i, :, :] = temp_pred[0]

# def hilbert_transform_torch(arr):
    # N : fft length
    # M : number of elements to zero out
    # U : DFT of u
    # v : IDFT of H(U)

# arr = dm.train_set.__getitem__(0)['prior']
# N = len(arr)
# # take forward Fourier transform
# U = torch.fft.fftshift(torch.fft.fft(arr))
# M = N - N//2 - 1
# # zero out negative frequency components
# # U[N//2+1:] = [0] * M
# U[N//2+1:] = 0
# # double fft energy except @ DC0
# U[1:N//2] = 2*U[1:N//2]
# # take inverse Fourier transform
# v = torch.fft.ifft(torch.fft.fftshift(U))
#     # return v
#
# arrh =scipy.signal.hilbert(arr)