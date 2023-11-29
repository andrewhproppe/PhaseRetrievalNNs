import torch

from PRNN.models.base import PRUNe
from PRNN.pipeline.image_data import make_interferogram_frames
from PRNN.utils import get_system_and_backend
from PRNN.visualization.figure_utils import *
from PRNN.pipeline.transforms import input_transform_pipeline
from data.utils import get_from_h5
from tqdm import tqdm

get_system_and_backend()

model = PRUNe.load_from_checkpoint(
    checkpoint_path="../trained_models/bkgd_free/jolly-cloud-1.ckpt",
    map_location=torch.device("cpu")
).eval()

transforms = input_transform_pipeline()
mse = torch.nn.MSELoss()
nsamples = 10
nbar = 1e3
nframes = 32
npixels = 64

y, E1, E2, vis = get_from_h5("../data/raw/flowers_n5000_npix64.h5", 2, model.device)

# Normalize y
# y = (y - y.min()) / (y.max() - y.min())
y = y / y.max()

# Generate predictions
mse_list = []
yhat_list = []
for i in tqdm(range(0, nsamples)):
    with torch.no_grad():
        x = make_interferogram_frames(y, E1, E2, vis, nbar, 0, npixels, nframes, model.device)
        x = transforms(x)
        yhat, _ = model(x.unsqueeze(0))
        yhat_list.append(yhat.squeeze(0))

# Scale NN outputs to have nbar photons each
for i in range(len(yhat_list)):
    yhat_list[i] = yhat_list[i] / yhat_list[i].sum() * nbar

yhat = torch.stack(yhat_list, dim=0).squeeze(1)
NN_μ = torch.mean(yhat, dim=0)
NN_σ = torch.std(yhat, dim=0)

# Multiply y by total number of photons in all frames (nframes * nbar)
scaled_y = y / y.sum() * nframes * nbar
SNL = 1/torch.sqrt(scaled_y)
QNL = 1/scaled_y
# SNL_σ = y / y.sum() * nframes * nbar