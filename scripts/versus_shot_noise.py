import torch
import PRNN.utils
from PRNN.models.base import PRUNe
from PRNN.pipeline.image_data import make_interferogram_frames
from PRNN.visualization.figure_utils import *
from PRNN.pipeline.transforms import input_transform_pipeline, truth_transform_pipeline
from data.utils import get_from_h5
from tqdm import tqdm
from PRNN.pipeline.PhaseImages import norm_to_phase

def generate_predictions(model, y, E1, E2, vis, nbar, npixels, nframes, nsamples):
    yhat_list = []  # List of NN reconstructions
    N_list = []  # Total number of photons in each pixel across all frames

    for i in tqdm(range(0, nsamples)):
        with torch.no_grad():
            x = make_interferogram_frames(y, E1, E2, vis, nbar, 0, npixels, nframes, model.device)
            N = torch.sum(x, dim=0)
            x = input_transforms(x)
            yhat, _ = model(x.unsqueeze(0))
            yhat = norm_to_phase(yhat.squeeze(0))
            yhat_list.append(yhat)
            N_list.append(N)

    yhat = torch.stack(yhat_list, dim=0).squeeze(1)
    N = torch.stack(N_list, dim=0).squeeze(1)

    return yhat, N

input_transforms = input_transform_pipeline()
truth_transforms = truth_transform_pipeline()
mse = torch.nn.MSELoss()
nsamples = 10
nbar = 1e3
nframes = 32
npixels = 64

model = PRUNe.load_from_checkpoint(
    checkpoint_path="../trained_models/bkgd_free/jolly-cloud-1.ckpt",
    map_location=torch.device("cpu")
).eval()

# Get true image and probe fields
y, E1, E2, vis = get_from_h5("../data/raw/flowers_n5000_npix64.h5", 2, model.device)

# # Generate predictions
yhat, N = generate_predictions(model, y, E1, E2, vis, nbar, npixels, nframes, nsamples)

# Compute statistics
Nxy = torch.mean(N, dim=0)

NN_μ = torch.mean(yhat, dim=0)
NN_σ = torch.std(yhat, dim=0)

SNL = 1/torch.sqrt(Nxy)
QNL = 1/Nxy