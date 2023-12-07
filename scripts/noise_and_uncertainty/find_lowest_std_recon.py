import torch
import pickle
from PRNN.models.base import PRUNe
from PRNN.pipeline.image_data import make_interferogram_frames
from PRNN.visualization.figure_utils import *
from PRNN.pipeline.transforms import input_transform_pipeline, truth_transform_pipeline
from data.utils import get_from_h5
from tqdm import tqdm
from PRNN.pipeline.PhaseImages import norm_to_phase, optimize_global_phases

"""
This script is used to find the lowest MSE and STD for a given model and dataset. Generates nsamples of the same true 
image y for each image in the dataset between idx_start and idx_stop, and then calculates the MSE and STD for each
reconstruction. The lowest MSE and STD are then printed out, along with the index of the image in the dataset that
"""

def generate_predictions(model, y, E1, E2, vis, nbar, npixels, nframes, nsamples, print=True):
    yhat_list = []  # List of NN reconstructions
    N_list = []  # Total number of photons in each pixel across all frames

    for i in tqdm(range(0, nsamples), desc='Generating predictions..', disable=not print):
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

def optimize_phases(y, yhat):
    """
    Helper function to do the global phase optimization for a list of reconstructions, instead of doing the
    optimization for each reconstruction during the loop in generate_predictions(). Does not benefit from GPU or
    CUDA so tensors are cast into numpy arrays, and then back into a tensor
    y: true image
    yhats: list of NN reconstructions for the same true image but with different random Poisson sampling
    """
    # Optimize global phases
    y = np.array(y.cpu())
    yhat = np.array(yhat.cpu())
    for i in tqdm(range(0, yhat.shape[0]), desc='Optimizing global phases..'):
        yhat[i], _, _ = optimize_global_phases(y, yhat[i])

    return torch.from_numpy(yhat)


input_transforms = input_transform_pipeline()
truth_transforms = truth_transform_pipeline()
mse_loss = torch.nn.MSELoss()

idx_start = 0
idx_stop = 1000
nsamples = 100
nbar = 1e3
nframes = 32
npixels = 64
optim = False

model = PRUNe.load_from_checkpoint(
    checkpoint_path="../../trained_models/bkgd_free/jolly-cloud-1.ckpt",
    # map_location=torch.device("cpu")
).eval()

ys, E1, E2, vis = get_from_h5("../data/raw/flowers_n5000_npix64.h5", model.device, idx_start, idx_stop)

mses = []
stds = []
for i, y in tqdm(enumerate(ys), desc=f'Calculating μ and σ for {nsamples} samples..', total=ys.shape[0]):
    # Get true image and probe fields
    yhat, N = generate_predictions(model, y, E1, E2, vis, nbar, npixels, nframes, nsamples, print=False)

    # Optimize global phases (optional and slow)
    # yhat = optimize_phases(y, yhat)

    # Compute and record statistics
    mse  = mse_loss(yhat, y.repeat(nsamples, 1, 1))
    std  = torch.mean(torch.std(yhat, dim=0))
    mses.append(mse)
    stds.append(std)

mses = np.array(torch.stack(mses, dim=0).cpu())
stds = np.array(torch.stack(stds, dim=0).cpu())
# Find lowest MSE and STD
min_mse_idx = np.argmin(mses)
min_std_idx = np.argmin(stds)
print(f'Lowest MSE: {mses[min_mse_idx]} at index {min_mse_idx}')
print(f'Lowest STD: {stds[min_std_idx]} at index {min_std_idx}')

# Save MSEs and STDs as a pickle
with open(f'../Stats_for_first_1000_flowers_nsample100_nbar1e3.pkl', 'wb') as f:
    pickle.dump((mses, stds), f)