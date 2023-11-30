import torch
import pickle
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
nsamples = 100
nbar = 1e3
nframes = 32
npixels = 64

model = PRUNe.load_from_checkpoint(
    checkpoint_path="../trained_models/bkgd_free/jolly-cloud-1.ckpt",
    map_location=torch.device("cpu")
).eval()

# Get true image and probe fields
y, E1, E2, vis = get_from_h5("../data/raw/flowers_n5000_npix64.h5", 2, model.device)

# # Generate or load predictions
# yhat, N = generate_predictions(model, y, E1, E2, vis, nbar, npixels, nframes, nsamples)
# with open("100samples_idx2.pkl", "wb") as f:
#     pickle.dump((yhat, N), f)

with open("100samples_idx2.pkl", "rb") as f:
    yhat, N = pickle.load(f)

# Compute statistics
Nxy = torch.mean(N, dim=0)

NN_μ = torch.mean(yhat, dim=0)
NN_σ = torch.std(yhat, dim=0)

SNL = 1/torch.sqrt(Nxy)
QNL = 1/Nxy

def plot_std_vs_n(images, n_values, positions):
    """
    Plot standard deviation vs number of images for specified pixel positions.

    Parameters:
    - images: List of 2D numpy arrays (images).
    - positions: List of tuples (x, y) specifying pixel positions.

    Returns:
    - None (displays a plot).
    """

    fig, axs = plt.subplots(2, len(positions), figsize=(15, 8))

    for i, pos in enumerate(positions):
        std_at_pos = []
        all_pixel_values = []

        for n in n_values:
            # Take the first 'n' images
            subset_images = images[:n]

            # Extract values at the specified position from each image
            pixel_values = [img[pos[0], pos[1]] for img in subset_images]
            all_pixel_values.append(pixel_values)

            # Calculate and store standard deviation
            std_at_pos.append(np.std(pixel_values))

        # Plotting in the first subplot
        axs[0, i].plot(n_values, std_at_pos, label=f"Pixel {pos}")
        axs[0, i].set_xlabel("Number of Images")
        axs[0, i].set_ylabel("Standard Deviation")
        axs[0, i].set_title(f"Pixel-wise Std Dev vs Num Images (Pixel {pos})")
        axs[0, i].legend()

        # Plotting the histogram and overlaying the pixel values in the second subplot
        axs[1, i].hist(all_pixel_values, bins=25, edgecolor='black', label=[f"n={n}" for n in n_values])
        axs[1, i].set_xlabel("Pixel Values")
        axs[1, i].set_ylabel("Frequency")
        axs[1, i].set_title(f"Histogram of Pixel Values (Pixel {pos})")
        axs[1, i].legend()

    plt.tight_layout()
    plt.show()

# plot_std_vs_n(yhat, [50, 80, 100], [(30, 30), (50, 50)])
# Example usage:
# Assuming you have a list 'images' containing your 64x64 images
# Assuming you want to check the standard deviation at positions (30, 30) and (50, 50)
# plot_std_vs_n(images, [(30, 30), (50, 50)])
