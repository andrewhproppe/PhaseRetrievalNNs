import torch
import pickle
import seaborn as sns

from PRNN.models.base import PRUNe
from PRNN.pipeline.image_data import make_interferogram_frames
from PRNN.visualization.figure_utils import *
from PRNN.pipeline.transforms import input_transform_pipeline, truth_transform_pipeline
from data.utils import get_from_h5
from tqdm import tqdm
from PRNN.pipeline.PhaseImages import norm_to_phase, optimize_global_phases
from scipy.stats import norm


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


def plot_std_vs_n(figsize, images, n_values, positions, bins=50):
    """
    Plot standard deviation vs number of images for specified pixel positions.

    Parameters:
    - images: List of 2D numpy arrays (images).
    - positions: List of tuples (x, y) specifying pixel positions.

    Returns:
    - None (displays a plot).
    """

    fig, axs = plt.subplots(2, len(positions), figsize=figsize, dpi=150)

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
        axs[0, i].plot(n_values, std_at_pos, '-s', label=f"Pixel {pos}")
        axs[0, i].set_xlabel("Number of Images")
        axs[0, i].set_ylabel("Standard Deviation")
        axs[0, i].set_title(f"Pixel-wise Std Dev vs Num Images (Pixel {pos})")
        axs[0, i].legend()

        # Plotting the histogram and overlaying the pixel values in the second subplot
        pixel_histos = []
        for j, pixel_vals in enumerate(all_pixel_values):
            pixel_histo = torch.histogram(torch.tensor(pixel_vals), bins=bins) #, range=(0, 2*np.pi))
            pixel_histos.append(pixel_histo)
            axs[1, i].bar(pixel_histo[1][:-1], pixel_histo[0]/pixel_histo[0].max(), width=pixel_histo[1][1] - pixel_histo[1][0], linewidth=0., edgecolor='black', alpha=0.7, label=f"n={n_values[j]}")
        # axs[1, i].hist(all_pixel_values, bins=bins, edgecolor='black', alpha=0.5, density=True, label=[f"n={n}" for n in n_values])
        # plt.bar(PI.nn_mse_histo[1][:-1], (PI.nn_mse_histo[0]), width=PI.nn_mse_histo[1][1] - PI.nn_mse_histo[1][0], linewidth=0., edgecolor='black', bottom=0.8, alpha=0.9)

        axs[1, i].set_xlabel("Pixel Values")
        axs[1, i].set_ylabel("Frequency")
        axs[1, i].set_title(f"Histogram of Pixel Values (Pixel {pos})")
        axs[1, i].legend()

    plt.tight_layout()


def calculate_std_and_fit(fig, images, positions, Ns, colors=None):
    # Set Seaborn color palette
    colors = np.array(sns.color_palette("icefire", len(Ns))) if colors is None else colors

    # Flatten the images and extract values at specified positions
    values = images[:, positions[0], positions[1]].flatten().numpy()

    for i, N in enumerate(Ns):
        # Take a subset of the images
        subset_values = values[:N]

        # Calculate standard deviation
        std_dev = np.std(subset_values)

        # Plot histogram with Seaborn color palette
        plt.hist(subset_values, bins=50, density=True, alpha=0.6, label=f'N={N}', color=colors[i])

        # Fit the histogram with a Gaussian distribution
        mu, sigma = norm.fit(subset_values)

        # Plot the fitted Gaussian with a darker shade
        x = np.linspace(min(subset_values), max(subset_values), 100)
        p = norm.pdf(x, mu, sigma)
        plt.plot(x, p, color=colors[i]*0.5, linewidth=2)
        dress_fig(xlabel="Pixel Values", ylabel="Frequency")

        # Display statistics
        plt.legend()

        print(f"Standard Deviation (N={N}) at pixel {positions}: {std_dev}")


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

idx = 490
nsamples = 100
nbar = 1e3
nframes = 32
npixels = 64
optim = False
load = False
save = False

fname = f"{nsamples}samples_{nbar}nbar_{idx}idx_{optim}optim.pkl"

model = PRUNe.load_from_checkpoint(
    checkpoint_path="../../trained_models/bkgd_free/feasible-tree-4.ckpt",
    # checkpoint_path="../trained_models/bkgd_free/jolly-cloud-1.ckpt",
    # map_location=torch.device("cpu")
).eval()

# Get true image and probe fields
y, E1, E2, vis = get_from_h5("../data/raw/flowers_n5000_npix64.h5", model.device, idx)

# Generate or load predictions
if load:
    with open(fname, "rb") as f:
        yhat, N = pickle.load(f)
else:
    yhat, N = generate_predictions(model, y, E1, E2, vis, nbar, npixels, nframes, nsamples)

    if optim:
        yhat = optimize_phases(y, yhat)

    if save:
        with open(fname, "wb") as f:
            pickle.dump((yhat, N), f)

# Compute statistics
Nxy = torch.mean(N, dim=0)

NN_μ = torch.mean(yhat, dim=0)
NN_σ = torch.std(yhat, dim=0)
NN_mse = torch.mean((yhat - y.repeat(nsamples, 1, 1)) ** 2, dim=0)
NN_mse2 = (NN_μ - y) ** 2

SNL = 1/torch.sqrt(Nxy)
QNL = 1/Nxy

colors = np.array(sns.color_palette("viridis", 4))  # You can use any Seaborn color palette
set_font_size(8)
fig = plt.figure(figsize=(10, 5), dpi=150)
calculate_std_and_fit(fig, yhat.cpu(), (33, 33), [500, 750, 1000], colors)

# # plot_std_vs_n(yhat.cpu(), [1000, 2000, 4000, 8000, 10000], [(32, 32), (1, 1)], 50)

fig, ax = plt.subplots(1, 3, figsize=(9, 3), dpi=150)
im1 = ax[0].imshow(E1[0].cpu(), label="E")
ax[0].set_title("E1")

# plot QNL
im2 = ax[1].imshow(QNL.cpu(), label="QNL")
ax[1].text(32, 4, f"Sum: {torch.sum(QNL.cpu()):.2f}", horizontalalignment='center', color='w')
ax[1].set_title("QNL")

# plot NN_σ
# im3 = ax[2].imshow(NN_σ.cpu(), label="NN σ")
# ax[2].text(32, 4, f"Sum: {torch.sum(NN_σ.cpu()):.2f}", horizontalalignment='center', color='w')
im3 = ax[2].imshow(NN_mse.cpu(), label="NN MSE")
ax[2].text(32, 4, f"Sum: {torch.sum(NN_mse.cpu()):.2f}", horizontalalignment='center', color='w')
# ax[2].set_title("NN σ")
ax[2].set_title("NN MSE")

fig.colorbar(im1)
fig.colorbar(im2)
fig.colorbar(im3)
plt.tight_layout()