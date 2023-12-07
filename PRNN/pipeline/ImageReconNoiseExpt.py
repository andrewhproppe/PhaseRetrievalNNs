import torch
import pickle
import seaborn as sns
import h5py
import os

from PRNN.models.base import PRUNe
from PRNN.pipeline.image_data import make_interferogram_frames
from PRNN.visualization.figure_utils import *
from PRNN.pipeline.transforms import input_transform_pipeline, truth_transform_pipeline
from tqdm import tqdm
from PRNN.pipeline.PhaseImages import norm_to_phase, optimize_global_phases
from scipy.stats import norm


class ImageReconNoiseExpt:
    def __init__(self, model, nbar, data_fname=None, idx=0, nsamples=100, npixels=64, nframes=32, optim_phase=False):

        self.model = model
        self.nbar = nbar
        self.data_fname = "../../data/raw/flowers_n5000_npix64.h5" if data_fname is None else data_fname
        self.idx = idx
        self.nsamples = nsamples
        self.npixels = npixels
        self.nframes = nframes
        self.optim_phase = optim_phase

        self.input_transforms = input_transform_pipeline()
        self.truth_transforms = truth_transform_pipeline()
        self.device = self.model.device

        self.get_from_h5()

    def get_from_h5(self, idx_stop=None):
        idx_start = self.idx
        idx_stop = idx_start + 1 if idx_stop is None else idx_stop
        with h5py.File(self.data_fname, "r") as f:
            self.y = torch.tensor(f["truths"][idx_start:idx_stop, :]).float().to(self.device)
            self.E1 = torch.tensor(f["E1"][:]).float().to(self.device)
            self.E2 = torch.tensor(f["E2"][:]).float().to(self.device)
            self.vis = torch.tensor(f["vis"][:]).float().to(self.device)


    def generate_predictions(self, nbar, print=True):
        yhat_list = []  # List of NN reconstructions
        N_list = []  # Total number of photons in each pixel across all frames

        for i in tqdm(range(0, self.nsamples), desc='Generating predictions..', disable=not print):
            with torch.no_grad():
                x = make_interferogram_frames(self.y, self.E1, self.E2, self.vis, nbar, 0, self.npixels, self.nframes, self.model.device)
                N = torch.sum(x, dim=0)
                x = self.input_transforms(x)
                yhat, _ = self.model(x.unsqueeze(0))
                yhat = norm_to_phase(yhat.squeeze(0))
                yhat_list.append(yhat)
                N_list.append(N)

        yhat = torch.stack(yhat_list, dim=0).squeeze(1)
        N = torch.stack(N_list, dim=0).squeeze(1)

        return yhat, N


    def optimize_phases(self, y, yhat):
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

        return torch.from_numpy(yhat).to(self.device)


    def evaluate_performance(self, y, yhat, N):
        # Calculate metrics such as standard deviation, mean MSE, etc.
        NN_μ = torch.mean(yhat, dim=0)
        NN_σ = torch.std(yhat, dim=0)
        NN_MSE = torch.mean((yhat - y.repeat(yhat.shape[0], 1, 1)) ** 2, dim=0)
        # NN_MSE = torch.mean((y - NN_μ)**2, dim=0)
        Nxy = torch.mean(N, dim=0)

        return NN_μ, NN_σ, NN_MSE, Nxy


    def run_experiment(self, overwrite=False, save=True, root='../../data/predictions/', keep_yhat=False):
        model_means = []
        model_stds = []
        model_mses = []
        model_Nxys = []
        model_yhats = []

        for signal_level in self.nbar:
            fname = f"{self.nsamples}samples_{signal_level}nbar_{self.idx}idx_{self.optim_phase}optim.pkl"
            file_path = os.path.join(root, fname)

            if not overwrite and os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    data_dict = pickle.load(f)
                    yhat = data_dict['yhat']
                    N = data_dict['N']
                    print(f"Loaded {fname}")
            else:
                yhat, N = self.generate_predictions(signal_level)
                if self.optim_phase:
                    yhat = self.optimize_phases(self.y, yhat)

                data_dict = {'yhat': yhat, 'N': N, 'optim_phase': self.optim_phase}
                if save:
                    with open(file_path, "wb") as f:
                        pickle.dump(data_dict, f)

            μ, σ, MSE, Nxy = self.evaluate_performance(self.y, yhat, N)
            model_means.append(μ)
            model_stds.append(σ)
            model_mses.append(MSE)
            model_Nxys.append(Nxy)
            if keep_yhat:
                model_yhats.append(yhat)

            self.model_mean = torch.stack(model_means, dim=0)
            self.model_std = torch.stack(model_stds, dim=0)
            self.model_mse = torch.stack(model_mses, dim=0)
            self.model_Nxy = torch.stack(model_Nxys, dim=0)
            if keep_yhat:
                self.yhats = torch.stack(model_yhats, dim=0)


    def calculate_std_and_fit(self, images, positions, Ns, colors=None, bins=50, fig=None):
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
            plt.hist(subset_values, bins=bins, density=True, alpha=0.6, label=f'N={N}', color=colors[i])

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

    def plot_model_metric_vs_nbar(self, pixel_positions=None, sum_over_image=True, use_std=False):
        if not hasattr(self, 'model_mse') or not hasattr(self, 'model_std') or not hasattr(self, 'model_Nxy'):
            print("Please run the experiment first to calculate model metrics.")
            return

        metric_values = self.model_mse if not use_std else self.model_std

        fig, ax = plt.subplots(figsize=(4, 4), dpi=150)

        if sum_over_image:
            metric_values_sum = torch.sum(metric_values, dim=(1, 2)).cpu().numpy()
            QNL_values = torch.sum(1/self.model_Nxy, dim=(1, 2)).cpu().numpy()
            SNL_values = torch.sum(1/torch.sqrt(self.model_Nxy), dim=(1, 2)).cpu().numpy()
        elif pixel_positions is not None:
            metric_values_sum = metric_values[:, pixel_positions[0], pixel_positions[1]].cpu().numpy()
            QNL_values = 1/self.model_Nxy[:, pixel_positions[0], pixel_positions[1]].cpu().numpy()
            SNL_values = 1/np.sqrt(self.model_Nxy[:, pixel_positions[0], pixel_positions[1]].cpu().numpy())
        else:
            print("Please specify pixel positions or set sum_over_image to True.")
            return

        ax.plot(self.nbar, metric_values_sum, marker='o', linestyle='-', color='b',
                label='MSE' if not use_std else 'Std')
        ax.plot(self.nbar, QNL_values, marker='s', linestyle='-', color='r', label='QNL')
        ax.plot(self.nbar, SNL_values, marker='s', linestyle='-', color='g', label='SNL')

        plt.xscale('log')
        plt.yscale('log')

        dress_fig(
            xlabel='Signal Level (nbar)',
            ylabel='Model Metric' if not use_std else 'Model Standard Deviation',
            title='Model Metric vs Signal Level',
            tight_layout=True,
            legend=True
        )

        # ax.set_xlabel('Signal Level (nbar)')
        # ax.set_ylabel('Model Metric' if not use_std else 'Model Standard Deviation', color='b')
        # ax.set_title('Model Metric vs Signal Level')
        # ax.legend(loc='upper left')
        # plt.show()