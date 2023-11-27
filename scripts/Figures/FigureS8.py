import pickle
from PRNN.visualization.figure_utils import *

PIa = pickle.load(open("../../data/analysis/sim/PhaseImages_1000.0nsig_1000.0nback.pickle", "rb"))
PIb = pickle.load(open("../../data/analysis/sim/PhaseImages_1000.0nsig_100.0nback.pickle", "rb"))
PIc = pickle.load(open("../../data/analysis/sim/PhaseImages_1000.0nsig_10.0nback.pickle", "rb"))

# PIa = pickle.load(open("../../data/analysis/expt/PhaseImages_0.025ms_20230829_1000n.pickle", "rb"))
# PIb = pickle.load(open("../../data/analysis/expt/PhaseImages_0.05ms_20230829_1000n.pickle", "rb"))
# PIc = pickle.load(open("../../data/analysis/expt/PhaseImages_0.1ms_20230829_1000n.pickle", "rb"))

with PIc as PI:
    bad_idx_nn = np.where(PI.nn_mse == np.max(PI.nn_mse))[0][0]
    bad_idx_svd = np.where(PI.svd_mse == np.max(PI.svd_mse))[0][0]
    best_idx_nn = np.where(PI.nn_mse == np.min(PI.nn_mse))[0][0]
    best_idx_svd = np.where(PI.svd_mse == np.min(PI.svd_mse))[0][0]
    mse_diff = np.array(PI.svd_mse) - np.array(PI.nn_mse)
    best_diff_idx = np.where(mse_diff == np.max(mse_diff))[0][0]

# idx = 0
idx = best_idx_nn

with PIc as PI:
    figsize = (2, 2)
    dpi = 150
    xlabel = '$\it{x}$ (pix.)'
    ylabel = '$\it{y}$ (pix.)'

    set_font_size(10)
    cmap_data = 'gray'

    fig, axs = plt.subplots(nrows=6, ncols=6, figsize=(8, 8))

    def plot_image(ax, data, cmap, xlabel='', ylabel=''):
        data = data - data.min()
        data = data / data.max()
        im = ax.imshow(data, cmap=cmap)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return im

    for i in range(6):
        for j in range(6):
            im_data = plot_image(axs[i, j], PI.data[idx, i+j, :, :], cmap_data, ylabel=ylabel, xlabel=xlabel)

    for ax in axs[:, 1:]:
        for a in ax:
            a.set_yticks([])
            a.set_ylabel('')

    for ax in axs[:-1, :]:
        for a in ax:
            a.set_xticks([])
            a.set_xlabel('')

    plt.tight_layout()
    plt.show()
