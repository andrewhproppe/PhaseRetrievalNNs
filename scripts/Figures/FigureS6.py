import pickle
from PRNN.visualization.figure_utils import *

PIa = pickle.load(open("../../data/analysis/expt/PhaseImages_0.025ms_20230829_1000n.pickle", "rb"))
PIb = pickle.load(open("../../data/analysis/expt/PhaseImages_0.05ms_20230829_1000n.pickle", "rb"))
PIc = pickle.load(open("../../data/analysis/expt/PhaseImages_0.1ms_20230829_1000n.pickle", "rb"))

idx = 3

figsize = (2, 2)
dpi = 150
log_histos = True
histo_xlims = (-0.01, 0.33)
xlabel = '$\it{x}$ (pix.)'
ylabel = '$\it{y}$ (pix.)'
tight = False
save_figs = False
rootpath = '/Users/andrewproppe/JCEP/Manuscripts/PhaseRetrievalML_wGuillaume/Figures/Figure 2'

set_font_size(10)
# cmap = 'hsv'
# cmap = husl_palette(as_cmap=True)
# cmap = 'twilight'
cmap = 'twilight_shifted'
cmap_data = 'gray'


def myfmt(x, pos):
    return '{0:.1f}'.format(x)

# fig_x = plt.figure(figsize=(4, 4), dpi=dpi)
# # fig_x = make_fig(figsize=figsize, dpi=dpi)
# y_true = PIa.y_true[idx, :, :]
# y_true = y_true - y_true.min()
# y_true = y_true / y_true.max()
# plt.imshow(norm_to_phase(y_true) - torch.pi, cmap=cmap)
# plt.colorbar(format=ticker.FuncFormatter(myfmt))
# dress_fig(tight=tight, xlabel=xlabel, ylabel=ylabel)


# Function to plot an image and customize the figure
def plot_image(ax, data, cmap, xlabel='', ylabel=''):
    data = data - data.min()
    data = data / data.max()
    im = ax.imshow(data, cmap=cmap)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return im

# fig0 = make_fig(figsize=figsize, dpi=dpi)
# plt.imshow(PIa.y_true[idx, :, :], cmap=cmap)
# plt.title('True image')
# dress_fig(tight=tight, xlabel=xlabel, ylabel=ylabel)
# raise RuntimeError

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(3 * figsize[0], 3 * figsize[1]), dpi=dpi)

# Loop through the instances and populate the subplots
for i, PI in enumerate([PIa, PIb, PIc]):
    data = PI.data[idx, 0, :, :]
    y_nn = PI.y_nn[idx, :, :]
    y_svd = PI.y_svd[idx, :, :]

    # Determine row and column indices
    row = i

    # Plot data with y-ticks and y-labels
    im_data = plot_image(axs[row, 0], data, cmap_data, ylabel=ylabel, xlabel=xlabel)

    # Plot y_nn
    im_y_nn = plot_image(axs[row, 1], y_nn, cmap, xlabel=xlabel)

    # Plot y_svd with x-ticks and x-labels
    im_y_svd = plot_image(axs[row, 2], y_svd, cmap, xlabel=xlabel)

# Remove ticks and labels for specific axes
for ax in axs[:, 1:]:
    for a in ax:
        a.set_yticks([])

for ax in axs[:-1, :]:
    for a in ax:
        a.set_xticks([])
        a.set_xlabel('')

axs[0, 0].set_title('Data')
axs[0, 1].set_title('PRUNe')
axs[0, 2].set_title('SVD')

plt.tight_layout()