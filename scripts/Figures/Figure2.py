import pickle
import torch
import matplotlib.ticker as ticker
from PRNN.visualization.figure_utils import *
from PRNN.pipeline.PhaseImages import norm_to_phase

PIa = pickle.load(open("../../data/analysis/expt/PhaseImages_0.025ms_20230829_1000n.pickle", "rb"))
PIb = pickle.load(open("../../data/analysis/expt/PhaseImages_0.05ms_20230829_1000n.pickle", "rb"))
PIc = pickle.load(open("../../data/analysis/expt/PhaseImages_0.1ms_20230829_1000n.pickle", "rb"))

# Worst index for NN for noisy data:
with PIc as PI:
    bad_idx_nn = np.where(PI.nn_mse == np.max(PI.nn_mse))[0][0]
    bad_idx_svd = np.where(PI.svd_mse == np.max(PI.svd_mse))[0][0]
    best_idx_nn = np.where(PI.nn_mse == np.min(PI.nn_mse))[0][0]
    best_idx_svd = np.where(PI.svd_mse == np.min(PI.svd_mse))[0][0]

# idx = 0
idx = best_idx_nn

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


fig0 = make_fig(figsize=figsize, dpi=dpi)
plt.imshow(PIa.y_true[idx, :, :], cmap=cmap)
dress_fig(tight=tight, xlabel=xlabel, ylabel=ylabel)
if save_figs:
    plt.savefig(rootpath+'/fig2a.pdf')

def myfmt(x, pos):
    return '{0:.1f}'.format(x)

fig_x = plt.figure(figsize=(4, 4), dpi=dpi)
# fig_x = make_fig(figsize=figsize, dpi=dpi)
y_true = PIa.y_true[idx, :, :]
y_true = y_true - y_true.min()
y_true = y_true / y_true.max()
plt.imshow(norm_to_phase(y_true) - torch.pi, cmap=cmap)
plt.colorbar(format=ticker.FuncFormatter(myfmt))
dress_fig(tight=tight, xlabel=xlabel, ylabel=ylabel)
if save_figs:
    plt.savefig(rootpath+'/fig2_colorbar_fig.pdf')

with PIa as PI:
    fig1 = make_fig(figsize=figsize, dpi=dpi)
    fig1 = plt.figure(figsize=figsize, dpi=dpi)
    data = PI.data[idx, 0, :, :]
    data = data - data.min()
    data = data / data.max()
    plt.imshow(data, cmap=cmap_data)
    # plt.imshow(PI.data[idx, 0, :, :], cmap=cmap_data)

    # raise RuntimeError

    plt.xticks([])
    dress_fig(tight=tight, ylabel=ylabel)
    if save_figs:
        plt.savefig(rootpath+'/fig2b.svg')

    fig2 = make_fig(figsize=figsize, dpi=dpi)
    plt.imshow(PI.y_nn[idx, :, :], cmap=cmap)
    dress_fig(tight=tight)
    plt.xticks([])
    plt.yticks([])
    if save_figs:
        plt.savefig(rootpath+'/fig2c.pdf')

    fig3 = make_fig(figsize=figsize, dpi=dpi)
    plt.imshow(PI.y_svd[idx, :, :], cmap=cmap)
    dress_fig(tight=tight)
    plt.xticks([])
    plt.yticks([])
    if save_figs:
        plt.savefig(rootpath+'/fig2d.pdf')

    PI.error_histograms() #x_min=histo_xlims[0], x_max=histo_xlims[1])

    fig4, ax4 = make_fig(figsize=(1.5*figsize[0], figsize[1]), dpi=dpi)
    plt.bar(PI.nn_mse_histo[1][:-1], (PI.nn_mse_histo[0]), width=PI.nn_mse_histo[1][1] - PI.nn_mse_histo[1][0], linewidth=0., edgecolor='black', bottom=0.8, alpha=0.9)
    plt.bar(PI.svd_mse_histo[1][:-1], (PI.svd_mse_histo[0]), width=PI.svd_mse_histo[1][1] - PI.svd_mse_histo[1][0], linewidth=0., edgecolor='red', bottom=0.8, alpha=0.7)
    # plt.xticks([])

    μ_nn_mse = np.mean(PI.nn_mse)
    μ_svd_mse = np.mean(PI.svd_mse)
    σ_nn_mse = np.var(PI.nn_mse)
    σ_svd_mse = np.var(PI.svd_mse)

    μ_nn_mse = np.mean(PI.nn_mse)
    μ_svd_mse = np.mean(PI.svd_mse)
    σ_nn_mse = np.var(PI.nn_mse)
    σ_svd_mse = np.var(PI.svd_mse)
    print(f'μ_nn_mse = {μ_nn_mse}')
    print(f'σ_nn_mse = {σ_nn_mse}')
    print(f'μ_svd_mse = {μ_svd_mse}')
    print(f'σ_svd_mse = {σ_svd_mse}')
    raise RuntimeError

    if log_histos:
        plt.yscale('log')
    dress_fig(tight=tight, ylabel='Counts', xlim=histo_xlims)
    if save_figs:
        plt.savefig(rootpath+'/fig2e.pdf')

with PIb as PI:
    fig5 = make_fig(figsize=figsize, dpi=dpi)
    data = PI.data[idx, 0, :, :]
    data = data - data.min()
    data = data / data.max()
    plt.imshow(data, cmap=cmap_data)
    # plt.imshow(PI.data[idx, 0, :, :], cmap=cmap_data)
    plt.xticks([])
    dress_fig(tight=tight, ylabel=ylabel)
    if save_figs:
        plt.savefig(rootpath+'/fig2f.svg')

    fig6 = make_fig(figsize=figsize, dpi=dpi)
    plt.imshow(PI.y_nn[idx, :, :], cmap=cmap)
    dress_fig(tight=tight)
    plt.xticks([])
    plt.yticks([])
    if save_figs:
        plt.savefig(rootpath+'/fig2g.pdf')

    fig7 = make_fig(figsize=figsize, dpi=dpi)
    plt.imshow(PI.y_svd[idx, :, :], cmap=cmap)
    dress_fig(tight=tight)
    plt.xticks([])
    plt.yticks([])
    if save_figs:
        plt.savefig(rootpath+'/fig2h.pdf')

    PI.error_histograms()

    fig8, ax8 = make_fig(figsize=(1.5*figsize[0], figsize[1]), dpi=dpi)
    plt.bar(PI.nn_mse_histo[1][:-1], PI.nn_mse_histo[0], width=PI.nn_mse_histo[1][1] - PI.nn_mse_histo[1][0], linewidth=0., edgecolor='black', bottom=0.8, alpha=0.9)
    plt.bar(PI.svd_mse_histo[1][:-1], PI.svd_mse_histo[0], width=PI.svd_mse_histo[1][1] - PI.svd_mse_histo[1][0], linewidth=0., edgecolor='red', bottom=0.8, alpha=0.7)
    # plt.xticks([])
    if log_histos:
        plt.yscale('log')
    dress_fig(tight=tight, ylabel='Counts', xlim=histo_xlims)
    if save_figs:
        plt.savefig(rootpath+'/fig2i.pdf')


with PIc as PI:
    fig9 = make_fig(figsize=figsize, dpi=dpi)
    data = PI.data[idx, 0, :, :]
    data = data - data.min()
    data = data / data.max()
    plt.imshow(data, cmap=cmap_data)
    # plt.imshow(PI.data[idx, 0, :, :], cmap=cmap_data)
    dress_fig(tight=tight, xlabel=xlabel, ylabel=ylabel)
    if save_figs:
        plt.savefig(rootpath+'/fig2j.svg')

    fig10 = make_fig(figsize=figsize, dpi=dpi)
    plt.imshow(PI.y_nn[idx, :, :], cmap=cmap)
    dress_fig(tight=tight, xlabel=xlabel)
    # plt.xticks([])
    plt.yticks([])
    if save_figs:
        plt.savefig(rootpath+'/fig2k.pdf')

    fig11 = make_fig(figsize=figsize, dpi=dpi)
    plt.imshow(PI.y_svd[idx, :, :], cmap=cmap)
    dress_fig(tight=tight, xlabel=xlabel)
    # plt.xticks([])
    plt.yticks([])
    if save_figs:
        plt.savefig(rootpath+'/fig2l.pdf')

    PI.error_histograms()

    fig12, ax12 = make_fig(figsize=(1.5*figsize[0], figsize[1]), dpi=dpi)
    plt.bar(PI.nn_mse_histo[1][:-1], PI.nn_mse_histo[0], width=PI.nn_mse_histo[1][1] - PI.nn_mse_histo[1][0], linewidth=0., edgecolor='black', bottom=0.8, alpha=0.9)
    plt.bar(PI.svd_mse_histo[1][:-1], PI.svd_mse_histo[0], width=PI.svd_mse_histo[1][1] - PI.svd_mse_histo[1][0], linewidth=0., edgecolor='red', bottom=0.8, alpha=0.7)
    if log_histos:
        plt.yscale('log')
    dress_fig(tight=tight, xlabel='MSE', ylabel='Counts', xlim=histo_xlims)
    if save_figs:
        plt.savefig(rootpath+'/fig2m.pdf')


# Plotting worst images at highest SNR
# Neural net
fig13, ax13 = make_fig(figsize=figsize, dpi=dpi)
plt.imshow(PIc.y_nn[bad_idx_nn, :, :], cmap=cmap)
dress_fig(tight=tight, xlabel=xlabel, ylabel=ylabel)
if save_figs:
    plt.savefig(rootpath+'/fig2n.pdf')

fig14, ax14 = make_fig(figsize=figsize, dpi=dpi)
plt.imshow(PIc.y_true[bad_idx_nn, :, :], cmap=cmap)
dress_fig(tight=tight, xlabel=xlabel, ylabel=ylabel)
if save_figs:
    plt.savefig(rootpath+'/fig2o.pdf')

# SVD
fig15, ax15 = make_fig(figsize=figsize, dpi=dpi)
plt.imshow(PIc.y_svd[bad_idx_svd, :, :], cmap=cmap)
dress_fig(tight=tight, xlabel=xlabel, ylabel=ylabel)
if save_figs:
    plt.savefig(rootpath+'/fig2p.pdf')

fig16, ax16 = make_fig(figsize=figsize, dpi=dpi)
plt.imshow(PIc.y_true[bad_idx_svd, :, :], cmap=cmap)
dress_fig(tight=tight, xlabel=xlabel, ylabel=ylabel)
if save_figs:
    plt.savefig(rootpath+'/fig2q.pdf')


# plt.colorbar()

# w = 1.5
# h = w
# nrow = 3
# ncol = 4
# histo_w = 2
# fig = plt.figure(figsize=(3*w+histo_w, 3*h), dpi=150)
# ax1 = plt.subplot(341)
# ax2 = plt.subplot(342)
# ax3 = plt.subplot(343)
# ax4 = plt.subplot(344)
# fig, ax = plt.subplots(nrow, ncol, figsize=(3*w+histo_w, 3*h), dpi=150, width_ratios=[1, 1, 1, histo_w/w])

# PI.plot_phase_images(6)


# # Histogram of errors
# nbins = 200
# min_error = min(min(PI.nn_mse, PI.svd_mse))
# max_error = max(max(PI.nn_mse, PI.svd_mse))
# bins = torch.linspace(min_error, max_error, nbins)
# nn_mse_histo = torch.histogram(torch.tensor(PI.nn_mse), bins=bins)
# svd_mse_histo = torch.histogram(torch.tensor(PI.svd_mse), bins=bins)
# fig, ax = plt.subplots(1, 2, figsize=(6, 2), dpi=150)
# ax[0].bar(nn_mse_histo[1][:-1], nn_mse_histo[0], width=nn_mse_histo[1][1] - nn_mse_histo[1][0], linewidth=0.1, edgecolor='black')
# ax[0].set_title("SRN3D")
# ax[0].set_yscale('log')
# ax[1].bar(svd_mse_histo[1][:-1], svd_mse_histo[0], width=svd_mse_histo[1][1] - svd_mse_histo[1][0], linewidth=0.1, edgecolor='black')
# ax[1].set_yscale('log')
# ax[1].set_title("SVD")
# dress_fig(tight=True, xlabel='MSE', ylabel='Counts')