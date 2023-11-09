import pickle

from PhaseImages import PhaseImages
from QIML.visualization.AP_figs_funcs import set_font_size
from matplotlib import pyplot as plt

save_figs = True
set_font_size(14)

PI = pickle.load(open("../data/analysis/expt/PhaseImages_0.1ms_20230829.pickle", "rb"))

# Grab a few images from the dataset and save to file (for Figure 1)
savepath = '/Users/andrewproppe/JCEP/Manuscripts/PhaseRetrievalML_wGuillaume/Figures/Figure 1/'
fig_images = PI.data[0, 0:10, :, :]


output_image = PI.y_nn[0, :, :]
plt.imshow(output_image, cmap='twilight_shifted')
plt.xticks([])
plt.yticks([])
if save_figs:
    plt.savefig(savepath+'nn_output.pdf')

cmap_data = 'gray'
for i, img in enumerate(fig_images):
    img = img - img.min()
    img = img / img.max()
    # img = img * 3.1459 * 2 - 3.1459
    plt.imshow(img, cmap=cmap_data)
    plt.colorbar() if i == 0 else None
    plt.xticks([])
    plt.yticks([])
    if save_figs:
        plt.savefig(savepath+f'fig1a_grey{i}.pdf')

