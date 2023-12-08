import pickle

from PhaseImages import PhaseImages
from PRNN.visualization.figure_utils import set_font_size
from matplotlib import pyplot as plt

save_figs = True
set_font_size(14)

import sys
import PRNN
sys.modules['QIML'] = PRNN # Directory was previously called QIML, now PRNN
PI = pickle.load(open("../../data/analysis/expt/PhaseImages_0.1ms_20230829_1000n.pickle", "rb"))

# Grab a few images from the dataset and save to file (for Figure 1)
savepath = '/Users/andrewproppe/JCEP/Manuscripts/PhaseRetrievalML_wGuillaume/Figures/Figure 1/'

# img_idx = 101
# img_idx = 157
img_idx = 344
true_image = PI.y_true[img_idx, :, :]
plt.imshow(true_image, cmap='twilight_shifted')
plt.xticks([])
plt.yticks([])
if save_figs:
    plt.savefig(savepath+'true_image.pdf')

output_image = PI.y_nn[img_idx, :, :]
plt.imshow(output_image, cmap='twilight_shifted')
plt.xticks([])
plt.yticks([])
if save_figs:
    plt.savefig(savepath+'nn_output.pdf')

cmap_data = 'gray'
fig_images = PI.data[img_idx, 0:10, :, :]
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

