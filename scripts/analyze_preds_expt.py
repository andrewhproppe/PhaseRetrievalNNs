import pickle

PI = pickle.load(open("../data/analysis/expt/PhaseImages_0.025ms_20230829.pickle", "rb"))
PI.plot_phase_images(6)

# Grab a few images from the dataset and save to file (for Figure 1)
savepath = '/Users/andrewproppe/JCEP/Manuscripts/PhaseRetrievalML_wGuillaume/Figures/Figure 1/'
fig_images = PI.data[0, 0:10, :, :]
from matplotlib import pyplot as plt
fig, ax = plt.subplots()
plt.xticks([])
plt.yticks([])

output_image = PI.y_nn[0, :, :]
plt.imshow(output_image, cmap='twilight_shifted')
plt.savefig(savepath+'nn_output.pdf')

# for i, img in enumerate(fig_images):
#     plt.imshow(img/img.max(), cmap='twilight_shifted')
#     plt.savefig(savepath+f'fig1a_{i}.pdf')

