import os
from data.utils import rgb_to_phase, crop_and_resize
from QIML.utils import get_system_and_backend
from matplotlib.image import imread
from matplotlib import pyplot as plt
from QIML.visualization.figure_utils import set_font_size, dress_fig

get_system_and_backend()

nx      = 64 # X pixels
ny      = nx # Y pixels

masks_folder = 'flowers'

filenames = os.listdir(os.path.join('../../data/masks', masks_folder))

filenames.sort()

mask = filenames[0]

filename = os.path.join('../../data/masks', masks_folder, mask)
rgb_image = imread(filename)

set_font_size(12)
plt.figure()
plt.imshow(rgb_image)

phase_mask_gray = rgb_to_phase(filename, color_balance=[0.6, 0.2, 0.2])
plt.figure()
plt.imshow(phase_mask_gray/phase_mask_gray.max())
plt.colorbar()

phase_mask_resized = crop_and_resize(phase_mask_gray, nx, ny)
plt.figure()
plt.imshow(phase_mask_resized/phase_mask_resized.max())
# plt.colorbar()
