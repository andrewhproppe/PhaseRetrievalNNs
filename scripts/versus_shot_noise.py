import torch
from PRNN.models.base import PRUNe
from PRNN.pipeline.image_data import make_interferogram_frames
from PRNN.utils import get_system_and_backend
from PRNN.visualization.figure_utils import *
from PRNN.pipeline.transforms import input_transform_pipeline
from data.utils import get_from_h5

get_system_and_backend()

model = PRUNe.load_from_checkpoint(
    checkpoint_path="../trained_models/bkgd_free/jolly-cloud-1.ckpt",
    # map_location=torch.device("gpu")
).eval()

transforms = input_transform_pipeline()
mse = torch.nn.MSELoss()
nbar = 1e3

y, E1, E2, vis = get_from_h5("../data/raw/flowers_n5000_npix64.h5", 2, model.device)
y = (y - y.min()) / (y.max() - y.min())

# plt.imshow(x[0, :, :].cpu().numpy(), cmap="gray")
mse_list = []
yhats = []
for i in range(0, 5):
    with torch.no_grad():
        x = make_interferogram_frames(y, E1, E2, vis, nbar, 0, 64, 32, model.device)
        x = transforms(x)
        yhat, _ = model(x.unsqueeze(0))
        yhats.append(yhat)


def shot_noise_limit_uncertainty(total_photon_counts, image_size):
    # Calculate mean photon count per pixel
    mean_photon_count = total_photon_counts / (image_size[0] * image_size[1])
    # Calculate shot noise limit (standard deviation)
    shot_noise_limit = torch.sqrt(torch.tensor(mean_photon_count))
    return shot_noise_limit


        # mse_list.append(mse(yhat.squeeze(0), y))

# plt.figure()
# plt.imshow(yhat[0, :, :].cpu(), cmap="twilight_shifted")
#
# plt.figure()
# plt.imshow(y.cpu(), cmap="twilight_shifted")
