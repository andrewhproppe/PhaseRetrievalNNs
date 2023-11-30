import torch

import PRNN.pipeline.PhaseImages
from PRNN.models.base import PRUNe
from PRNN.utils import get_system_and_backend
from PRNN.pipeline.PhaseImages import PhaseImages
from PRNN.pipeline.image_data import ImageDataModule

get_system_and_backend()

model = PRUNe.load_from_checkpoint(
    checkpoint_path="../trained_models/SRN3Dv3_optim.ckpt",
    map_location=torch.device("cpu")
).eval()

data_fname = "flowers_n5000_npix64.h5"

data = ImageDataModule(
    data_fname,
    batch_size=100,
    num_workers=0,
    nbar_signal=(1e3, 2e3),
    nbar_bkgrnd=(1e1, 1e2),
    nframes=32,
    shuffle=True,
    randomize=True,
    experimental=False,
)

data.setup()
X, Y = next(iter(data.train_dataloader()))
# plt.imshow(X[0, 1, :, :], cmap="twilight_shifted")

raise RuntimeError

# Load experimental data set and SVD phase
PI = PhaseImages()
PI.load_sim_data(X, Y)
PI.model_reconstructions(model)
PI.svd_reconstructions()
PRNN.pipeline.PhaseImages.optimize_global_phases(type='nn')
PRNN.pipeline.PhaseImages.optimize_global_phases(type='svd')
PI.phase_to_norm()
PI.compute_losses()
PI.plot_phase_images(idx=2)

# PI.save()
