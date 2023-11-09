import torch
from QIML.models.base import SRAE3D
from QIML.utils import get_system_and_backend
from PhaseImages import PhaseImages
from QIML.pipeline.QI_data import QIDataModule
import matplotlib.pyplot as plt

get_system_and_backend()

model = SRAE3D.load_from_checkpoint(
    checkpoint_path="../trained_models/SRN3Dv3_optim.ckpt",
    map_location=torch.device("cpu")
).eval()

data_fname = "flowers_n5000_npix64.h5"

data = QIDataModule(
    data_fname,
    batch_size=10,
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
PI = PhaseImages(acq_time=0.1, date="20230829")
PI.load_sim_data(idx=10)
PI.model_reconstructions(model)
PI.phase_to_norm()
PI.optimize_global_phases()
PI.compute_losses()
PI.plot_phase_images(idx=2)

# PI.save()
