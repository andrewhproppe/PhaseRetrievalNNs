import torch
from QIML.models.base import PRUNe
from QIML.utils import get_system_and_backend
from PhaseImages import PhaseImages
get_system_and_backend()


model = PRUNe.load_from_checkpoint(
    checkpoint_path="../trained_models/SRN3D_expt_gdl_0.025ms.ckpt",
    # map_location=torch.device("cpu")
).eval()

# Load experimental data set and SVD phase
PI = PhaseImages(acq_time=0.025, date="20230829")
PI.load_expt_data(idx=1000)
PI.model_reconstructions(model)
PI.phase_to_norm()
PI.optimize_global_phases(type='nn')
PI.compute_losses()
PI.error_histograms()
PI.plot_phase_images(idx=2)


PI.save()
