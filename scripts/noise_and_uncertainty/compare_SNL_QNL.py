import matplotlib.pyplot as plt

from PRNN.pipeline.ImageReconNoiseExpt import ImageReconNoiseExpt
from PRNN.models.base import PRUNe
from PRNN.visualization.figure_utils import *

# a = torch.tensor([5, 100, 500, 1000, 5000, 10000])
# b = 1/a
# plt.scatter(a, b)
# plt.plot(a, b)
# plt.xscale('log')
# plt.yscale('log')

# raise RuntimeError

# Load model
model = PRUNe.load_from_checkpoint(
    checkpoint_path="../../trained_models/bkgd_free/feasible-tree-4.ckpt",
    # map_location=torch.device("cpu")
).eval()

expt = ImageReconNoiseExpt(
    model,
    idx=490,
    nbar=[1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5],
    # nbar=[2e3],
    nsamples=10000,
)

expt.run_experiment(keep_yhat=True, save=False)

expt.plot_model_metric_vs_nbar(pixel_positions=(32, 32), sum_over_image=False)
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
# Check to make sure the value at a given pixel is converged with sufficient samples
# expt.calculate_std_and_fit(expt.yhats[-1].cpu(), (32, 32), [1000, 2000, 4000, 8000, 10000], bins=100)



# print(f'{expt.model_mse[0].sum()}')
# print(f'{expt.model_mse[1].sum()}')
# print(f'{expt.model_mse[2].sum()}')