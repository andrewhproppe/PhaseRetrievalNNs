from PRNN.pipeline.ImageReconNoiseExpt import ImageReconNoiseExpt
from PRNN.models.base import PRUNe
from PRNN.visualization.figure_utils import *

idx = 6
nsamples = 1000
nbar = 1e3

# Model trained with nbar = 1e2 - 1e5
model1 = PRUNe.load_from_checkpoint(checkpoint_path="../trained_models/bkgd_free/feasible-tree-4.ckpt", ).eval()
expt1 = ImageReconNoiseExpt(
    model1,
    idx=idx,
    nbar=[nbar],
    nsamples=nsamples,
)
expt1.get_from_h5()
expt1.run_experiment(keep_yhat=True, save=False)

# Model trained with nbar only 1e3 - 1.1e3
model2 = PRUNe.load_from_checkpoint(checkpoint_path="../trained_models/bkgd_free/still-bush-7.ckpt", ).eval()
expt2 = ImageReconNoiseExpt(
    model2,
    idx=idx,
    nbar=[nbar],
    nsamples=nsamples,
)
expt2.get_from_h5()
expt2.run_experiment(keep_yhat=True, save=False)

# Plot the mean MSE for both expts in subplots
fig, ax = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
mse1 = expt1.model_mse[0].cpu()
ax[0].imshow(mse1, label="Model 1")
ax[0].text(32, 4, f"Sum: {mse1.sum():.2f}", horizontalalignment='center', color='w')

mse2 = expt2.model_mse[0].cpu()
ax[1].imshow(mse2, label="Model 2")
ax[1].text(32, 4, f"Sum: {mse2.sum():.2f}", horizontalalignment='center', color='w')

dress_fig(xlabel="Pixel", ylabel="Pixel", title="Mean MSE", tight_layout=True)