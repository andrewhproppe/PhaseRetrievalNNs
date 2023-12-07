import torch
from PRNN.models.base import PRUNe
from PRNN.visualization.figure_utils import *
from PRNN.pipeline.image_data import ImageDataModule

nbar = 1e3

# Model trained with nbar = 1e2 - 1e5
model1 = PRUNe.load_from_checkpoint(checkpoint_path="../../trained_models/bkgd_free/feasible-tree-4.ckpt", ).eval()

# Model trained with nbar = 1e3 - 1.1e3
model2 = PRUNe.load_from_checkpoint(checkpoint_path="../../trained_models/bkgd_free/still-bush-7.ckpt", ).eval()

data = ImageDataModule(
    "flowers_n5000_npix64.h5",
    batch_size=100,
    num_workers=0,
    nbar_signal=(1e3, 1.1e3),
    nbar_bkgrnd=(0, 0),
    nframes=32,
    shuffle=True,
    randomize=True,
)
data.setup()

X, Y = batch = next(iter(data.train_dataloader()))

with torch.no_grad():
    yhat1, _ = model1(X.to(model1.device))
    yhat2, _ = model2(X.to(model2.device))

mse = torch.nn.MSELoss()

mse1 = mse(yhat1, Y.to(model1.device))
print('MSE for model trained with nbar = 1e2 - 1e5:', mse1.cpu())

mse2 = mse(yhat2, Y.to(model2.device))
print('MSE for model trained with nbar = 1e3 - 1.1e3:', mse2.cpu())
