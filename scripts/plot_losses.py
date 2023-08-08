import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

loss_path = "../data/training_losses/"
fname = "autumn-water-75.csv"
data = pd.read_csv(loss_path + fname, skiprows=1, header=None)
data = data.to_numpy()
data = data[:, [0, 1, 4, 7, 11]]
step = data[:, 0]

train_loss = data[:, 1]
train_step = step[np.where(train_loss > 0)]
train_loss = train_loss[np.where(train_loss > 0)]

val_loss = data[:, 2]
val_step = step[np.where(val_loss > 0)]
val_loss = val_loss[np.where(val_loss > 0)]

mse = data[:, 3]
mse_step = step[np.where(mse > 0)]
mse = mse[np.where(mse > 0)]

ssim = data[:, 4]
ssim_step = step[np.where(ssim > 0)]
ssim = ssim[np.where(ssim > 0)]

# losses = np.loadtxt(loss_path + fname, delimiter=",", skiprows=1)
