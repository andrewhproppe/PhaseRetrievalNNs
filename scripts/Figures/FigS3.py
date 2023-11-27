from PRNN.visualization.figure_utils import *
from PRNN.pipeline.WandbLoss import WandbLoss

"""
Comparing models with and without MSE/SSIM/GDL losses
"""

L1 = WandbLoss('morning-terrain-16.csv') # with MSE, with SSIM, with GDL
L2 = WandbLoss('zany-frost-17.csv') # with MSE, no SSIM, no GDL
L3 = WandbLoss('floral-valley-18.csv') # no MSE, with SSIM, with GDL

# MSE Loss
fig = make_fig(figsize=(3, 2), dpi=150)
plt.xlabel('Steps')
plt.ylabel('MSE Loss')

L1.plot('recon', fig, label='MSE+SSIM+GDL')
L2.plot('recon', fig, label='MSE')
L3.plot('recon', fig, label='SSIM+GDL')

dress_fig()
plt.yscale('log')
plt.xlim([-5, 800])


# SSIM Loss
fig = make_fig(figsize=(3, 2), dpi=150)
plt.xlabel('Steps')
plt.ylabel('SSIM Loss')

L1.plot('ssim', fig, label='MSE+SSIM+GDL')
L2.plot('ssim', fig, label='MSE')
L3.plot('ssim', fig, label='SSIM+GDL')

dress_fig()
plt.yscale('log')
plt.xlim([-5, 800])

# GDL Loss
fig = make_fig(figsize=(3, 2), dpi=150)
plt.xlabel('Steps')
plt.ylabel('GDL Loss')

L1.plot('gdl', fig, label='MSE+SSIM+GDL')
L2.plot('gdl', fig, label='MSE')
L3.plot('gdl', fig, label='SSIM+GDL')

dress_fig()
plt.yscale('log')
plt.xlim([-5, 800])

# Train Loss
fig = make_fig(figsize=(3, 2), dpi=150)
plt.xlabel('Steps')
plt.ylabel('Train Loss')

L1.plot('train_loss', fig, label='MSE+SSIM+GDL')
L2.plot('train_loss', fig, label='MSE')
L3.plot('train_loss', fig, label='SSIM+GDL')

dress_fig()
plt.yscale('log')
plt.xlim([-5, 800])

# Val Loss
fig = make_fig(figsize=(3, 2), dpi=150)
plt.xlabel('Steps')
plt.ylabel('Train Loss')

L1.plot('val_loss', fig, label='MSE+SSIM+GDL')
L2.plot('val_loss', fig, label='MSE')
L3.plot('val_loss', fig, label='SSIM+GDL')

dress_fig()
plt.yscale('log')
plt.xlim([-5, 800])