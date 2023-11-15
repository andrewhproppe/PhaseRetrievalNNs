from QIML.visualization.figure_utils import *
from QIML.modules.WandbLoss import WandbLoss

"""
Plotting loss functions for optimal models
"""
set_font_size(10, lgnd=0)

# Experimental, high signal
L1 = WandbLoss('devoted-sunset-30.csv')
fig = make_fig(figsize=(3, 2), dpi=150)
plt.xlabel('Steps')
plt.ylabel('Loss')
L1.plot('train_loss', fig, label='Training loss')
L1.plot('val_loss', fig, label='Valdiation loss')
dress_fig()

# Simulated, low signal
L1 = WandbLoss('chocolate-capybara-90.csv')
fig = make_fig(figsize=(3, 2), dpi=150)
plt.xlabel('Steps')
plt.ylabel('Loss')
L1.plot('train_loss', fig, label='Training loss')
L1.plot('val_loss', fig, label='Valdiation loss')
dress_fig()