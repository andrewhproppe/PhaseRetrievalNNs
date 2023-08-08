from QIML.visualization.AP_figs_funcs import *
from matplotlib import pyplot as plt

"""
visualize.py

Include routines here that will visualize parts of your analysis.

This can include taking serialized models and seeing how a trained
model "sees" inputs at each layer, as well as just making figures
for talks and writeups.
"""


def plot_frames(frames, nrows=4, ncols=None, figsize=(4, 4), dpi=150, cmap="gray"):
    if ncols is None:
        ncols = nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(frames[i], cmap=cmap)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    dress_fig()
