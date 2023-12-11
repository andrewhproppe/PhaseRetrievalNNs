import wandb
import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import WandbLogger
from PRNN.pipeline.image_data import ImageDataModule
from PRNN.models.utils import CircularMSELoss
from PRNN.models.base_gen2 import PRAUNe

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

if __name__ == "__main__":
    data_fname = "flowers_n5000_npix64.h5"
    # data_fname = "mnist_n10000_npix64.h5"
    # data_fname = "flowers_expt_n5000_npix64_0.05ms.h5"

    data = ImageDataModule(
        data_fname,
        batch_size=100,
        num_workers=0,
        nbar_signal=(1e5, 1.1e5),
        nbar_bkgrnd=(0, 0),
        minmax=(-1, 1),
        nframes=32,
        shuffle=True,
        randomize=True,
        # experimental=True,
    )

    model = PRAUNe(
        depth=6,
        channels=4,
        pixel_kernels=(5, 3),
        frame_kernels=(5, 3),
        pixel_downsample=4,
        frame_downsample=32,
        attn=[0, 0, 0, 0, 0, 0,],
        activation="GELU",
        norm=True,
        metric=CircularMSELoss,
        lr=5e-4,
        weight_decay=1e-6,
        fwd_skip=False,
        sym_skip=True,
        plot_interval=3,
    )

    logger = WandbLogger(
        project="PRUNe_noBkgd",
        entity="aproppe",
        # save_dir='/Users/andrewproppe/Desktop/g2-pcfs_backup/wandb_garbage',
        mode="offline",
        # mode="online",
        # log_model=True,
    )

    trainer = pl.Trainer(
        max_epochs=1000,
        logger=logger,
        # enable_checkpointing=True,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=1,
    )

    trainer.fit(model, data)

    wandb.finish()
