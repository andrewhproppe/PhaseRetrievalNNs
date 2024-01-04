import wandb
import torch
import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from PRNN.pipeline.image_data import SVDDataModule
from PRNN.models.base_gen2 import SVDAE

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

if __name__ == "__main__":

    seed_everything(42, workers=True)

    # data_fname = "flowers_n5000_npix64_SVD_20231214.h5"
    # data_fname = "flowers_n5000_npix64_SVD_20231220.h5"
    # data_fname = "flowers_n25600_npix64_SVD_20231220.h5"
    # data_fname = "flowers_n25600_npix64_Eigen_20231220.h5"
    # data_fname = "flowers_pruned_n51200_npix64_Eigen_20240103.h5"
    data_fname = "flowers_pruned_n25600_npix64_Eigen_20240103.h5"

    data = SVDDataModule(
        data_fname,
        batch_size=128,
        num_workers=4,
        pin_memory=True,
        split_type='fixed'
    )

    model = SVDAE(
        depth=4,
        # channels=[2, 32, 64, 128, 256, 256],
        channels=[2, 64, 128, 256, 256, 256],
        # channels=16,
        pixel_kernels=(5, 3),
        pixel_downsample=4,
        attn=[0, 0, 0, 0, 0, 0,],
        activation="GELU",
        norm=True,
        lr=5e-4,
        lr_schedule='Cyclic',
        weight_decay=1e-6,
        dropout=0.,
        plot_interval=5,
        data_info=data.data_module_info
    )

    logger = WandbLogger(
        project="SVDAE",
        entity="aproppe",
        # mode="offline",
        mode="online",
        # log_model=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(
        max_epochs=250,
        logger=logger,
        # enable_checkpointing=True,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=[0],
        log_every_n_steps=35*4,
        callbacks=[lr_monitor],
        deterministic=True
        # enable_progress_bar=False,
    )

    trainer.fit(model, data)

    wandb.finish()