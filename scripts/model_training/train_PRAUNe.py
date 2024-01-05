import wandb
import torch
import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from PRNN.pipeline.image_data import ImageDataModule
from PRNN.models.utils import CircularMSELoss
from PRNN.models.base_gen2 import PRAUNe

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

if __name__ == "__main__":

    seed_everything(42, workers=True)

    # data_fname = "flowers_n5000_npix64.h5"
    # data_fname = "flowers_pruned_n5000_npix64_Eigen_20240104.h5"
    # data_fname = "flowers102_n5000_npix64_20240104_test5.h5"
    data_fname = "flowers_pruned_n25600_npix64_Eigen_20240105.h5"

    data = ImageDataModule(
        data_fname,
        batch_size=128,
        num_workers=4,
        pin_memory=True,
        split_type='random',
        data_type='frames',
        premade=True,
    )

    model = PRAUNe(
        depth=5,
        # channels=[1, 64, 128, 256, 256, 256],
        channels=32,
        pixel_kernels=(5, 3),
        frame_kernels=(5, 3),
        pixel_downsample=4,
        frame_downsample=32,
        attn=[0, 0, 0, 0, 0, 0,],
        activation="GELU",
        norm=True,
        lr=5e-4,
        # lr_schedule='Cyclic',
        weight_decay=1e-4,
        dropout=0.,
        plot_interval=3,
        data_info=data.header
    )

    logger = WandbLogger(
        project="PRAUNe",
        entity="aproppe",
        # save_dir='/Users/andrewproppe/Desktop/g2-pcfs_backup/wandb_garbage',
        # mode="offline",
        mode="online",
        # log_model=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(
        # max_epochs=1000,
        max_steps=30000,
        logger=logger,
        # enable_checkpointing=True,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=[1],
        log_every_n_steps=20,
        callbacks=[lr_monitor],
        deterministic=True
        # enable_progress_bar=False,
    )

    trainer.fit(model, data)

    wandb.finish()