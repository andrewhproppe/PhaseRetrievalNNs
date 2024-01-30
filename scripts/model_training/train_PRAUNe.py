import wandb
import torch
import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging
from PRNN.pipeline.image_data import ImageDataModule
from PRNN.models.utils import CircularMSELoss
from PRNN.models.base_gen2 import PRAUNe, EPRAUNe

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

if __name__ == "__main__":

    seed_everything(666, workers=True)

    data_fname = "flowers_n5000_npix64.h5"
    # data_fname = "flowers102_n5000_npix64_Eigen_20240110_test10.h5"

    data = ImageDataModule(
        data_fname,
        batch_size=128,
        num_workers=4,
        pin_memory=True,
        split_type='random',
        data_type='frames',
        premade=False,
    )

    model = PRAUNe(
        depth=6,
        channels=64,
        pixel_downsample=4,
        frame_downsample=32,
        attn=[0, 1, 1, 0, 0, 0,],
        activation="GELU",
        lr=5e-3,
        # lr_schedule='Step',
        plot_interval=3,
        data_info=data.header
    )

    logger = WandbLogger(
        project="PRAUNe",
        entity="aproppe",
        # save_dir='/Users/andrewproppe/Desktop/g2-pcfs_backup/wandb_garbage',
        mode="offline",
        # mode="online",
        # log_model=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(
        # max_epochs=int(850*30000/30000),
        max_epochs=850,
        max_steps=30000,
        logger=logger,
        # enable_checkpointing=True,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=[1],
        log_every_n_steps=20,
        callbacks=[
            lr_monitor,
            StochasticWeightAveraging(swa_lrs=1e-3)
        ],
        gradient_clip_val=1.0,
        deterministic=True,
        # precision=16,
        # enable_progress_bar=False,
    )

    trainer.fit(model, data)

    wandb.finish()