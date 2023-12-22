import wandb
import torch
import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from PRNN.pipeline.image_data import ImageDataModule
from PRNN.models.base_gen2 import FSVDAE

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

if __name__ == "__main__":

    seed_everything(42, workers=True)

    data_fname = "flowers_n5120_npix64_20231221.h5"

    data = ImageDataModule(
        data_fname,
        batch_size=128,
        num_workers=4,
        pin_memory=True,
        split_type='fixed',
        premade='True,'
    )

    model = FSVDAE(
        depth=4,
        channels=[2, 32, 64, 128, 256, 256],
        # channels=[2, 64, 128, 256, 256, 256],
        pixel_kernels=(5, 3),
        pixel_downsample=4,
        attn=[0, 0, 0, 0, 0, 0,],
        activation="GELU",
        norm=True,
        lr=1e-2,
        weight_decay=1e-6,
        dropout=0.,
        fwd_skip=True,
        sym_skip=True,
        plot_interval=3,
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
        devices=[3],
        log_every_n_steps=35*1,
        callbacks=[lr_monitor],
        deterministic=True
        # enable_progress_bar=False,
    )

    trainer.fit(model, data)

    wandb.finish()
