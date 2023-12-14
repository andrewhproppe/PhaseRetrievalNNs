import wandb
import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import WandbLogger
from PRNN.pipeline.image_data import SVDDataModule
from PRNN.models.base_gen2 import SVDAE

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

if __name__ == "__main__":
    data_fname = "flowers_n100_npix64_SVD_20231214.h5"
    data = SVDDataModule(
        data_fname,
        type='svd',
        batch_size=10,
        num_workers=0,
        shuffle=True,
        device='cpu',
    )

    model = SVDAE(
        depth=5,
        channels=32,
        pixel_kernels=(5, 3),
        pixel_downsample=4,
        attn=[1, 0, 0, 0, 0, 0,],
        activation="GELU",
        norm=True,
        lr=5e-4,
        weight_decay=1e-6,
        dropout=0.,
        fwd_skip=False,
        sym_skip=True,
        plot_interval=3,
        data_info=data.data_module_info
    )

    logger = WandbLogger(
        project="SVDAE",
        entity="aproppe",
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
        log_every_n_steps=20,
        # enable_progress_bar=False,
    )

    trainer.fit(model, data)

    wandb.finish()
