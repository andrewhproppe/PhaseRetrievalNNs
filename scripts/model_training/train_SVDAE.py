import wandb
import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import WandbLogger
from PRNN.pipeline.image_data import SVDDataModule
from PRNN.models.base_gen2 import SVDAE

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

if __name__ == "__main__":
    # data_fname = "flowers_n5000_npix64_SVD_20231214.h5"
    data_fname = "flowers_n20000_npix64_SVD_20231214.h5"
    data = SVDDataModule(
        data_fname,
        type='svd',
        batch_size=128,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
        device='cpu',
    )

    model = SVDAE(
        depth=4,
        channels=64,
        pixel_kernels=(5, 3),
        pixel_downsample=4,
        attn=[0, 0, 0, 0, 0, 0,],
        activation="GELU",
        norm=True,
        lr=5e-4,
        weight_decay=1e-4,
        dropout=0.,
        fwd_skip=False,
        sym_skip=True,
        plot_interval=1,
        data_info=data.data_module_info
    )

    logger = WandbLogger(
        project="SVDAE",
        entity="aproppe",
        # mode="offline",
        mode="online",
        # log_model=True,
    )

    trainer = pl.Trainer(
        max_epochs=1000,
        logger=logger,
        # enable_checkpointing=True,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=[0],
        log_every_n_steps=35,
        # enable_progress_bar=False,
    )

    trainer.fit(model, data)

    wandb.finish()
