import wandb
import torch
import pytorch_lightning as pl
import os
from torch import nn
from pytorch_lightning.loggers import WandbLogger
from PRNN.pipeline.image_data import ImageDataModule
from PRNN.models.base import PRUNe

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

sweep_config = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "depth": {"values": [4, 5, 6, 7]},
        "pixel_kernels": {"values": [(3, 3), (5, 3), (7, 3)]},
        "frame_kernels": {"values": [(3, 3), (5, 3), (7, 3)]},
        "pixel_downsample": {"values": [4, 8, 16]},
        "frame_downsample": {"values": [8, 16, 32]},
        "dropout": {"values": [0.0, 0.1, 0.2]},
        "activation": {"values": ["ReLU", "SiLU", "PReLU", "LeakyReLU", "GELU"]},
        "norm": {"values": [True, False]},
        "window_size": {"values": [7, 11, 15]},
        "lr": {"values": [5e-4, 1e-3, 2e-3]},
        "weight_decay": {"values": [1e-6, 1e-5, 1e-4, 1e-3]},
    },
}

data_fname = "flowers_n5000_npix64.h5"

data = ImageDataModule(
    data_fname,
    batch_size=100,
    num_workers=0,
    nbar_signal=(0.5e5, 1e5),
    nbar_bkgrnd=(1e6, 1.3e6),
    nframes=32,
    shuffle=True,
    randomize=True,
)


def train():
    # Default hyperparameters
    config_defaults = dict(
        depth=5,
        channels=32,
        pixel_kernels=(3, 3),
        frame_kernels=(3, 3),
        pixel_downsample=4,
        frame_downsample=32,
        dropout=0.0,
        activation="ReLU",
        norm=True,
        window_size=15,
        lr=5e-4,
        weight_decay=1e-5,
        fwd_skip=False,
        sym_skip=True,
        plot_interval=1000,  # training
    )

    # Initialize a new wandb run
    wandb.init(
        config=config_defaults,
        project="SRN3D_sweeps",
        entity="aproppe",
        mode="online",
    )

    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config

    model = PRUNe(**config)

    logger = WandbLogger(log_model="False", save_code="False")

    trainer = pl.Trainer(
        max_epochs=50,
        logger=logger,
        enable_checkpointing=False,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=[0],
    )

    trainer.fit(model, data)

    return trainer, logger


sweep_id = wandb.sweep(sweep_config, project="SRN3D_sweeps")

wandb.agent(sweep_id, function=train, count=50)
