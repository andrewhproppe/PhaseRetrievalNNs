import wandb
import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import WandbLogger
from PRNN.pipeline.image_data import SVDDataModule
from PRNN.models.base_gen2 import SVDAE

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

sweep_config = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "depth": {"values": [4, 5, 6, 7]},
        "pixel_kernels": {"values": [(3, 3), (5, 3), (7, 3)]},
        "pixel_downsample": {"values": [2, 4, 8, 16]},
        "attn": {"values": [ [0, 0, 0, 0, 0, 0, 0, 0], [[1, 1, 0, 0, 0, 0, 0, 0]] ]},
        "dropout": {"values": [0.0, 0.1, 0.2]},
        "activation": {"values": ["ReLU", "SiLU", "PReLU", "LeakyReLU", "GELU"]},
        "norm": {"values": [True, False]},
        "lr": {"values": [1e-4, 5e-4, 1e-3, 2e-3]},
        "weight_decay": {"values": [1e-6, 1e-5, 1e-4, 1e-3]},
    },
}

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


def train():
    # Default hyperparameters
    config_defaults = dict(
        depth=5,
        channels=64,
        pixel_kernels=(5, 3),
        pixel_downsample=4,
        attn=[0, 0, 0, 0, 0, 0, 0, 0],
        dropout=0.0,
        activation="ReLU",
        norm=True,
        lr=5e-4,
        weight_decay=1e-5,
        fwd_skip=True,
        sym_skip=True,
        plot_interval=1000,  # training
    )

    # Initialize a new wandb run
    wandb.init(
        config=config_defaults,
        project="SVDAE_sweeps",
        entity="aproppe",
        mode="online",
    )

    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config

    model = SVDAE(**config)

    logger = WandbLogger(log_model="False", save_code="False")

    trainer = pl.Trainer(
        max_epochs=50,
        logger=logger,
        enable_checkpointing=False,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=[3],
        log_every_n_steps=35,
    )

    trainer.fit(model, data)

    return trainer, logger


sweep_id = wandb.sweep(sweep_config, project="SVDAE_sweeps")

wandb.agent(sweep_id, function=train, count=50)
