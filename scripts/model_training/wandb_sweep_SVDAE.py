import wandb
import torch
import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging

from PRNN.pipeline.image_data import SVDDataModule
from PRNN.models.base_gen2 import SVDAE

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

seed_everything(666, workers=True)

sweep_config = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "depth": {"values": [4, 5, 6, 7]},
        # "channels": {"values": [32, 64, 128, 256]},
        "pixel_kernels": {"values": [(3, 3), (5, 3), (7, 3)]},
        # "pixel_downsample": {"values": [2, 4, 8, 16]},
        # "attn": {"values": [ [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0] ]},
        # "dropout": {"values": [0.0, 0.05, 0.1, 0.2]},
        "activation": {"values": ["ReLU", "PReLU", "SiLU", "LeakyReLU", "GELU"]},
        # "norm": {"values": [True, False]},
        # "lr": {"values": [1e-4, 5e-4, 1e-3, 2e-3]},
        "weight_decay": {"values": [0.0, 1e-7, 1e-6]},
    },
}

# data_fname = "flowers_pruned_n25600_npix64_Eigen_20240105.h5"
data_fname = "flowers102_n5000_npix64_Eigen_20240110_test10.h5"

data = SVDDataModule(
    data_fname,
    batch_size=128,
    num_workers=4,
    pin_memory=True,
    split_type='random'
)

def train():
    # Default hyperparameters
    config_defaults = dict(
        depth=6,
        channels=128,
        pixel_kernels=(5, 3),
        pixel_downsample=4,
        attn=[0, 0, 0, 0, 0, 0, 0, 0],
        dropout=0.0,
        activation="GELU",
        norm=True,
        lr=5e-4,
        lr_schedule='Cyclic',
        weight_decay=1e-6,
        plot_interval=1000,
        data_info=data.header
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

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(
        max_epochs=110*5,
        max_steps=20000,
        logger=logger,
        # enable_checkpointing=True,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=[2],
        log_every_n_steps=35,
        callbacks=[
            lr_monitor,
            StochasticWeightAveraging(swa_lrs=1e-4)
        ],
        gradient_clip_val=1.0,
        deterministic=True,
        # precision="16-mixed",
        # enable_progress_bar=False,
    )

    trainer.fit(model, data)

    return trainer, logger


sweep_id = wandb.sweep(sweep_config, project="SVDAE_sweeps")

wandb.agent(sweep_id, function=train, count=100)
