import wandb
import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import WandbLogger
from QIML.pipeline.QI_data import QIDataModule
from QIML.models.base import TransformerAutoencoder
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

sweep_config = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'minimize',
        'name': 'val_loss'
        },
    'parameters': {
        'weight_decay': {'values': [1e-5, 1e-4, 1e-3]},
        'lr': {'values': [5e-4, 1e-3, 1e-2]},
        'dropout': {'values': [0.0, 0.2, 0.4]},
        'patch_dim': {'values': [4, 8, 16, 32, 64]},
        'hidden_dim': {'values': [16, 64]},
        'num_heads': {'values': [2, 4, 8, 16]},
        'num_layers': {'values': [2, 4, 8, 16]},
     }
}


img_size = 32
# data_fname = 'QIML_flowers_data_n10000_npix64.h5'
data_fname = 'QIML_flowers_data_n10000_npix32.h5'
data = QIDataModule(data_fname, batch_size=100, num_workers=0, nbar=1e4, nframes=1000, flat_background=0., corr_matrix=True, shuffle=True)


def train():
    # Default hyperparameters
    config_defaults = dict(
        input_dim=img_size**2,
        output_dim=img_size,
        patch_dim=img_size//1,
        hidden_dim=64,
        num_heads=4,
        num_layers=4,
        dropout=0.,
        decoder='Deconv',
        # decoder='MLP',
        lr=1e-3,
        weight_decay=1e-5,
        plot_interval=1000,
    )

    # Initialize a new wandb run
    wandb.init(
        config=config_defaults,
        project="VTAE_sweeps",
        entity="aproppe",
        mode="online",
    )

    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config

    model = TransformerAutoencoder(**config)

    logger = WandbLogger(log_model='False', save_code='False')

    trainer = pl.Trainer(
        max_epochs=50,
        logger=logger,
        enable_checkpointing=False,
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        devices=[0]
    )

    trainer.fit(model, data)

    return trainer, logger

sweep_id = wandb.sweep(sweep_config, project="VTAE_sweeps")

wandb.agent(sweep_id, function=train, count=50)