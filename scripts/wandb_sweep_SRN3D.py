import wandb
import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import WandbLogger
from QIML.models.utils import get_encoded_size, VGGPerceptualLoss
from QIML.pipeline.QI_data import QIDataModule
from QIML.models.QI_models import QI3Dto2DConvAE, SRN3D
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

sweep_config = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'minimize',
        'name': 'val_loss'
        },
    'parameters': {
        'weight_decay': {'values': [1e-5, 1e-4, 1e-3, 1e-2]},
        'lr': {'values': [1e-4, 1e-3, 1e-2]},
        'dropout': {'values': [0.0, 0.2, 0.4]},
        'depth': {'values': [6, 7, 8, 9]},
     }
}


data_fname = 'QIML_flowers_data_n10000_npix64.h5'
data = QIDataModule(data_fname, batch_size=100, num_workers=0, nbar=1e4, nframes=64)

def train():
    # Default hyperparameters
    config_defaults = dict(
        first_layer_args={'kernel': (3, 3, 3), 'stride': (2, 2, 2), 'padding': (1, 1, 1)},
        depth=5,
        # channels=32,
        channels=[1, 32, 64, 64, 128, 128, 128, 256, 256, 256],
        pixel_strides=[2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        frame_strides=[2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1],
        dropout=0.2,
        lr=1e-3,
        weight_decay=1e-4,
        fwd_skip=True,
        sym_skip=True,
        perceptual_loss=None,
        plot_interval=1000,
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

    model = SRN3D(**config)

    logger = WandbLogger(log_model='False', save_code='False')

    trainer = pl.Trainer(
        max_epochs=100,
        logger=logger,
        enable_checkpointing=False,
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        devices=1
    )

    trainer.fit(model, data)

    return trainer, logger

sweep_id = wandb.sweep(sweep_config, project="SRN3D_sweeps")

wandb.agent(sweep_id, function=train, count=30)