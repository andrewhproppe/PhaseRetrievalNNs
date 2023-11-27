import wandb
import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import WandbLogger
from QIML.pipeline.QI_data import QIDataModule
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

if __name__ == '__main__':
    from QIML.models.base import TransformerAutoencoder

    img_size = 32

    model = TransformerAutoencoder(
        input_dim=img_size**2,
        output_dim=img_size,
        patch_dim=img_size//1,
        hidden_dim=16,
        num_heads=1,
        num_layers=4,
        dropout=0.,
        decoder='Deconv',
        # decoder='MLP',
        lr=5e-4,
        weight_decay=1e-6,
        plot_interval=3,
    )

    data_fname = 'flowers_n5000_npix32.h5'
    # data_fname = 'flowers_n600_npix32.h5'

    data = QIDataModule(
        data_fname,
        batch_size=20,
        num_workers=0,
        nbar_signal=(1e2, 1e3),
        nbar_bkgrnd=(1e1, 1e3),
        nframes=1000,
        corr_matrix=True,
        fourier=False,
        shuffle=True,
        randomize=True,
        # experimental=True,
    )

    logger = WandbLogger(
        project="VTAE32pix",
        entity="aproppe",
        # mode="offline",
        mode="online",
        log_model=False,
    )

    trainer = pl.Trainer(
        max_epochs=1000,
        logger=logger,
        enable_checkpointing=False,
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        devices=[2]
    )

    trainer.fit(model, data)

    wandb.finish()