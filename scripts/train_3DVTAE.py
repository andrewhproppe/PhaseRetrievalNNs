import wandb
import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import WandbLogger
from QIML.pipeline.QI_data import QIDataModule
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

if __name__ == '__main__':
    from QIML.models.base import TransformerAutoencoder3D

    data_fname = 'flowers_n5000_npix64.h5'
    data = QIDataModule(
        data_fname,
        batch_size=100,
        num_workers=0,
        nbar_signal=(1e2, 1e3),
        nbar_bkgrnd=(1e1, 1e3),
        nframes=32,
        corr_matrix=False,
        fourier=False,
        shuffle=True,
        randomize=True,
        # experimental=True,
    )

    nframe = data.data_kwargs['nframes']
    input_dim = int(data_fname.split('.h5')[0].split('_')[-1].split('npix')[-1])

    model = TransformerAutoencoder3D(
        nframe=nframe,
        input_dim=input_dim,
        hidden_dim=128,
        patch_dim=4,
        deconv_dim=4,
        deconv_depth=4,
        num_heads=8,
        num_layers=6,
        dropout=0.,
        lr=5e-4,
        weight_decay=1e-6,
        plot_interval=1,
    )

    data.setup()
    batch = next(iter(data.train_dataloader()))[0]
    out = model(batch)[0]
    print(out.shape)

    logger = WandbLogger(
        project="VTAE3D",
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
        devices=[3]
    )

    trainer.fit(model, data)

    wandb.finish()