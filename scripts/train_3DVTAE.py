import wandb
import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import WandbLogger
from QIML.models.utils import get_encoded_size, VGGPerceptualLoss
from QIML.pipeline.QI_data import QIDataModule
from torch import nn
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

if __name__ == '__main__':
    from QIML.models.QI_models import TransformerAutoencoder3D
    # pl.seed_everything(42)

    data_fname = 'QIML_flowers_data_n600_npix64.h5'
    data = QIDataModule(data_fname, batch_size=20, num_workers=0, nbar=1e3, nframes=64, flat_background=0., corr_matrix=False, shuffle=True)

    nframe = data.data_kwargs['nframes']
    input_dim = int(data_fname.split('.h5')[0].split('_')[-1].split('npix')[-1])

    model = TransformerAutoencoder3D(
        nframe=nframe,
        input_dim=input_dim,
        hidden_dim=100,
        patch_dim=4,
        deconv_dim=4,
        num_heads=4,
        num_layers=4,
        dropout=0.
    )

    data.setup()
    batch = next(iter(data.train_dataloader()))[0]
    out = model(batch)[0]
    print(out.shape)

    # raise RuntimeError
    # name = 'flws600_nb%.0e_nf%.0e' % (data.data_kwargs['nbar'], data.data_kwargs['nframes'])

    logger = WandbLogger(
        # name=name,
        project="VTAE3D",
        entity="aproppe",
        mode="offline",
        # mode="online",
        log_model=False,
    )

    trainer = pl.Trainer(
        max_epochs=1000,
        logger=logger,
        enable_checkpointing=False,
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        devices=1
    )

    trainer.fit(model, data)

    wandb.finish()