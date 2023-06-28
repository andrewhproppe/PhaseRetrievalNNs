import wandb
import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import WandbLogger
from QIML.models.utils import get_encoded_size, VGGPerceptualLoss
from QIML.pipeline.QI_data import QIDataModule
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

if __name__ == '__main__':
    from QIML.models.QI_models import VTAE
    # pl.seed_everything(42)

    transformer_args = {
        'nframe': 64,
        'npixel': 32,
        'patch_size': 4,
        'hidden_dim': 256,
        'num_heads': 4,
        'num_layers': 6,
        'dropout': 0.1,
    }

    model = VTAE(
        transformer_args,
        lr=1e-3,
        weight_decay=1e-5,
        plot_interval=1,
    )

    # data_fname = 'QIML_emoji_data_n2000_npix64.h5'
    # data_fname = 'QIML_mnist_data_n10000_npix32.h5'
    # data_fname = 'QIML_mnist_data_n3000_npix32.h5'
    # data_fname = 'QIML_mnist_data_n10000_npix64.h5'
    data_fname = 'QIML_flowers_data_n600_npix32.h5'
    # data_fname = 'QIML_flowers_data_n3000_npix64.h5'
    # data_fname = 'QIML_flowers_data_n10000_npix64.h5'
    # data_fname = 'QIML_mnist_data_n10_npix32.h5'

    data = QIDataModule(data_fname, batch_size=10, num_workers=0, nbar=1e4, nframes=64)

    # data.setup()
    # batch = next(iter(data.train_dataloader()))[0]
    # out = model(batch)[0]
    # z, _, out = get_encoded_size(data, model) # to ensure frame dimension is compressed to 1
    # print(z.shape)

    # raise RuntimeError

    logger = WandbLogger(
        project="VTAE",
        entity="aproppe",
        mode="offline",
        # mode="online",
        # log_model=True,
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