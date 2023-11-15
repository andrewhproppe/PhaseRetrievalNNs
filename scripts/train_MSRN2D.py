import wandb
import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import WandbLogger
from QIML.pipeline.QI_data import QIDataModule
from QIML.models.base import MSRN2D
# from QIML.models.utils import PerceptualLoss

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

if __name__ == '__main__':
    # data_fname = 'flowers_n5000_npix32.h5'
    data_fname = 'flowers_n600_npix32.h5'

    # data = QIDataModule(data_fname, batch_size=50, num_workers=0, nbar=1e4, nframes=1000, corr_matrix=True, fourier=False, shuffle=True)

    data = QIDataModule(
        data_fname,
        batch_size=20,
        num_workers=0,
        nbar_signal=(0.1e5, 2e5),
        nbar_bkgrnd=(1e6, 1.3e6),
        nframes=1000,
        corr_matrix=True,
        fourier=False,
        shuffle=True,
        randomize=True,
        # experimental=True,
    )

    # Multiscale resnet using correlation matrix
    encoder_args = {
        'first_layer_args': {'kernel_size': (3, 3), 'stride': (2, 2), 'padding': (1, 1)},
        # 'first_layer_args': {'kernel_size': (1, 1), 'stride': (1, 1), 'padding': (1, 1)},
        'nbranch': 3,
        'branch_depth': 5,
        'kernels': [3, 3, 3, 3, 3, 3],
        'channels': [8, 32, 64, 128, 256, 256],
        # 'channels': [8, 16, 16, 16, 16, 16],
        'strides': [4, 2, 2, 2, 2, 2],
        'dilations': [1, 3, 9, 4, 5, 6],
        'activation': torch.nn.PReLU,
        'dropout': 0.3,
        'residual': True,
        'fourier': False,
    }

    # Deconv decoder
    decoder_args = {
        'depth': 3,
        'channels': [256, 128, 64, 32, 16],
        # 'channels': [16, 16, 16, 16, 16],
        # 'mode': 'bilinear'
    }

    model = MSRN2D(
        encoder_args,
        decoder_args,
        lr=5e-4,
        weight_decay=1e-6,
        # metric=torch.nn.L1Loss,
        # metric=PerceptualLoss,
        plot_interval=1,  # training
    )

    # Look at encoded size before training
    data.setup()
    batch = next(iter(data.train_dataloader()))
    X = batch[0][0:3, :, :]
    # some shape tests before trying to actually train
    z = model.encoder(X)
    d = model.decoder(z)
    print(z.shape)
    print(d.shape)

    logger = WandbLogger(
        project="MSRN2D",
        entity="aproppe",
        # save_dir='/Users/andrewproppe/Desktop/g2-pcfs_backup/wandb_garbage',
        mode="offline",
        # mode="online",
        # log_model=True,
    )

    trainer = pl.Trainer(
        max_epochs=1000,
        logger=logger,
        # enable_checkpointing=True,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=1,
    )

    trainer.fit(model, data)

    wandb.finish()