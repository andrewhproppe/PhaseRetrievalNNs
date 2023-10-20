import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import WandbLogger
from QIML.pipeline.QI_data import QIDataModule
from QIML.models.base import MSRN2D
# from QIML.models.utils import PerceptualLoss

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

if __name__ == '__main__':
    # data_fname = 'QIML_mnist_data_n10_npix32.h5'
    # data_fname = 'QIML_mnist_data_n10000_npix32.h5'
    # data_fname = 'QIML_emojis_data_n2000_npix32.h5'
    data_fname = 'flowers_n600_npix32.h5'
    # data_fname = 'QIML_flowers_data_n10000_npix32.h5'


    data = QIDataModule(data_fname, batch_size=50, num_workers=0, nbar=1e4, nframes=1000, corr_matrix=True, fourier=False, shuffle=True)

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
        lr=1e-4,
        weight_decay=1e-4,
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

    raise RuntimeError

    logger = WandbLogger(
        entity="aproppe",
        project="MSRN2D",
        log_model=False,
        offline=False,
    )

    trainer = pl.Trainer(
        max_epochs=300,
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        devices=[0],
        logger=logger,
        enable_checkpointing=False,
    )

    trainer.fit(model, data)