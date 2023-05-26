import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import WandbLogger
from QIML.pipeline.QI_data import QIDataModule
from QIML.models.QI_models import MSRN2D
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

if __name__ == '__main__':
    # data_fname = 'QIML_mnist_data_n10_npix32.h5'
    data_fname = 'QIML_mnist_data_n10000_npix32.h5'

    data = QIDataModule(data_fname, batch_size=100, num_workers=0, nbar=1e4, nframes=64, corr_matrix=True, fourier=True, shuffle=True)

    # Multiscale resnet using correlation matrix
    encoder_args = {
        'first_layer_args': {'kernel_size': (3, 3), 'stride': (2, 2), 'padding': (1, 1)},
        # 'first_layer_args': {'kernel_size': (1, 1), 'stride': (1, 1), 'padding': (1, 1)},
        'nbranch': 5,
        'branch_depth': 5,
        'kernels': [3, 3, 3, 5, 5, 5],
        'channels': [8, 16, 32, 64, 128, 256],
        'strides': [4, 2, 2, 2, 2, 2],
        'dilations': [1, 2, 3, 1, 2, 3],
        'activation': torch.nn.ReLU,
        'residual': True,
        'fourier': True,
    }

    # Deconv decoder
    decoder_args = {
        'depth': 3,
        'channels': [256, 128, 64, 32, 16],
    }

    model = MSRN2D(
        encoder_args,
        decoder_args,
        lr=5e-4,
        weight_decay=1e-4,
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

    # raise RuntimeError

    logger = WandbLogger(
        entity="aproppe",
        project="MSRN2D",
        log_model=False,
        offline=True,
    )

    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=logger,
        enable_checkpointing=False,
    )

    trainer.fit(model, data)