import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import WandbLogger
from QIML.pipeline.QI_data import QIDataModule
from QIML.models.QI_models import MSRN2D
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

if __name__ == '__main__':
    # data_fname = 'QIML_mnist_data_n10_npix32.h5'
    data_fname = 'QIML_mnist_data_n2000_npix32.h5'

    data = QIDataModule(data_fname, batch_size=100, num_workers=0, nbar=1e4, nframes=64, corr_matrix=True)

    # Multiscale resnet using correlation matrix
    encoder_args = {
        'first_layer_args': {'kernel_size': (3, 3), 'stride': (2, 2), 'padding': (2, 2)},
        'nbranch': 4,
        'branch_depth': 5,
        'kernels': [3, 5, 7, 9, 11],
        'channels': [8, 16, 32, 64, 128],
        'strides': [2, 2, 2, 2, 2, 2],
        'dilations': [1, 2, 3, 4, 5, 2],
        'activation': torch.nn.ReLU,
        'residual': False,
    }

    # Deconv decoder
    decoder_args = {
        'depth': 2,
        'channels': [1, 64, 64, 64, 128],
    }

    model = MSRN2D(
        encoder_args,
        decoder_args,
        z_size=64,
        lr=5e-4,
        weight_decay=1e-4,
        plot_interval=5,  # training
    )

    logger = WandbLogger(
        entity="aproppe",
        project="MSRN2D",
        log_model=False,
        offline=True,
    )

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=logger,
        enable_checkpointing=False,
    )

    trainer.fit(model, data)