import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import WandbLogger
from QIML.pipeline.QI_data import QIDataModule
from QIML.models.QI_models import MSRN2D
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

if __name__ == '__main__':
    data_fname = 'QIML_mnist_data_n10_npix32.h5'

    data = QIDataModule(data_fname, batch_size=8, num_workers=0, nbar=1e4, nframes=64, corr_matrix=True)

    # Multiscale resnet using correlation matrix
    encoder_args = {
        'first_layer_args': {'kernel_size': (7, 7), 'stride': (2, 2), 'padding': (3, 3)},
        'nbranch': 3,
        'branch_depth': 5,
        'kernels': [3, 7, 21, 28, 56],
        'channels': [4, 8, 16, 32, 64],
        'strides': [2, 2, 2, 2, 2, 2],
        'dilations': [1, 2, 3, 4, 2, 2],
        'activation': torch.nn.ReLU,
        'residual': False,
    }

    # # MLP decoder
    # decoder_args = {
    #     'out_dim': 32*32,
    #     'depth': 2,
    #     'activation': nn.ReLU,
    #     'residual': False,
    # }

    # Deconv decoder
    decoder_args = {
        'depth': 2
    }

    model = MSRN2D(
        encoder_args,
        decoder_args
    )

    logger = WandbLogger(
        entity="aproppe",
        project="MSRN2D",
        log_model=False,
        save_code=False,
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