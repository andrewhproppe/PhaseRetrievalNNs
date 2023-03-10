import wandb
import torch
import pytorch_lightning as pl
import warnings
import os
from pytorch_lightning.loggers import WandbLogger
from QIML.models.QI_models import models
from QIML.pipeline.QI_data import QIDataModule
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


if __name__ == '__main__':

    nframes = 16
    nx, ny = 32, 32

    from QIML.models.QI_models import QI3Dto2DConvAE, SRN3D

    # model = QI3Dto2DConvAE(
    #     input_dim=(1, 1, nframes, nx, ny),
    #     channels=[1, 20, 20, 20],
    #     flat_bottleneck=False,
    #     plot_interval=100,
    #     residual=False,
    # )


    model = SRN3D(
        depth=4,
        channels=[1, 4, 8, 16, 32, 64],
        strides=[2, 2, 1, 1, 1, 1],
        layers=[1, 1, 1, 1, 1],
        residual=True,
    )

    # decide to train on GPU or CPU based on availability or user specified
    if not torch.cuda.is_available():
        GPU = 0
    else:
        GPU = 1

    logger = WandbLogger(
        entity="aproppe",
        project="SRN3D",
        log_model=False,
        save_code=False,
        offline=True,
    )

    trainer = pl.Trainer(
        max_epochs=100,
        gpus=int(torch.cuda.is_available()),
        logger=logger,
        checkpoint_callback=False,
        enable_checkpointing=False,
    )

    # data_fname = 'QIML_data_n100_nbar10000_nframes16_npix32.h5'
    data_fname = 'QIML_data_n100_nbar10000_nframes16_npix32.h5'
    data = QIDataModule(data_fname, batch_size=10)

    trainer.fit(model, data)
