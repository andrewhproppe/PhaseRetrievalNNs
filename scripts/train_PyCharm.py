import wandb
import torch
import pytorch_lightning as pl
import warnings
import os
from pytorch_lightning.loggers import WandbLogger
from QIML.models.QI_models import models
from QIML.utils import get_encoded_size
from QIML.pipeline.QI_data import QIDataModule
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


if __name__ == '__main__':
    from QIML.models.QI_models import QI3Dto2DConvAE, SRN3D
    pl.seed_everything(42)

    # model = QI3Dto2DConvAE(
    #     input_dim=(1, 1, nframes, nx, ny),
    #     channels=[1, 20, 20, 20],
    #     flat_bottleneck=False,
    #     plot_interval=100,
    #     residual=False,
    # )

    # raise RuntimeError

    # noinspection PyTypeChecker
    model = SRN3D(
        first_layer_args={'kernel': (9, 7, 7), 'stride': (9, 2, 2), 'padding': (0, 3, 3)},
        last_layer_args={'kernel': (7, 7), 'stride': (2, 2), 'padding': (3, 3)},
        depth=4,
        channels=[1, 32, 64, 128, 256, 512],
        strides=[1, 2, 2, 1, 1, 1],
        layers=[1, 2, 1, 1, 1],
        dropout=[0.0, 0.1, 0.1, 0.1],
        lr=1e-3,
        weight_decay=1e-5,
        # plot_interval=2, # debugging
        plot_interval=20,  # training
    )

    # decide to train on GPU or CPU based on availability or user specified
    if not torch.cuda.is_available():
        GPU = 0
    else:
        GPU = 1

    # data_fname = 'QIML_data_n100_nbar10000_nframes16_npix32.h5'
    # data_fname = 'QIML_data_n100_nbar10000_nframes16_npix32.h5'
    # data_fname = 'QIML_data_n1000_nbar10000_nframes32_npix32.h5'
    # data_fname = 'QIML_3logos_data_n2000_nbar10000_nframes64_npix64.h5'
    data_fname = 'QIML_3logos_data_n2000_nbar1000_nframes64_npix64.h5'
    # data_fname = 'QIML_data_n64_nbar10000_nframes32_npix64.h5'
    data = QIDataModule(data_fname, batch_size=100, num_workers=0)

    # z, _ = get_encoded_size(data, model) # to ensure frame dimension is compressed to 1
    # print(z.shape)

    # raise RuntimeError

    wandb.init(
        project="QIML",
        entity="aproppe",
        mode="offline"
        # mode="online"
    )

    logger = WandbLogger(
        log_model=True,
    )

    trainer = pl.Trainer(
        max_epochs=1000,
        gpus=int(torch.cuda.is_available()),
        logger=logger,
        enable_checkpointing=False,
    )

    trainer.fit(model, data)

    wandb.finish()