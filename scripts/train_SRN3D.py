import wandb
import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import WandbLogger
from QIML.models.utils import get_encoded_size
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

    model = SRN3D(
        first_layer_args={'kernel': (16, 5, 5), 'stride': (16, 2, 2), 'padding': (2, 2, 2)},
        depth=4,
        # channels=[1, 32, 64, 128, 256, 512],
        channels=[1, 16, 32, 64, 128, 256],
        pixel_strides=[1, 2, 2, 2, 1, 1],
        frame_strides=[2, 2, 2, 2, 2, 2], # stride for frame dimension
        layers=[1, 1, 1, 1, 1],
        dropout=[0.1, 0.1, 0.2, 0.3],
        lr=5e-4,
        weight_decay=1e-4,
        fwd_skip=True,
        sym_skip=True,
        plot_interval=5,  # training
    )

    # data_fname = 'QIML_nhl_poisson_data_n2000_npix64.h5'
    # data_fname = 'QIML_emoji_data_n2000_npix64.h5'
    # data_fname = 'QIML_mnist_data_n10000_npix32.h5'
    # data_fname = 'QIML_mnist_data_n3000_npix32.h5'
    data_fname = 'QIML_mnist_data_n10_npix32.h5'
    # data_fname = 'QIML_poisson_testset.h5'


    data = QIDataModule(data_fname, batch_size=250, num_workers=0, nbar=1e2, nframes=64)

    z, _ = get_encoded_size(data, model) # to ensure frame dimension is compressed to 1
    print(z.shape)

    # raise RuntimeError

    # wandb.init(
    #
    #     # mode="offline"
    # )

    logger = WandbLogger(
        project="QIML",
        entity="aproppe",
        # mode="offline",
        mode="online",
        log_model=True,
    )

    trainer = pl.Trainer(
        max_epochs=1000,
        logger=logger,
        enable_checkpointing=False,
        # accelerator="gpu",
        # devices=4,
        # strategy="ddp",
        gpus=int(torch.cuda.is_available()),
    )

    trainer.fit(model, data)

    wandb.finish()