import wandb
import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import WandbLogger
from QIML.models.utils import get_encoded_size, VGGPerceptualLoss
from QIML.pipeline.QI_data import QIDataModule
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

if __name__ == '__main__':
    from QIML.models.QI_models import QI3Dto2DConvAE, SRN3D

    pl.seed_everything(42)

    model = SRN3D(
        first_layer_args={'kernel': (5, 5, 5), 'stride': (2, 2, 2), 'padding': (2, 2, 2)},
        depth=5,
        # channels=[1, 32, 64, 128, 256, 512],
        # channels=[1, 16, 32, 64, 128, 256, 512],
        # channels=[1, 32, 32, 32, 32, 32, 32, 32],
        channels=[1, 64, 64, 64, 64, 64, 64, 64],
        # channels=[1, 128, 128, 128, 128, 128, 128, 128],
        pixel_strides=[2, 2, 2, 1, 1, 1, 1, 1],
        frame_strides=[2, 2, 2, 2, 2, 2, 1, 1], # stride for frame dimension
        dropout=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        lr=5e-4,
        weight_decay=1e-4,
        fwd_skip=True,
        sym_skip=True,
        # metric=VGGPerceptualLoss,
        # perceptual_loss=True,
        plot_interval=5,  # training
    )

    # data_fname = 'QIML_emoji_data_n2000_npix64.h5'
    # data_fname = 'QIML_mnist_data_n10000_npix32.h5'
    # data_fname = 'QIML_mnist_data_n3000_npix32.h5'
    # data_fname = 'QIML_mnist_data_n10000_npix64.h5'
    # data_fname = 'QIML_flowers_data_n600_npix64.h5'
    data_fname = 'QIML_flowers_data_n3000_npix64.h5'
    # data_fname = 'QIML_mnist_data_n10_npix32.h5'

    data = QIDataModule(data_fname, batch_size=100, num_workers=0, nbar=2e3, nframes=64)

    z, _, out = get_encoded_size(data, model) # to ensure frame dimension is compressed to 1
    print(z.shape)
    print(out[0].shape)

    # raise RuntimeError

    logger = WandbLogger(
        project="SRN3D",
        entity="aproppe",
        log_model=False,
        # mode="offline",
        mode="online",
    )

    trainer = pl.Trainer(
        max_epochs=1000,
        logger=logger,
        enable_checkpointing=False,
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        devices=[0]
    )

    trainer.fit(model, data)

    wandb.finish()