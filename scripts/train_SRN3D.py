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
    # pl.seed_everything(42)

    model = SRN3D(
        first_layer_args={'kernel': (3, 3, 3), 'stride': (2, 2, 2), 'padding': (1, 1, 1)},
        depth=7,
        channels=128,
        pixel_strides=[2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        frame_strides=[2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1], # stride for frame dimension
        dropout=0.,
        lr=1e-3,
        weight_decay=1e-5,
        fwd_skip=True,
        sym_skip=True,
        plot_interval=1,  # training
    )

    # data_fname = 'flowers_curated_n495_npix64.h5'
    data_fname = 'flowers_n5000_npix64.h5'
    data = QIDataModule(data_fname, batch_size=50, num_workers=0, nbar=(1e3, 2e3), nframes=64, shuffle=True, randomize=True)
    #
    # z, _, out = get_encoded_size(data, model) # to ensure frame dimension is compressed to 1
    # print(z.shape)

    # raise RuntimeError

    logger = WandbLogger(
        project="SRN3D",
        entity="aproppe",
        # mode="offline",
        mode="online",
        # log_model=True,
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