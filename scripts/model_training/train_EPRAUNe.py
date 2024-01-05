import wandb
import torch
import os
import time

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from PRNN.pipeline.image_data import ImageDataModule
from PRNN.models.utils import CircularMSELoss
from PRNN.models.base_gen2 import EPRAUNe, SVDAE

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

if __name__ == "__main__":

    # seed_everything(42, workers=True)

    # data_fname = "flowers_n5000_npix64.h5"
    # data_fname = "flowers_n100_npix64_SVD_20231214.h5"
    data_fname = "flowers_pruned_n25600_npix64_Eigen_20240103.h5"

    data = ImageDataModule(
        data_fname,
        batch_size=128,
        num_workers=4,
        pin_memory=True,
        data_type='hybrid',
        shuffle=True,
        randomize=True,
    )

    # data.setup()
    # X, Y, P = next(iter(data.train_dataloader()))

    # svd_model = SVDAE.load_from_checkpoint(
    #     checkpoint_path="../../trained_models/SVDAE/zany-sea-82.ckpt",
    #     map_location=torch.device("cpu")
    # )

    # svd_model = SVDAE(
    #     depth=4,
    #     channels=[2, 32, 64, 128, 256],
    #     pixel_downsample=4,
    # )
    # channels = svd_model.encoder_channels
    # channels[0] = 1

    model = EPRAUNe(
        # SVD_encoder=svd_model.encoder,
        # depth=svd_model.hparams.depth,
        # channels=channels,
        # pixel_downsample=svd_model.hparams.pixel_downsample,
        SVD_encoder=None,
        frame_downsample=32,
        depth=4,
        channels=32,
        pixel_downsample=4,
        attn=[0, 0, 0, 0, 0, 0,],
        lr=5e-4,
        lr_schedule='Cyclic',
        weight_decay=1e-6,
        dropout=0.,
        data_info=data.header
    )

    # out, _ = model(X, P)

    # raise RuntimeError

    logger = WandbLogger(
        project="EPRAUNe",
        entity="aproppe",
        # save_dir='/Users/andrewproppe/Desktop/g2-pcfs_backup/wandb_garbage',
        mode="offline",
        # mode="online",
        # log_model=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(
        # max_epochs=250,
        max_steps=30000,
        logger=logger,
        # enable_checkpointing=True,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=[0],
        log_every_n_steps=20,
        callbacks=[lr_monitor],
        # deterministic=True
        # enable_progress_bar=False,
    )

    trainer.fit(model, data)

    wandb.finish()
