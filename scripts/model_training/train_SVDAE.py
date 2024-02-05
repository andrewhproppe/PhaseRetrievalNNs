import wandb
import torch
import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging
from PRNN.pipeline.image_data import SVDDataModule
from PRNN.models.base_gen2 import SVDAE

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

if __name__ == "__main__":

    seed_everything(666, workers=True)

    data_fname = "flowers102_n5000_npix64_Eigen_20240110_test10.h5"
    # data_fname = "eigen_devset.h5"

    data = SVDDataModule(
        data_fname,
        batch_size=128,
        num_workers=4,
        pin_memory=True,
        split_type='random'
    )

    model = SVDAE(
        depth=6,
        channels=128,
        pixel_kernels=(5, 3),
        pixel_downsample=4,
        attn=[0, 0, 0, 0, 0, 0, 0, 0],
        activation="GELU",
        lr=5e-4,
        lr_schedule='Cyclic',
        data_info=data.header
    )

    # model = SVDAE.load_from_checkpoint(
    #     # 'SVDAE/wbkvv28u/checkpoints/svdae_step15k.ckpt'
    #     '../../trained_models/SVDAE/treasured-glade-127.ckpt'
    # )

    # raise RuntimeError

    logger = WandbLogger(
        project="SVDAE",
        entity="aproppe",
        mode="offline",
        # mode="online",
        # log_model=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(
        max_epochs=550,
        # max_epochs=200,
        max_steps=20000,
        logger=logger,
        # enable_checkpointing=True,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=[1],
        log_every_n_steps=35,
        callbacks=[
            lr_monitor,
            StochasticWeightAveraging(swa_lrs=1e-5, swa_epoch_start=0.8)
        ],
        gradient_clip_val=1.0,
        deterministic=True,
    )

    trainer.fit(model, data)

    wandb.finish()