import wandb
import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import WandbLogger
from QIML.models.utils import get_encoded_size
from QIML.pipeline.QI_data import QIDataModule

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

if __name__ == "__main__":
    from QIML.models.base import PRUNe

    model = PRUNe(
        depth=6,
        channels=64,
        pixel_kernels=(5, 3),
        frame_kernels=(5, 3),
        pixel_downsample=4,
        frame_downsample=32,
        activation="GELU",
        norm=True,
        ssim_weight=1.0,
        window_size=11,
        lr=5e-4,
        weight_decay=1e-6,
        fwd_skip=True,
        sym_skip=True,
        plot_interval=3,
    )

    data_fname = "flowers_n5000_npix64.h5"
    # data_fname = "flowers_expt_n5000_npix64_0.05ms.h5"

    data = QIDataModule(
        data_fname,
        batch_size=100,
        num_workers=0,
        nbar_signal=(0.1e5, 2e5),
        nbar_bkgrnd=(1e6, 1.3e6),
        nframes=32,
        shuffle=True,
        randomize=True,
        # experimental=True,
    )

    # # to ensure frame dimension is compressed to 1
    # z, _, out = get_encoded_size(data, model)
    # print(z.shape)
    #
    # # raise RuntimeError

    logger = WandbLogger(
        project="PRUNe_final",
        entity="aproppe",
        # save_dir='/Users/andrewproppe/Desktop/g2-pcfs_backup/wandb_garbage',
        # mode="offline",
        mode="online",
        # log_model=True,
    )

    trainer = pl.Trainer(
        max_epochs=1000,
        logger=logger,
        # enable_checkpointing=True,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=[3],
    )

    trainer.fit(model, data)

    wandb.finish()
