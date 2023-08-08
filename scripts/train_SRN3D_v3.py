import wandb
import torch
import pytorch_lightning as pl
import os
from torch import nn
from pytorch_lightning.loggers import WandbLogger
from QIML.models.utils import get_encoded_size
from QIML.pipeline.QI_data import QIDataModule

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# import torch, gc
# gc.collect()
# torch.cuda.empty_cache()

if __name__ == "__main__":
    from QIML.models.QI_models import SRN3D_v3

    pl.seed_everything(42)

    model = SRN3D_v3(
        depth=6,
        channels=64,
        pixel_kernels=(5, 3),
        frame_kernels=(3, 3),
        pixel_downsample=4,
        frame_downsample=32,
        dropout=0.0,
        activation="GELU",
        norm=True,
        ssim_weight=1.0,
        window_size=15,
        lr=5e-4,
        weight_decay=1e-5,
        fwd_skip=False,
        sym_skip=True,
        plot_interval=2,  # training
    )

    # data_fname = 'flowers_curated_n495_npix64.h5'
    data_fname = "flowers_n5000_npix64.h5"
    # data_fname = 'QIML_mnist_data_n10000_npix64.h5'
    # data_fname = 'flowers_n600_npix64.h5'
    # data_fname = "flowers_n10000_npix64.h5"

    data = QIDataModule(
        data_fname,
        batch_size=100,
        num_workers=0,
        nbar=(1e3, 2e3),
        nframes=32,
        shuffle=True,
        randomize=True,
        flat_background=0,
    )
    #
    z, _, out = get_encoded_size(
        data, model
    )  # to ensure frame dimension is compressed to 1
    print(z.shape)

    # raise RuntimeError

    logger = WandbLogger(
        project="SRN3D_final",
        entity="aproppe",
        # save_dir='/Users/andrewproppe/Desktop/g2-pcfs_backup/wandb_garbage',
        # mode="offline",
        mode="online",
        # log_model=True,
    )

    trainer = pl.Trainer(
        max_epochs=1000,
        logger=logger,
        enable_checkpointing=True,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=[2],
    )

    trainer.fit(model, data)
    trainer.save_checkpoint("SRN3Dv3_final_window15.ckpt")
    # new_model = MyModel.load_from_checkpoint(checkpoint_path="example.ckpt")

    wandb.finish()
