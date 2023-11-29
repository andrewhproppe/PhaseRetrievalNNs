import wandb
import torch
import pytorch_lightning as pl
import os
from torch import nn
from pytorch_lightning.loggers import WandbLogger
from PRNN.models.utils import get_encoded_size
from PRNN.pipeline.image_data import ImageDataModule

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# import torch, gc
# gc.collect()
# torch.cuda.empty_cache()

def ask_model_save(trainer):
    while True:
        save = input("Save this model? (y/n): ")
        if save == "y":
            trainer.save_checkpoint("PRAUN_test.ckpt")
            break
        elif save == "n":
            pass
            break
        else:
            print("Invalid input, select again.")


if __name__ == "__main__":
    from PRNN.models.base import PRAUN

    pl.seed_everything(42)

    attn_args = {
        "image_patch_size": 4,
        "frame_patch_size": 4,
        "embedding_size": 64,
        "hidden_size": 128,
        "head_size": 128,
        "depth": 2,
        "nheads": 4,
        "dropout": 0.0,
    }

    model = PRAUN(
        depth=6,
        # channels=[1, 32, 32, 64, 64, 128, 128],
        channels=64,
        pixel_kernels=(5, 3),
        frame_kernels=(5, 3),
        pixel_downsample=4,
        frame_downsample=32,
        attn_on=[0, 1, 1, 1, 1, 1, 1],
        attn_heads=2,
        attn_depth=2,
        activation="GELU",
        norm=True,
        ssim_weight=1.0,
        window_size=11,
        lr=5e-4,
        weight_decay=1e-6,
        fwd_skip=False,
        sym_skip=True,
        plot_interval=3,  # training
    )

    # data_fname = 'flowers_curated_n495_npix64.h5'
    data_fname = "flowers_n5000_npix64.h5"
    # data_fname = 'mnist_n10000_npix64.h5'
    # data_fname = 'flowers_n600_npix64.h5'
    # data_fname = "flowers_n10000_npix64.h5"

    data = ImageDataModule(
        data_fname,
        batch_size=100,
        num_workers=0,
        nbar_signal=(0.1e5, 2e5),
        nbar_bkgrnd=(1e6, 1.3e6),
        nframes=32,
        shuffle=True,
        randomize=True,
    )

    # to ensure frame dimension is compressed to 1
    z, _, out = get_encoded_size(data, model)
    print(z.shape)

    # raise RuntimeError

    logger = WandbLogger(
        project="SRN3D_bg",
        entity="aproppe",
        # save_dir='/Users/andrewproppe/Desktop/g2-pcfs_backup/wandb_garbage',
        mode="offline",
        # mode="online",
        # log_model=True,
    )

    trainer = pl.Trainer(
        max_epochs=1000,
        logger=logger,
        enable_checkpointing=True,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=[0],
    )

    trainer.fit(model, data)

    ask_model_save(trainer)

    wandb.finish()
