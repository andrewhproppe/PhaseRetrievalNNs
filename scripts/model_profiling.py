import wandb
import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import WandbLogger
from PRNN.pipeline.image_data import ImageDataModule
from PRNN.models.utils import CircularMSELoss
from PRNN.models.base_gen2 import PRAUNe
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from torch.profiler import profile, record_function, ProfilerActivity

def train():
    data_fname = "flowers_n5000_npix64_20231212_.h5"
    gpu_num = 3
    device = torch.device("cuda", index=gpu_num)
    torch.cuda.set_device(gpu_num)
    # torch.multiprocessing.set_start_method('spawn')

    data = ImageDataModule(
        data_fname,
        batch_size=128,
        num_workers=4,
        pin_memory=True,
        nbar_signal=(1e2, 1e5),
        nbar_bkgrnd=(0, 0),
        nframes=32,
        shuffle=True,
        randomize=True,
        premade=True,
        device="cpu"
        # experimental=True,
    )

    model = PRAUNe(
        depth=5,
        channels=32,
        pixel_kernels=(5, 3),
        frame_kernels=(5, 3),
        pixel_downsample=4,
        frame_downsample=32,
        attn=[0, 0, 0, 0, 0, 0,],
        activation="GELU",
        norm=True,
        metric=CircularMSELoss,
        lr=5e-4,
        weight_decay=1e-6,
        fwd_skip=True,
        sym_skip=True,
        plot_interval=3,
        data_info=data.data_module_info
    )

    logger = WandbLogger(
        project="PRAUNe",
        entity="aproppe",
        # save_dir='/Users/andrewproppe/Desktop/g2-pcfs_backup/wandb_garbage',
        mode="offline",
        # mode="online",
        # log_model=True,
    )

    trainer = pl.Trainer(
        max_epochs=1,
        logger=logger,
        # enable_checkpointing=True,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=[3],
        log_every_n_steps=35,
        profiler="simple",
        enable_progress_bar=True,
        benchmark=True,
    )

    trainer.fit(model, data)

if __name__ == "__main__":
    train()

# with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
#     with record_function("model_inference"):
#         model(X)
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))