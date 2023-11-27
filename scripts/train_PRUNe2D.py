import wandb
import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import WandbLogger
from QIML.pipeline.QI_data import QIDataModule
from QIML.models.base import MSRN2D
# from QIML.models.utils import PerceptualLoss

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

if __name__ == '__main__':
    from QIML.models.base import PRUNe2D

    model = PRUNe2D(
        depth=6,
        img_size=64,
        channels=32,
        pixel_kernels=(5, 3),
        pixel_downsample=64,
        activation="GELU",
        norm=True,
        ssim_weight=1.0,
        window_size=11,
        lr=5e-4,
        weight_decay=1e-6,
        fwd_skip=False,
        sym_skip=True,
        plot_interval=3,
    )

    # data_fname = 'flowers_n5000_npix32.h5'
    data_fname = 'flowers_n5000_npix64.h5'
    # data_fname = 'flowers_n600_npix32.h5'

    data = QIDataModule(
        data_fname,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        nbar_signal=(1e2, 1e3),
        nbar_bkgrnd=(1e1, 1e3),
        nframes=1000,
        corr_matrix=True,
        fourier=False,
        shuffle=True,
        randomize=True,
        # experimental=True,
    )

    # Look at encoded size before training
    data.setup()
    batch = next(iter(data.train_dataloader()))
    X = batch[0][0:3, :, :]
    # some shape tests before trying to actually train
    test = model(X)
    z, r = model.encoder(X.unsqueeze(1))
    d = model.decoder(z, r)
    print(z.shape)
    print(d.shape)

    # raise RuntimeError

    logger = WandbLogger(
        project="PRUNe2D",
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
        devices=[1],
    )

    trainer.fit(model, data)

    wandb.finish()