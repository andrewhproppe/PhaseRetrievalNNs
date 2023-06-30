import wandb
import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import WandbLogger
from QIML.models.utils import get_encoded_size, VGGPerceptualLoss
from QIML.pipeline.QI_data import QIDataModule
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

if __name__ == '__main__':
    from QIML.models.QI_models import VTAE, TransformerAutoencoder
    # pl.seed_everything(42)

    # transformer_args = {
    #     'nframe': 32,
    #     'npixel': 32**2,
    #     'patch_size': 4,
    #     'hidden_dim': 256,
    #     'num_heads': 4,
    #     'num_layers': 6,
    #     'dropout': 0.,
    #     'dimension': 2,
    # }

    # output_dim = 32
    # input_dim = output_dim**2
    # hidden_dim = 100
    # num_heads = 4
    # num_layers = 6
    # dropout = 0.1
    #
    # transformer_args = {
    #     'input_dim': output_dim**2,
    #     'output_dim': output_dim,
    #     'patch_dim': output_dim,
    #     'hidden_dim': 16,
    #     'num_heads': 2,
    #     'num_layers': 2,
    #     'dropout': 0.1,
    # }
    img_size = 32

    model = TransformerAutoencoder(
        input_dim=img_size**2,
        output_dim=img_size,
        patch_dim=img_size//2,
        hidden_dim=64,
        num_heads=16,
        num_layers=16,
        dropout=0.,
        decoder='Deconv',
        # decoder='MLP',
        lr=1e-3,
        weight_decay=1e-5,
        plot_interval=1,
    )

    # data_fname = 'QIML_emoji_data_n2000_npix64.h5'
    # data_fname = 'QIML_mnist_data_n10000_npix32.h5'
    # data_fname = 'QIML_mnist_data_n3000_npix32.h5'
    # data_fname = 'QIML_mnist_data_n10000_npix64.h5'
    # data_fname = 'QIML_flowers_data_n600_npix32.h5'
    # data_fname = 'QIML_flowers_data_n3000_npix64.h5'
    data_fname = 'QIML_flowers_data_n10000_npix32.h5'
    # data_fname = 'QIML_mnist_data_n10_npix32.h5'

    data = QIDataModule(data_fname, batch_size=100, num_workers=0, nbar=1e3, nframes=1000, flat_background=0., corr_matrix=True, shuffle=True)

    data.setup()
    batch = next(iter(data.train_dataloader()))[0]
    out = model(batch)[0]
    print(out.shape)
    # z, _, out = get_encoded_size(data, model) # to ensure frame dimension is compressed to 1
    # print(z.shape)

    # raise RuntimeError

    name = 'flws600_nb%.0e_nf%.0e' % (data.data_kwargs['nbar'], data.data_kwargs['nframes'])

    logger = WandbLogger(
        # name=name,
        project="VTAE32pix",
        entity="aproppe",
        mode="offline",
        # mode="online",
        log_model=False,
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