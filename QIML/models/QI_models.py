from argparse import ArgumentParser
from QIML.models.ViT_models import *
from QIML.models.submodels import *

import torch
import random
from torch import nn
import pytorch_lightning as pl
import wandb

from QIML.visualization.AP_figs_funcs import *
from QIML.models.utils import BetaRateScheduler, SSIM, phase_loss


def common_parser():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--encoder_num_layers", type=int, default=3)
    parser.add_argument("--decoder_num_layers", type=int, default=1)
    parser.add_argument("--bidirectional", type=bool, default=False)
    parser.add_argument("--hidden_dim", type=int, default=8)
    parser.add_argument("--input_dropout", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    return parser


"""
Notes on style

1. Use capitals to denote what are meant to be tensors, excluding batches
2. Use `black` for code formatting
3. Use NumPy style docstrings
"""


class QIAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        metric=nn.MSELoss,
        plot_interval: int = 1000,
    ) -> None:
        super().__init__()
        self.encoder = None
        self.decoder = None
        self.recoder = nn.Identity()
        self.metric = metric()
        # self.save_hyperparameters("lr", "weight_decay", "metric", "plot_interval")
        self.save_hyperparameters()

    def encode(self, X: torch.Tensor):
        """Encodes input into latent space"""
        return self.encoder(X)

    def recode(self, X: torch.Tensor):
        """Perform transformations on the latent space"""
        return self.recoder(X)

    def decode(self, Z: torch.Tensor):
        """Decodes latent space into output"""
        return self.decoder(Z)

    def forward(self, X: torch.Tensor):
        Z = self.encode(X)
        R = self.recode(Z)
        D = self.decode(R)
        return D, Z

    def step(self, batch, batch_idx):
        X, Y = batch
        pred_Y, Z = self(X)
        recon = self.metric(Y, pred_Y)
        loss = recon
        log = {"recon": recon}
        return loss, log, X, Y, pred_Y

    def training_step(self, batch, batch_idx):
        loss, log, X, Y, pred_Y = self.step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log, X, Y, pred_Y = self.step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        if (
            self.current_epoch > 0
            and self.current_epoch % self.hparams.plot_interval == 0
            and self.epoch_plotted == False
        ):
            self.epoch_plotted = True  # don't plot again in this epoch
            with torch.no_grad():
                fig = self.plot_training_results(X, Y, pred_Y)
                log.update({"plot": fig})
                self.logger.experiment.log(log)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        return optimizer

    def on_train_epoch_end(self) -> None:
        self.epoch_plotted = False

    def plot_training_results(self, X, Y, pred_Y):
        X = X.cpu()
        Y = Y.cpu()
        pred_Y = pred_Y.cpu()

        if X.ndim == 4:  # frames x Xpix x Ypix
            fig, ax = plt.subplots(ncols=3, nrows=1, dpi=150, figsize=(5, 2.5))
            idx = random.randint(0, Y.shape[0] - 1)
            frame_idx = random.randint(0, X.shape[1] - 1)
            ax[0].imshow(X[idx, frame_idx, :, :])
            ax[0].set_title("Input")
            ax[1].imshow(pred_Y[idx, :, :])
            ax[1].set_title("Prediction")
            ax[2].imshow(Y[idx, :, :])
            ax[2].set_title("Truth")
            dress_fig(tight=True, xlabel="x pixels", ylabel="y pixels", legend=False)
        elif X.ndim == 3:  # correlation matrix
            if pred_Y.ndim == 4:
                pred_Y = pred_Y.squeeze(1)
            fig, ax = plt.subplots(ncols=2, nrows=1, dpi=150, figsize=(5, 2.5))
            idx = random.randint(0, Y.shape[0] - 1)
            ax[0].imshow(pred_Y[idx, :, :])
            ax[0].set_title("Prediction")
            ax[1].imshow(Y[idx, :, :])
            ax[1].set_title("Truth")
            dress_fig(tight=True, xlabel="x pixels", ylabel="y pixels", legend=False)

        wandb.Image(plt)
        plt.close()

        return fig

    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("AutoEncoder")
        group.add_argument("--lr", type=float, default=1e-3)
        group.add_argument("--weight_decay", type=float, default=0.0)
        return parent_parser

    def initialize_lazy(self, input_shape):
        # if self.has_lazy_modules:
        with torch.no_grad():
            dummy = torch.ones(*input_shape)
            _ = self(dummy)  # this initializes the shapes

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0.0)

    # def initialize_weights(self, μ=0, σ=0.1):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, mean=μ, std=σ)
    #             nn.init.normal_(m.bias, mean=μ, std=σ)

    def count_parameters(self):
        return sum(param.numel() for param in self.parameters())


class SRN2D(QIAutoEncoder):
    """Symmetric Resnet 2D-to-2D Convolutional Autoencoder"""

    def __init__(
        self,
        depth: int = 4,
        first_layer_args={"kernel": (7, 7), "stride": (2, 2), "padding": (3, 3)},
        channels: list = [1, 4, 8, 16, 32, 64],
        strides: list = [2, 2, 2, 1, 2, 1],
        layers: list = [1, 1, 1, 1, 1],
        fwd_skip: bool = False,
        sym_skip: bool = True,
        dropout: float = [0.0, 0.0, 0.0, 0.0, 0.0],
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        plot_interval=50,
    ) -> None:
        super().__init__(lr, weight_decay, plot_interval)

        enc_activation = nn.ReLU
        dec_activation = nn.ReLU

        self.encoder = ResNet2D(
            block=ResBlock2d,
            first_layer_args=first_layer_args,
            depth=depth,
            channels=channels[0 : depth + 1],
            strides=strides[0:depth],
            layers=layers[0:depth],
            dropout=dropout[0:depth],
            activation=enc_activation,
            residual=fwd_skip,
        )

        last_layer_args = {
            "kernel": first_layer_args["kernel"],
            "stride": tuple(
                np.sqrt(np.array(first_layer_args["stride"])).astype(int)
            ),  # quadratically smaller stride than encoder
            "padding": first_layer_args["padding"],
        }

        self.decoder = DeconvNet2D(
            block=DeconvBlock2d,
            last_layer_args=last_layer_args,
            depth=depth,
            channels=list(reversed(channels[0 : depth + 1])),
            strides=list(
                reversed(np.sqrt(strides[0:depth]).astype(int))
            ),  # quadratically smaller stride than encoder
            layers=list(reversed(layers[0:depth])),
            activation=dec_activation,
            residual=sym_skip,
        )

    def forward(self, X: torch.Tensor):
        if X.ndim < 4:
            X = X.unsqueeze(1)  # adds the channel dimension
        Z, res = self.encoder(X)
        D = self.decoder(Z, res)
        D = D.squeeze(1)  # removes the channel dimension
        return D, Z


class SRN3D(QIAutoEncoder):
    """Symmetric Resnet 3D-to-2D Convolutional Autoencoder"""

    def __init__(
        self,
        depth: int = 6,
        first_layer_args={
            "kernel": (7, 7, 7),
            "stride": (2, 2, 2),
            "padding": (3, 3, 3),
        },
        channels: list = [1, 4, 8, 16, 32, 64],
        pixel_strides: list = [2, 2, 1, 1, 1, 1, 1, 1, 1],
        frame_strides: list = [2, 2, 2, 2, 2, 1, 1, 1, 1],
        layers: list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        fwd_skip: bool = True,
        sym_skip: bool = True,
        dropout: float = 0.0,
        activation=nn.ReLU,
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        metric=nn.MSELoss,
        ssim=None,
        ssim_weight=1.0,
        plot_interval: int = 50,
    ) -> None:
        super().__init__(lr, weight_decay, metric, plot_interval)

        if isinstance(channels, int):
            channels = np.repeat(channels, depth + 1)
            channels[0] = 1
            channels = list(channels)

        if isinstance(dropout, int):
            dropout = float(dropout)

        if isinstance(dropout, float):
            dropout = np.repeat(dropout, depth)
            dropout = list(dropout)

        self.encoder = ResNet3D_original(
            block=ResBlock3d,
            first_layer_args=first_layer_args,
            depth=depth,
            channels=channels[0 : depth + 1],
            pixel_strides=pixel_strides[0:depth],
            frame_strides=frame_strides[0:depth],
            layers=layers[0:depth],
            dropout=dropout[0:depth],
            activation=activation,
            residual=fwd_skip,
        )

        # Remove first frame dimension from
        last_layer_args = dict((k, v[1:]) for k, v in first_layer_args.items())

        self.decoder = DeconvNet2D(
            block=DeconvBlock2d,
            last_layer_args=last_layer_args,
            depth=depth,
            channels=list(reversed(channels[0 : depth + 1])),
            strides=list(reversed(pixel_strides[0:depth])),
            layers=list(reversed(layers[0:depth])),
            activation=activation,
            residual=sym_skip,
        )

        # Perception loss and β scheduler
        self.ssim = None if ssim is None else SSIM()

        # beta_scheduler_kwargs = {
        #     'initial_beta': 0.0,
        #     'end_beta': 0.1,
        #     'cap_steps': 2000,
        #     'hold_steps': 50,
        # }
        # self.beta_scheduler = BetaRateScheduler(**beta_scheduler_kwargs)
        # self.beta_scheduler.reset()

        self.save_hyperparameters()

    def forward(self, X: torch.Tensor):
        if X.ndim < 5:
            X = X.unsqueeze(1)  # adds the channel dimension
        Z, res = self.encoder(X)
        if Z.shape[2] > 1:
            print("Latent shape needs to be compressed down to 1")
            raise RuntimeError
        D = self.decoder(Z.squeeze(2), res)
        D = D.squeeze(1)  # removes the channel dimension
        return D, Z

    def step(self, batch, batch_idx):
        X, Y = batch
        pred_Y, Z = self(X)
        recon = self.metric(pred_Y, Y)
        if self.ssim is not None:
            ssim = 1 - self.ssim(pred_Y.unsqueeze(1), Y.unsqueeze(1))
            loss = recon + self.hparams.ssim_weight * ssim
            log = {"recon": recon, "ssim": ssim}
        else:
            loss = recon
            log = {"recon": recon}

        # Y = torch.cos(2*torch.pi*Y)
        # recon = phase_loss(pred_Y, Y)

        # if self.perceptual_loss is not None:
        #     percep = self.perceptual_loss(Y.unsqueeze(1), pred_Y.unsqueeze(1))
        #     β = next(self.beta_scheduler.beta())
        #     percep *= β
        #     loss = recon + percep
        #     log = {"recon": recon, "percep": percep, "beta": β}
        # else:
        #     loss = recon
        #     log = {"recon": recon}

        return loss, log, X, Y, pred_Y


class SRN3Dv2(QIAutoEncoder):
    """
    Symmetric Resnet 3D-to-2D Convolutional Autoencoder version 2
    - This variation has a different final layer, meant to avert the checkboard artifacts from Conv2DTranspose
    - Added AttentionBlock to the latent space
    - OR, without attention, take the mean of the 3D encoder latent space so that an arbitrary number of frames can be used
    """

    def __init__(
        self,
        input_shape=(2, 64, 64, 64),
        depth: int = 6,
        first_layer_args={
            "kernel": (7, 7, 7),
            "stride": (2, 2, 2),
            "padding": (3, 3, 3),
        },
        final_deconv_kernel: int = 4,
        channels: list = [1, 4, 8, 16, 32, 64],
        pixel_strides: list = [2, 2, 1, 1, 1, 1, 1, 1, 1],
        frame_strides: list = [2, 2, 2, 2, 2, 1, 1, 1, 1],
        fwd_skip: bool = True,
        sym_skip: bool = True,
        dropout: float = 0.0,
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        attention_on: bool = False,
        attention_dim: int = 64,
        metric=nn.MSELoss,
        perceptual_loss=None,
        plot_interval: int = 5,
    ) -> None:
        super().__init__(lr, weight_decay, metric, plot_interval)

        layers = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        if isinstance(channels, int):
            channels = np.repeat(channels, depth + 1)
            channels[0] = 1
            channels = list(channels)

        if isinstance(dropout, int):
            dropout = float(dropout)

        if isinstance(dropout, float):
            dropout = np.repeat(dropout, depth)
            dropout = list(dropout)

        self.encoder = ResNet3D(
            block=ResBlock3d,
            first_layer_args=first_layer_args,
            depth=depth,
            channels=channels[0 : depth + 1],
            pixel_strides=pixel_strides[0:depth],
            frame_strides=frame_strides[0:depth],
            layers=layers[0:depth],
            dropout=dropout[0:depth],
            residual=fwd_skip,
        )

        # Remove first frame dimension from
        last_layer_args = dict((k, v[1:]) for k, v in first_layer_args.items())

        decode_depth = depth - int(np.log2(final_deconv_kernel))

        self.final_deconv = nn.ConvTranspose2d(
            in_channels=channels[depth - 2],
            out_channels=1,
            kernel_size=final_deconv_kernel,
            stride=final_deconv_kernel,
        )

        self.decoder = DeconvNet2D(
            block=DeconvBlock2d,
            last_layer_args=last_layer_args,
            depth=decode_depth,
            channels=list(reversed(channels[1 : depth + 1]))[0:decode_depth],
            strides=list(reversed(pixel_strides[0:depth]))[0:decode_depth],
            # depth=depth,
            # channels=list(reversed(channels[1:depth + 1])),
            # strides=list(reversed(pixel_strides[0:depth])),
            layers=list(reversed(layers[0:depth])),
            residual=sym_skip,
        )

        """ Attention """
        if attention_on:
            with torch.no_grad():
                dummy_input = torch.rand(input_shape)
                Z_shape = self.encoder(dummy_input.unsqueeze(1))[0].shape
            latent_frame_dim = Z_shape[2]
            latent_dim = torch.prod(torch.tensor(Z_shape[1:]))

            reshape_in = Reshape(-1, latent_dim)
            attn_in = nn.Linear(latent_dim, attention_dim)
            attn = AttentionBlock(
                attention_dim,
                depth=2,
                num_heads=2,
            )
            attn_out = nn.Linear(attention_dim, latent_dim // latent_frame_dim)
            reshape_out = Reshape(-1, Z_shape[1], 1, Z_shape[3], Z_shape[4])

            self.attention = nn.Sequential(
                reshape_in,
                attn_in,
                nn.ReLU(),
                attn,
                attn_out,
                nn.ReLU(),
                reshape_out,
            )

        else:
            self.attention = nn.Identity()

        """ Losses """
        # Perception loss and β scheduler
        # self.perceptual_loss = None if perceptual_loss is None else VGGPerceptualLoss()
        beta_scheduler_kwargs = {
            "initial_beta": 0.0,
            "end_beta": 0.1,
            "cap_steps": 2000,
            "hold_steps": 50,
        }
        self.beta_scheduler = BetaRateScheduler(**beta_scheduler_kwargs)
        self.beta_scheduler.reset()

        self.save_hyperparameters()

    def forward(self, X: torch.Tensor):
        if X.ndim < 5:
            X = X.unsqueeze(1)  # adds the channel dimension
        Z, res = self.encoder(X)
        Z = self.attention(Z)
        # Z = Z.mean(dim=2, keepdim=True)  # take the mean of the latent space
        Z = self.decoder(Z.squeeze(2), res)
        Z = self.final_deconv(Z)
        Z = Z.squeeze(1)  # removes the channel dimension
        return Z, 1

    def step(self, batch, batch_idx):
        X, Y = batch
        pred_Y, Z = self(X)
        # recon = self.metric(Y, pred_Y)
        recon = phase_loss(pred_Y, Y)
        if self.perceptual_loss is not None:
            percep = self.perceptual_loss(Y.unsqueeze(1), pred_Y.unsqueeze(1))
            β = next(self.beta_scheduler.beta())
            percep *= β
            loss = recon + percep
            log = {"recon": recon, "percep": percep, "beta": β}
        else:
            loss = recon
            log = {"recon": recon}
        return loss, log, X, Y, pred_Y


class SRN3D_v3(QIAutoEncoder):
    """
    Symmetric Resnet 3D-to-2D Convolutional Autoencoder
    - In the previous model, the decoder only took symmetric residuals from the encoder side, and did not use dropout
    - This updated version includes both 'forward' and 'symmetric' residuals, and includes dropout
    - Essentially, the decoder class ResNet2DT now uses ResBlock2dT, which more closely mimic their ResBlock3d counterparts
    - Dropout is now the same for every layer at each depth (given as a single float and not a list of floats)
    """

    def __init__(
        self,
        depth: int = 6,
        channels: list = [1, 4, 8, 16, 32, 64],
        pixel_kernels: tuple = (3, 3),
        frame_kernels: tuple = (3, 3),
        pixel_downsample: int = 4,
        frame_downsample: int = 32,
        layers: list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        dropout: float = 0.0,
        activation="ReLU",
        norm=True,
        fwd_skip: bool = True,
        sym_skip: bool = True,
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        metric=nn.MSELoss,
        ssim_weight=1.0,
        window_size=15,
        plot_interval: int = 5,
    ) -> None:
        super().__init__(lr, weight_decay, metric, plot_interval)

        self.ssim = SSIM(window_size=window_size)
        self.ssim_weight = ssim_weight
        try:
            activation = getattr(nn, activation)
        except:
            activation = activation

        channels = [1] + [channels] * depth if isinstance(channels, int) else channels

        # Automatically calculate the strides for each layer
        pixel_strides = [
            2 if i < int(np.log2(pixel_downsample)) else 1 for i in range(depth)
        ]
        frame_strides = [
            2 if i < int(np.log2(frame_downsample)) else 1 for i in range(depth)
        ]

        # And automatically fill the kernel sizes
        pixel_kernels = [
            pixel_kernels[0] if i == 0 else pixel_kernels[1] for i in range(depth)
        ]
        frame_kernels = [
            frame_kernels[0] if i == 0 else frame_kernels[1] for i in range(depth)
        ]

        self.encoder = ResNet3D(
            block=ResBlock3d,
            depth=depth,
            channels=channels[0 : depth + 1],
            pixel_kernels=pixel_kernels,
            frame_kernels=frame_kernels,
            pixel_strides=pixel_strides,
            frame_strides=frame_strides,
            layers=layers[0:depth],
            dropout=dropout,
            activation=activation,
            norm=norm,
            residual=fwd_skip,
        )

        self.decoder = ResNet2DT(
            block=ResBlock2dT,
            depth=depth,
            channels=list(reversed(channels[0 : depth + 1])),
            kernels=list(reversed(pixel_kernels)),
            strides=list(reversed(pixel_strides)),
            layers=list(reversed(layers[0:depth])),
            dropout=dropout,
            activation=activation,
            norm=norm,
            sym_residual=sym_skip,
            fwd_residual=fwd_skip,
        )

        self.save_hyperparameters()

    def forward(self, X: torch.Tensor):
        X = X.unsqueeze(1) if X.ndim < 5 else X
        Z, res = self.encoder(X)
        D = self.decoder(Z.sum(dim=2), res).squeeze(1)
        return D, 1

    def step(self, batch, batch_idx):
        X, Y = batch
        pred_Y, _ = self(X)
        recon = self.metric(pred_Y, Y)  # pixel-wise recon loss
        ssim = 1 - self.ssim(pred_Y.unsqueeze(1), Y.unsqueeze(1))  # SSIM loss
        loss = recon + self.ssim_weight * ssim
        log = (
            {"recon": recon, "ssim": ssim}
            if self.ssim is not None
            else {"recon": recon}
        )

        return loss, log, X, Y, pred_Y


class PRAUN(QIAutoEncoder):
    """
    """

    def __init__(
        self,
        depth: int = 6,
        channels: list = [1, 4, 8, 16, 32, 64],
        pixel_kernels: tuple = (3, 3),
        frame_kernels: tuple = (3, 3),
        pixel_downsample: int = 4,
        frame_downsample: int = 32,
        layers: list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        attn_on: list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        attn_heads: int = 2,
        attn_depth: int = 2,
        dropout: float = 0.0,
        activation="ReLU",
        norm=True,
        fwd_skip: bool = True,
        sym_skip: bool = True,
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        metric=nn.MSELoss,
        ssim_weight=1.0,
        window_size=15,
        plot_interval: int = 5,
    ) -> None:
        super().__init__(lr, weight_decay, metric, plot_interval)

        self.ssim = SSIM(window_size=window_size)
        self.ssim_weight = ssim_weight
        try:
            activation = getattr(nn, activation)
        except:
            activation = activation

        # Automatically generate the channel list if give only a single int
        channels = [1] + [channels] * depth if isinstance(channels, int) else channels

        # Automatically calculate the strides for each layer
        pixel_strides = [
            2 if i < int(np.log2(pixel_downsample)) else 1 for i in range(depth)
        ]
        frame_strides = [
            2 if i < int(np.log2(frame_downsample)) else 1 for i in range(depth)
        ]

        # And automatically fill the kernel sizes
        pixel_kernels = [
            pixel_kernels[0] if i == 0 else pixel_kernels[1] for i in range(depth)
        ]
        frame_kernels = [
            frame_kernels[0] if i == 0 else frame_kernels[1] for i in range(depth)
        ]

        self.encoder = AttnResNet3D(
            depth=depth,
            channels=channels[0 : depth + 1],
            pixel_kernels=pixel_kernels,
            frame_kernels=frame_kernels,
            pixel_strides=pixel_strides,
            frame_strides=frame_strides,
            attn_on=attn_on[0:depth],
            attn_heads=attn_heads,
            attn_depth=attn_depth,
            dropout=dropout,
            activation=activation,
            norm=norm,
            residual=fwd_skip,
        )

        self.decoder = ResNet2DT(
            block=ResBlock2dT,
            depth=depth,
            channels=list(reversed(channels[0 : depth + 1])),
            kernels=list(reversed(pixel_kernels)),
            strides=list(reversed(pixel_strides)),
            layers=list(reversed(layers[0:depth])),
            dropout=dropout,
            activation=activation,
            norm=norm,
            sym_residual=sym_skip,
            fwd_residual=fwd_skip,
        )

        self.save_hyperparameters()

    def forward(self, X: torch.Tensor):
        X = X.unsqueeze(1) if X.ndim < 5 else X
        Z, res = self.encoder(X)
        D = self.decoder(Z.sum(dim=2), res).squeeze(1)
        return D, 1

    def step(self, batch, batch_idx):
        X, Y = batch
        pred_Y, _ = self(X)
        recon = self.metric(pred_Y, Y)  # pixel-wise recon loss
        ssim = 1 - self.ssim(pred_Y.unsqueeze(1), Y.unsqueeze(1))  # SSIM loss
        loss = recon + self.ssim_weight * ssim
        log = (
            {"recon": recon, "ssim": ssim}
            if self.ssim is not None
            else {"recon": recon}
        )

        return loss, log, X, Y, pred_Y


class PRAUNet(QIAutoEncoder):
    """
    Uses a different 3D attention block
    """
    def __init__(
        self,
        input_shape,
        depth,
        channels,
        pixel_kernels,
        frame_kernels,
        pixel_downsample,
        frame_downsample,
        layers: list,
        attn_on: list,
        attn_args: dict,
        dropout: float = 0.0,
        activation="ReLU",
        norm=True,
        fwd_skip: bool = True,
        sym_skip: bool = True,
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        metric=nn.MSELoss,
        ssim_weight=1.0,
        window_size=15,
        plot_interval: int = 5,
    ) -> None:
        super().__init__(lr, weight_decay, metric, plot_interval)

        self.ssim = SSIM(window_size=window_size)
        self.ssim_weight = ssim_weight
        try:
            activation = getattr(nn, activation)
        except:
            activation = activation

        # Automatically generate the channel list if give only a single int
        channels = [1] + [channels] * depth if isinstance(channels, int) else channels

        layers = layers * depth if (len(layers) == 1) else layers

        # Automatically calculate the strides for each layer
        pixel_strides = [
            2 if i < int(np.log2(pixel_downsample)) else 1 for i in range(depth)
        ]
        frame_strides = [
            2 if i < int(np.log2(frame_downsample)) else 1 for i in range(depth)
        ]

        # And automatically fill the kernel sizes
        pixel_kernels = [
            pixel_kernels[0] if i == 0 else pixel_kernels[1] for i in range(depth)
        ]
        frame_kernels = [
            frame_kernels[0] if i == 0 else frame_kernels[1] for i in range(depth)
        ]

        self.encoder = AttentionResNet3D(
            input_shape=input_shape,
            depth=depth,
            channels=channels[0 : depth + 1],
            pixel_kernels=pixel_kernels,
            frame_kernels=frame_kernels,
            pixel_strides=pixel_strides,
            frame_strides=frame_strides,
            attn_on=attn_on[0:depth],
            attn_args=attn_args,
            dropout=dropout,
            activation=activation,
            norm=norm,
            residual=fwd_skip,
        )

        self.decoder = ResNet2DT(
            block=ResBlock2dT,
            depth=depth,
            channels=list(reversed(channels[0 : depth + 1])),
            kernels=list(reversed(pixel_kernels)),
            strides=list(reversed(pixel_strides)),
            layers=list(reversed(layers[0:depth])),
            dropout=dropout,
            activation=activation,
            norm=norm,
            sym_residual=sym_skip,
            fwd_residual=fwd_skip,
        )

        self.save_hyperparameters()

    def forward(self, X: torch.Tensor):
        X = X.unsqueeze(1) if X.ndim < 5 else X
        Z, res = self.encoder(X)
        D = self.decoder(Z.sum(dim=2), res).squeeze(1)
        return D, 1

    def step(self, batch, batch_idx):
        X, Y = batch
        pred_Y, _ = self(X)
        recon = self.metric(pred_Y, Y)  # pixel-wise recon loss
        ssim = 1 - self.ssim(pred_Y.unsqueeze(1), Y.unsqueeze(1))  # SSIM loss
        loss = recon + self.ssim_weight * ssim
        log = (
            {"recon": recon, "ssim": ssim}
            if self.ssim is not None
            else {"recon": recon}
        )

        return loss, log, X, Y, pred_Y

class MultiScaleCNN(pl.LightningModule):
    def __init__(
        self,
        first_layer_args={"kernel_size": (7, 7), "stride": (2, 2), "padding": (3, 3)},
        nbranch: int = 3,
        branch_depth: int = 1,
        kernels: list = [3, 5, 7],
        channels: list = [4, 8, 16, 32, 64],
        strides: list = [2, 2, 2, 2, 2, 2],
        dilations: list = [1, 1, 1, 1, 1, 1],
        activation: nn.Module = nn.ReLU,
        dropout: float = 0.1,
        residual: bool = True,
        fourier: bool = False,
    ) -> None:
        super(MultiScaleCNN, self).__init__()

        ch0 = 2 if fourier else 1

        # First convolutional layer
        self.conv1 = nn.Conv2d(ch0, channels[0], **first_layer_args)
        self.actv1 = activation()

        self.branches = nn.ModuleList([])
        for i in range(0, nbranch):
            self.inchannels = channels[0]
            branch_layers = self._make_branch(
                branch_depth,
                channels,
                kernels[i],
                strides,
                dilations[i],
                activation,
                dropout,
                residual,
            )
            self.branches.append(branch_layers)

        # Final convolutional layer for concatenated branch outputs
        self.conv3 = nn.Conv2d(
            nbranch
            * channels[
                branch_depth - 1
            ],  # number of channels in concatenated branch outputs
            channels[branch_depth],
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def _make_branch(
        self,
        branch_depth,
        channels,
        kernel,
        strides,
        dilation,
        activation,
        dropout,
        residual,
    ):
        layers = []
        for i in range(0, branch_depth):
            layers.append(
                self._make_layer(
                    channels[i],
                    kernel=kernel,
                    stride=strides[i],
                    dilation=dilation,
                    activation=activation,
                    dropout=dropout,
                    residual=residual,
                )
            )
        return nn.Sequential(*layers)

    def _make_layer(
        self, channels, kernel, stride, dilation, activation, dropout, residual
    ):
        """Modified from Nourman (https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/)"""
        downsample = None
        if stride != 1 or self.inchannels != channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.inchannels, channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(channels),
            )
        layer = ResBlock2D(
            self.inchannels,
            channels,
            kernel,
            stride,
            dilation,
            downsample,
            activation,
            dropout=dropout,
            residual=residual,
        )
        self.inchannels = channels
        return layer

    def forward(self, x):
        # Add channel dimension
        if x.ndim < 4:
            x = x.unsqueeze(1)

        # Pass input through the first convolutional layer
        x = self.conv1(x)
        x = self.actv1(x)

        # Pass input through the multi-scale convolutional layers
        branch_x = []
        for i in range(0, len(self.branches)):
            branch_x.append(self.branches[i](x)[0])

        # Concatenate branch outputs
        x = torch.cat(branch_x, dim=1)
        # x = self.actv1(x)

        # Pass input through the final convolutional layer
        x = self.conv3(x)
        x = self.actv1(x)

        return x


class MSRN2D(QIAutoEncoder):
    def __init__(
        self,
        encoder_args,
        decoder_args,
        z_size: int = 64,
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        plot_interval=50,
        metric=nn.MSELoss,
        init_lazy: bool = False,  # Set to false when testing encoded and decoded shapes; true for training
        input_shape: tuple = (2, 1, 1024, 1024),
    ) -> None:
        super().__init__(lr, weight_decay, metric, plot_interval)

        self.encoder = MultiScaleCNN(**encoder_args)
        # self.decoder = UpsampleConvStack(**decoder_args) # Interpolate upsample + conv decoder
        self.decoder = DeconvolutionNetwork(**decoder_args)  # Conv2DTranspose decoder

        ## For a flattened bottleneck:
        # self.flatten = nn.Flatten()
        # self.linear1 = nn.LazyLinear(z_size)
        # self.reshape = Reshape(-1, 1, int(np.sqrt(z_size)), int(np.sqrt(z_size)))

        if init_lazy:
            self.initialize_lazy(input_shape)

        self.save_hyperparameters()

    def forward(self, X: torch.Tensor):
        Z = self.encode(X)
        # Z = self.flatten(Z)
        # Z = self.linear1(Z)
        # Z = self.reshape(Z)
        del X  # helps with memory allocation
        return self.decoder(Z), 1  # return a dummy Z, reduce memory load


class VTAE(QIAutoEncoder):
    """Vision Transformer Encoder, Deconvolutional Decoder"""

    def __init__(
        self,
        transformer_args,
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        metric=nn.MSELoss,
        plot_interval: int = 50,
    ) -> None:
        super().__init__(lr, weight_decay, metric, plot_interval)

        self.transformer = VisionTransformerAutoencoder(**transformer_args)
        self._init_weights()

        self.save_hyperparameters()

    def forward(self, X: torch.Tensor):
        return self.transformer(X), 1  # return a dummy Z

    def step(self, batch, batch_idx):
        X, Y = batch
        pred_Y, Z = self(X)
        recon = self.metric(Y, pred_Y)
        loss = recon
        log = {"recon": recon}
        return loss, log, X, Y, pred_Y


class TransformerAutoencoder(QIAutoEncoder):
    """Vision Transformer Encoder, Deconvolutional Decoder"""

    def __init__(
        self,
        input_dim=1024,
        output_dim=32,
        patch_dim=32,
        hidden_dim=16,
        num_heads=2,
        num_layers=2,
        dropout=0.1,
        decoder="Deconv",
        downsample_latent=1,
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        metric=nn.MSELoss,
        plot_interval: int = 50,
    ) -> None:
        super().__init__(lr, weight_decay, metric, plot_interval)

        self.encoder = VisTransformerEncoder2D(
            input_dim, output_dim, patch_dim, hidden_dim, num_heads, num_layers, dropout
        )

        # if downsample_latent != 1:
        linear1 = nn.Linear(hidden_dim, hidden_dim // downsample_latent)
        hidden_dim = hidden_dim // downsample_latent
        actv1 = nn.ReLU()
        self.downsample = nn.Sequential(linear1, actv1)
        # else:
        #     self.downsample = nn.Identity()

        self.decoder_type = decoder
        if self.decoder_type == "Deconv":
            conv_channels = input_dim // patch_dim
            reshape = Reshape(
                -1,
                conv_channels,
                int(hidden_dim ** (1 / 2)),
                int(hidden_dim ** (1 / 2)),
            )

            deconv = DeconvNet(
                depth=3,
                # channels=[conv_channels, conv_channels//2, conv_channels//4, conv_channels//8, conv_channels//16],
                channels=[
                    conv_channels,
                    conv_channels,
                    conv_channels,
                    conv_channels,
                    conv_channels,
                ],
                size_ratio=int(output_dim / int(hidden_dim ** (1 / 2))),
                last_layer_args={"kernel": 4, "stride": 4},
            )

            self.decoder = nn.Sequential(reshape, deconv)

        elif self.decoder_type == "MLP":
            flatten = nn.Flatten()
            MLP_depth = 0
            MLP_dim = 1024
            MLP_layers = []
            for i in range(0, MLP_depth - 1):
                MLP_layers.append(nn.LazyLinear(MLP_dim))
            linear_out = nn.LazyLinear(output_dim**2)
            reshape = Reshape(-1, output_dim, output_dim)
            self.decoder = nn.Sequential(flatten, *MLP_layers, linear_out, reshape)

        self.initialize_lazy((2, input_dim, input_dim))
        # self._init_weights()
        self.save_hyperparameters()

    def forward(self, X: torch.Tensor):
        X = self.encode(X)
        X = self.downsample(X)
        X = self.decoder(X)
        return X, 1


class TransformerAutoencoder3D(QIAutoEncoder):
    """Vision Transformer Encoder, Deconvolutional Decoder"""

    def __init__(
        self,
        nframe: int,
        input_dim: int,
        hidden_dim: int = 100,
        patch_dim: int = 4,
        deconv_dim: int = 4,
        deconv_depth: int = 3,
        num_heads: int = 4,
        num_layers: int = 6,
        dropout: float = 0.0,
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        metric=nn.MSELoss,
        plot_interval: int = 50,
    ) -> None:
        super().__init__(lr, weight_decay, metric, plot_interval)

        channels = input_dim // patch_dim

        self.encoder = VisTransformerEncoder3D(
            nframe=nframe,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            patch_dim=patch_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.recoder = nn.Sequential(
            nn.Linear(hidden_dim, deconv_dim**2),
            Reshape(-1, channels, deconv_dim, deconv_dim),
        )

        self.decoder = DeconvNet(
            size_ratio=(input_dim // deconv_dim),
            channels=[channels, channels, channels, channels, channels],
            depth=deconv_depth,
            last_layer_args={"kernel": 4, "stride": 4},
        )

        self._init_weights()
        self.save_hyperparameters()


class MLPAutoencoder(QIAutoEncoder):
    def __init__(
        self,
        input_dim=1024,
        output_dim=32,
        hidden_dim=16,
        depth=2,
        dropout=0.1,
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        metric=nn.MSELoss,
        plot_interval: int = 50,
    ) -> None:
        super().__init__(lr, weight_decay, metric, plot_interval)

        self.flatten = nn.Flatten()

        layers = []
        for i in range(0, depth - 1):
            layers.append(nn.LazyLinear(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LazyBatchNorm1d())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.LazyLinear(output_dim**2))

        self.MLP = nn.Sequential(*layers)
        self.reshape = Reshape(-1, output_dim, output_dim)

        self.initialize_lazy((2, input_dim, input_dim))
        self._init_weights()
        self.save_hyperparameters()

    def forward(self, X: torch.Tensor):
        X = self.flatten(X)
        X = self.MLP(X)
        X = self.reshape(X)
        return X, 1


""" For testing """
if __name__ == "__main__":
    from QIML.pipeline.QI_data import QIDataModule

    #
    # data_fname = "flowers_n5000_npix64.h5"
    # # data_fname = 'flowers_n600_npix32.h5'
    # data = QIDataModule(
    #     data_fname,
    #     batch_size=10,
    #     num_workers=0,
    #     nbar_signal=(1e3, 1e4),
    #     nbar_bkgrnd=(1e3, 1e4),
    #     nframes=32,
    #     shuffle=True,
    # )
    # data.setup()
    # batch = next(iter(data.train_dataloader()))
    # X = batch[0]
    X = torch.rand((2, 1, 32, 64, 64))
    # raise RuntimeError

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

    model = PRAUNet(
        input_shape=X.shape,
        depth=4,
        channels=32,
        pixel_kernels=(5, 3),
        frame_kernels=(5, 3),
        pixel_downsample=4,
        frame_downsample=32,
        layers=[1],
        attn_on=[1, 1, 0, 0, 0, 0, 0, 0],
        attn_args=attn_args,
        dropout=0.0,
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

    Y, Z = model(X)
    print(Y.shape)
