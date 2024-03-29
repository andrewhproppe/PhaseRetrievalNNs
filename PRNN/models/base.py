import random
import pytorch_lightning as pl
import wandb

from PRNN.visualization.figure_utils import *
from PRNN.models.utils import SSIM, GradientDifferenceLoss
from PRNN.models.ViT_models import *
from PRNN.models.submodels import *
from math import log

class AutoEncoder(pl.LightningModule):
    """
    Base autoencoder model. encoder and decoder are assigned in the child class. recoder is usually just nn.Identity(),
    but can be used to perform transformations on the latent space.
    """
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
        pred_Y, _ = self(X)
        recon = self.metric(pred_Y, Y)  # pixel-wise recon loss
        ssim = 1 - self.ssim(pred_Y.unsqueeze(1), Y.unsqueeze(1))  # SSIM loss
        gdl = self.gdl(pred_Y.unsqueeze(1), Y.unsqueeze(1))  # gradient difference loss
        loss = recon + self.ssim_weight * ssim + self.gdl_weight * gdl
        log = (
            {"recon": recon, "ssim": ssim, "gdl": gdl}
            if self.ssim is not None
            else {"recon": recon}
        )

        return loss, log, X, Y, pred_Y

    def training_step(self, batch, batch_idx):
        loss, log, X, Y, pred_Y = self.step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, log, X, Y, pred_Y = self.step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        if (
            self.current_epoch > 0
            and self.current_epoch % self.hparams.plot_interval == 0
            and self.epoch_plotted == False
        ):
            self.epoch_plotted = True  # don't plot again in this epoch
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

    def count_parameters(self):
        return sum(param.numel() for param in self.parameters())


class PRUNe(AutoEncoder):
    """
    Phase Retrieval U-Net (PRUNe). 3D ResNet encoder, 2D ResNet decoder
    """
    def __init__(
        self,
        depth: int = 6,
        channels: list = [1, 4, 8, 16, 32, 64],
        pixel_kernels: tuple = (5, 3),
        frame_kernels: tuple = (5, 3),
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
        gdl_weight=1.0,
        window_size=15,
        plot_interval: int = 5,
    ) -> None:
        super().__init__(lr, weight_decay, metric, plot_interval)

        self.ssim = SSIM(window_size=window_size)
        self.ssim_weight = ssim_weight
        self.gdl = GradientDifferenceLoss()
        self.gdl_weight = gdl_weight

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

        self._init_weights()
        self.save_hyperparameters()

    def forward(self, X: torch.Tensor):
        X = X.unsqueeze(1) if X.ndim < 5 else X
        Z, res = self.encoder(X)
        D = self.decoder(Z.sum(dim=2), res).squeeze(1)
        return D, 1


class PRUNe2D(AutoEncoder):
    """
    Variantion of PRUNe that encodes the 3D ensemble of images but treats the nframes instead as convolutional channels
    instead of performing 3D convolutions over the frame, x, and y dimensions
    """
    def __init__(
        self,
        depth: int = 6,
        img_size: int = 32,
        channels: list = [1, 4, 8, 16, 32, 64],
        pixel_kernels: tuple = (3, 3),
        pixel_downsample: int = 4,
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
        gdl_weight=1.0,
        window_size=15,
        plot_interval: int = 5,
    ) -> None:
        super().__init__(lr, weight_decay, metric, plot_interval)

        self.ssim = SSIM(window_size=window_size)
        self.ssim_weight = ssim_weight
        self.gdl = GradientDifferenceLoss()
        self.gdl_weight = gdl_weight

        try:
            activation = getattr(nn, activation)
        except:
            activation = activation

        channels = [1] + [channels] * depth if isinstance(channels, int) else channels
        encoder_channels = channels
        decoder_channels = list(reversed(channels[0:depth+1]))
        encoder_channels[0] = 32

        # Automatically calculate the strides for each layer
        pixel_strides = [
            2 if i < int(np.log2(pixel_downsample)) else 1 for i in range(depth)
        ]

        # And automatically fill the kernel sizes
        pixel_kernels = [
            pixel_kernels[0] if i == 0 else pixel_kernels[1] for i in range(depth)
        ]

        self.encoder = ResNet2D(
            block=ResBlock2d,
            depth=depth,
            channels=encoder_channels,
            pixel_kernels=pixel_kernels,
            pixel_strides=pixel_strides,
            layers=layers[0:depth],
            dropout=dropout,
            activation=activation,
            norm=norm,
            residual=fwd_skip,
        )

        self.decoder = ResNet2DT(
            block=ResBlock2dT,
            depth=depth,
            channels=decoder_channels,
            kernels=list(reversed(pixel_kernels)),
            strides=list(reversed(pixel_strides)),
            layers=list(reversed(layers[0:depth])),
            dropout=dropout,
            activation=activation,
            norm=norm,
            sym_residual=sym_skip,
            fwd_residual=fwd_skip,
        )

        self._init_weights()
        self.save_hyperparameters()

    def forward(self, X: torch.Tensor):
        X = X.unsqueeze(1) if X.ndim < 4 else X
        Z, res = self.encoder(X)
        D = self.decoder(Z, res).squeeze(1)
        return D, 1


class PRUNe2Dc(AutoEncoder):
    """
    Performs convolutions on the correlation matrix of multiple frames, so converts an image of size
    npix^2 * ypix*2 to npix * ypix. The decoder strides are set as the square root of the encoder strides
    to scale the spatial dimension properly.
    """
    def __init__(
        self,
        depth: int = 6,
        img_size: int = 32,
        channels: list = [1, 4, 8, 16, 32, 64],
        pixel_kernels: tuple = (3, 3),
        pixel_downsample: int = 4,
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
        gdl_weight=1.0,
        window_size=15,
        plot_interval: int = 5,
    ) -> None:
        super().__init__(lr, weight_decay, metric, plot_interval)

        self.ssim = SSIM(window_size=window_size)
        self.ssim_weight = ssim_weight
        self.gdl = GradientDifferenceLoss()
        self.gdl_weight = gdl_weight

        try:
            activation = getattr(nn, activation)
        except:
            activation = activation

        channels = [1] + [channels] * depth if isinstance(channels, int) else channels

        # Automatically calculate the strides for each layer
        encoder_pixel_strides = [
            # 4 if i < int(np.log2(pixel_downsample)) else 1 for i in range(depth)
            4 if i < int(log(pixel_downsample, 4)) else 1 for i in range(depth)
        ]

        pixel_upsample = pixel_downsample // img_size
        decoder_pixel_strides = [
            2 if i < int(log(pixel_upsample, 2)) else 1 for i in range(depth)
        ]

        decoder_pixel_strides = list(reversed(decoder_pixel_strides))

        # And automatically fill the kernel sizes
        pixel_kernels = [
            pixel_kernels[0] if i == 0 else pixel_kernels[1] for i in range(depth)
        ]

        self.encoder = ResNet2D(
            block=ResBlock2d,
            depth=depth,
            channels=channels[0 : depth + 1],
            pixel_kernels=pixel_kernels,
            pixel_strides=encoder_pixel_strides,
            layers=layers[0:depth],
            dropout=dropout,
            activation=activation,
            norm=norm,
            residual=fwd_skip,
        )

        self.decoder = ResNet2DT(
            block=ResBlock2dT,
            depth=depth,
            channels=list(reversed(channels[0:depth+1])),
            kernels=list(reversed(pixel_kernels)),
            strides=decoder_pixel_strides,
            layers=list(reversed(layers[0:depth])),
            dropout=dropout,
            activation=activation,
            norm=norm,
            sym_residual=sym_skip,
            fwd_residual=fwd_skip,
        )

        self._init_weights()
        self.save_hyperparameters()

    def forward(self, X: torch.Tensor):
        X = X.unsqueeze(1) if X.ndim < 4 else X
        Z, res = self.encoder(X)
        D = self.decoder(Z, res).squeeze(1)
        return D, 1


class MSRN2D(AutoEncoder):
    """
    Multi-scale ResNet 2D-to-2D Autoencoder
    - Uses a multiscale convolutional encoder and a Conv2DTranpose decoder
    """
    def __init__(
        self,
        encoder_args,
        decoder_args,
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        plot_interval=50,
        metric=nn.MSELoss,
        mse_weight=1.0,
        ssim_weight=1.0,
        gdl_weight=1.0,
        init_lazy: bool = False,  # Set to false when testing encoded and decoded shapes; true for training
        input_shape: tuple = (2, 1, 1024, 1024),
    ) -> None:
        super().__init__(lr, weight_decay, metric, plot_interval)

        self.ssim = SSIM(window_size=11)
        self.ssim_weight = ssim_weight
        self.gdl = GradientDifferenceLoss()
        self.gdl_weight = gdl_weight

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


class VTAE(AutoEncoder):
    """
    - As implemented, showed poorer performance than SRNAE3D
    """
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


class TransformerAutoencoder(AutoEncoder):
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
        mse_weight=1.0,
        ssim_weight=1.0,
        gdl_weight=1.0,
        plot_interval: int = 50,
    ) -> None:
        super().__init__(lr, weight_decay, metric, plot_interval)

        self.ssim = SSIM(window_size=11)
        self.ssim_weight = ssim_weight
        self.gdl = GradientDifferenceLoss()
        self.gdl_weight = gdl_weight

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


class TransformerAutoencoder3D(AutoEncoder):
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
        mse_weight=1.0,
        ssim_weight=1.0,
        gdl_weight=1.0,
        plot_interval: int = 50,
    ) -> None:
        super().__init__(lr, weight_decay, metric, plot_interval)

        self.ssim = SSIM(window_size=11)
        self.ssim_weight = ssim_weight
        self.gdl = GradientDifferenceLoss()
        self.gdl_weight = gdl_weight

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


class MLPAutoencoder(AutoEncoder):
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