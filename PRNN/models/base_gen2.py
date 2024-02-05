from PRNN.utils import get_system_and_backend
get_system_and_backend()

import torch
import pytorch_lightning as pl
import wandb
import random
import torch.nn as nn
import torch.nn.functional as F

from PRNN.visualization.figure_utils import *
from PRNN.models.utils import SSIM, GradientDifferenceLoss, CircularMSELoss
from PRNN.models.cbam import CBAM, CBAM3D
from PRNN.models.submodels_gen2 import FramesToEigenvalues
from typing import Optional, Type

matplotlib.use('Agg')

# TODO: Implement a ResVANet3D model that uses the VAN block of OverlapPatchEmbed->Transformer->PatchMerging
# https://github.com/Visual-Attention-Network/VAN-Classification/blob/main/models/van.py

def plot_with_agg_backend(func):
    def wrapper(*args, **kwargs):
        original_backend = matplotlib.get_backend()
        matplotlib.use('Agg')
        # plt.switch_backend('Agg')
        func(*args, **kwargs)
        # plt.switch_backend(original_backend)
        matplotlib.use(original_backend)
    return wrapper


""" BLOCKS """
class AttnResBlock2d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel=(3, 3),
            stride=1,
            downsample=None,
            activation: Optional[Type[nn.Module]] = nn.ReLU,
            dropout=0.0,
            norm: bool = True,
            residual: bool = True,
            attn_on: bool = False,
            attn_depth: int = 1,
            attn_heads: int = 1,
    ) -> None:
        super().__init__()

        # Whether or not to activate ResNet block skip connections
        self.residual = residual
        self.residual_scale = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

        padding = tuple(k // 2 for k in kernel)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=not norm)
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.activation = nn.Identity() if activation is None else activation()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=1, padding=padding, bias=not norm)
        self.bn2 = nn.BatchNorm2d(out_channels) if norm else nn.Identity()

        # Add or skip attention layer based on the use_attention flag
        # self.attention = Attention3D(out_channels) if attn_on else nn.Identity()
        self.attention = CBAM(out_channels, 16) if attn_on else nn.Identity()

        self.downsample = downsample
        self.out_channels = out_channels

    def forward(self, x):
        x = x[0] if isinstance(x, tuple) else x  # get only x, ignore residual that is fed back into forward pass
        residual = self.downsample(x) if self.downsample else x

        out = self.conv1(x)
        out = self.dropout(out)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)

        residual = self.attention(residual)  # Apply attention if available

        if self.residual:  # forward skip connection
            out += residual*self.residual_scale

        out = self.activation(out)

        return out, residual


class AttnResBlock3d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel=(3, 3, 3),
            stride=1,
            downsample=None,
            activation: Optional[Type[nn.Module]] = nn.ReLU,
            dropout=0.0,
            norm: bool = True,
            residual: bool = True,
            attn_on: bool = False,
            attn_depth: int = 1,
            attn_heads: int = 1,
    ) -> None:
        super().__init__()

        # Whether or not to activate ResNet block skip connections
        self.residual = residual
        self.residual_scale = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

        padding = tuple(k // 2 for k in kernel)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=not norm)
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm3d(out_channels) if norm else nn.Identity()
        self.activation = nn.Identity() if activation is None else activation()

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel, stride=1, padding=padding, bias=not norm)
        self.bn2 = nn.BatchNorm3d(out_channels) if norm else nn.Identity()

        # Add or skip attention layer based on the use_attention flag
        # self.attention = Attention3D(out_channels) if attn_on else nn.Identity()
        self.attention = CBAM3D(out_channels, 16) if attn_on else nn.Identity()

        self.downsample = downsample
        self.out_channels = out_channels

    def forward(self, x):
        x = x[0] if isinstance(x, tuple) else x  # get only x, ignore residual that is fed back into forward pass
        residual = self.downsample(x) if self.downsample else x

        out = self.conv1(x)
        out = self.dropout(out)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)

        residual = self.attention(residual)

        if self.residual:  # forward skip connection
            out += residual*self.residual_scale
        out = self.activation(out)

        return out, residual


class AttnResBlock2dT(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel=3,
            stride=1,
            upsample=None,
            activation: Optional[Type[nn.Module]] = nn.ReLU,
            dropout=0,
            norm=True,
            residual: bool = True,
            attn_on: bool = False,
            attn_depth: int = 1,
            attn_heads: int = 1,
    ) -> None:
        super().__init__()

        self.residual = residual
        self.residual_scale = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.activation = nn.Identity() if activation is None else activation()
        padding = kernel // 2

        self.convt1 = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=kernel, stride=1, padding=padding, output_padding=0, bias=not norm
        )

        self.bn1 = nn.BatchNorm2d(in_channels) if norm else nn.Identity()
        self.activation = nn.Identity() if activation is None else activation()

        self.attention = Attention2D(in_channels) if attn_on else nn.Identity()

        self.convt2 = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, output_padding=stride - 1, bias=not norm
        )

        self.bn2 = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        self.upsample = upsample
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.convt1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.attention(out)

        out = self.convt2(out)
        out = self.bn2(out)

        if self.upsample:
            residual = self.upsample(x)
        if self.residual:
            out += residual*self.residual_scale
        out = self.activation(out)
        return out

""" ATTENTION """
class LKA3D(nn.Module):
    def __init__(self, nchannels):
        super().__init__()
        self.conv0 = nn.Conv3d(nchannels, nchannels, (5, 5, 5), padding=(2, 2, 2), groups=nchannels)
        self.conv_spatial = nn.Conv3d(nchannels, nchannels, (7, 7, 7), stride=1, padding=(9, 9, 9), groups=nchannels, dilation=(3, 3, 3))
        self.conv1 = nn.Conv3d(nchannels, nchannels, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class Attention3D(nn.Module):
    def __init__(self, nchannels):
        super().__init__()

        self.proj_1 = nn.Conv3d(nchannels, nchannels, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA3D(nchannels)
        self.proj_2 = nn.Conv3d(nchannels, nchannels, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x


class LKA2D(nn.Module):
    def __init__(self, nchannels):
        super().__init__()
        self.conv0 = nn.Conv2d(nchannels, nchannels, (5, 5), padding=(2, 2), groups=nchannels)
        self.conv_spatial = nn.Conv2d(nchannels, nchannels, (7, 7), stride=1, padding=(9, 9), groups=nchannels, dilation=(3, 3))
        self.conv1 = nn.Conv2d(nchannels, nchannels, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class Attention2D(nn.Module):
    def __init__(self, nchannels):
        super().__init__()

        self.proj_1 = nn.Conv2d(nchannels, nchannels, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA2D(nchannels)
        self.proj_2 = nn.Conv2d(nchannels, nchannels, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x



class Conv2DFusion(nn.Module):
    def __init__(
        self,
        channels
    ) -> None:
        super().__init__()

        self.conv_fusion = nn.Conv2d(
            channels*2,
            channels,
            1,
            1
        )

    def forward(self, x):
        return self.conv_fusion(x)


""" RESNETS """
class AttnResNet2DT(nn.Module):
    def __init__(
        self,
        block: nn.Module = AttnResBlock2dT,
        depth: int = 4,
        channels: list = [512, 256, 128, 64, 1],
        kernels: list = [3, 3, 3, 3, 3],
        strides: list = [1, 1, 1, 1, 1],
        dropout: float = 0.0,
        activation=nn.ReLU,
        norm=True,
        sym_residual: bool = True,
        fwd_residual: bool = True,
        attn_on: list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        attn_depth: int = 1,
        attn_heads: int = 1,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.inplanes = channels[0]
        self.sym_residual = sym_residual  # for symmetric skip connections
        self.fwd_residual = fwd_residual  # for forward (normal) skip connections
        self.attn_on = attn_on
        self.residual_scales = nn.ParameterList([nn.Parameter(torch.tensor([1.0]), requires_grad=True) for _ in range(depth)])

        self.layers = nn.ModuleDict({})
        # self.fusion_layers = nn.ModuleDict({})

        for i in range(0, self.depth):
            attn_enabled = False if self.attn_on is None else bool(self.attn_on[i])
            self.layers[str(i)] = self._make_layer(
                block,
                channels[i + 1], # // 2, # CCCCC
                kernel=kernels[i],
                stride=strides[i],
                dropout=dropout,
                activation=activation,
                norm=norm,
                residual=fwd_residual,
                attn_on=attn_enabled,
                attn_depth=attn_depth,
                attn_heads=attn_heads,
            )

            # self.fusion_layers[str(i)] = Conv2DFusion(channels[i])

    def _make_layer(
        self, block, planes, kernel, stride, dropout, activation, norm, residual, attn_on, attn_depth, attn_heads
    ):
        upsample = None
        if stride != 1 or self.inplanes != planes:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    self.inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    output_padding=stride - 1,
                    bias=not norm
                ),
                nn.BatchNorm2d(planes) if norm else nn.Identity(),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                kernel,
                stride,
                upsample,
                activation,
                dropout,
                norm,
                residual,
                attn_on=attn_on,
                attn_depth=attn_depth,
                attn_heads=attn_heads,
            )
        )
        self.inplanes = planes # * 2 # CCCCC

        return nn.Sequential(*layers)

    def forward(self, x, residuals):
        for i in range(0, self.depth):
            if self.sym_residual:  # symmetric skip connection
                res = residuals[-1 - i]
                if res.ndim > x.ndim:  # for 3D to 2D
                    res = torch.mean(res, dim=2)
                if res.shape != x.shape:  # for 2D to 2D with correlation matrix
                    res = F.interpolate(
                        res, size=x.shape[2:], mode="bilinear", align_corners=True
                    )

                # Element-wise addition of residual
                x = x + res * self.residual_scales[i]

                # Concatenation and fusion of residual
                # x = torch.concat((x, res), dim=1)
                # x = self.fusion_layers[str(i)](x)

            x = self.layers[str(i)](x)
        return x


class AttnResNet3D(nn.Module):
    def __init__(
        self,
        block: nn.Module = AttnResBlock3d,
        depth: int = 4,
        channels: list = [1, 64, 128, 256, 512],
        pixel_kernels: list = [3, 3, 3, 3, 3],
        frame_kernels: list = [3, 3, 3, 3, 3],
        pixel_strides: list = [1, 1, 1, 1, 1],
        frame_strides: list = [1, 1, 1, 1, 1],
        dropout: float = 0.0,
        activation=nn.ReLU,
        norm=True,
        residual: bool = False,
        attn_on: list = [0, 0, 0, 0, 0, 0, 0],  # List of 0s and 1s indicating whether attention is applied in each layer
        attn_depth: int = 1,
        attn_heads: int = 1,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.inplanes = channels[0]
        self.attn_on = attn_on

        self.layers = nn.ModuleDict({})
        for i in range(0, self.depth):
            attn_enabled = False if self.attn_on is None else bool(self.attn_on[i])
            _kernel = (frame_kernels[i], pixel_kernels[i], pixel_kernels[i])
            _stride = (frame_strides[i], pixel_strides[i], pixel_strides[i])
            self.layers[str(i)] = self._make_layer(
                block,
                channels[i + 1],
                kernel=_kernel,
                stride=_stride,
                dropout=dropout,
                activation=activation,
                norm=norm,
                residual=residual,
                attn_on=attn_enabled,
                attn_depth=attn_depth,
                attn_heads=attn_heads,
            )

    def _make_layer(
        self, block, planes, kernel, stride, dropout, activation, norm, residual, attn_on, attn_depth, attn_heads
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride, bias=not norm),
                nn.BatchNorm3d(planes) if norm else nn.Identity(),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                kernel,
                stride,
                downsample,
                activation,
                dropout,
                norm,
                residual,
                attn_on=attn_on,
                attn_depth=attn_depth,
                attn_heads=attn_heads,
            )
        )
        self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        residuals = []
        for i in range(0, self.depth):
            x, res = self.layers[str(i)](x)
            residuals.append(res)

        return x, residuals


class AttnResNet2D(nn.Module):
    def __init__(
        self,
        block: nn.Module = AttnResBlock2d,
        depth: int = 4,
        channels: list = [1, 64, 128, 256, 512],
        pixel_kernels: list = [3, 3, 3, 3, 3],
        pixel_strides: list = [1, 1, 1, 1, 1],
        dropout: float = 0.0,
        activation=nn.ReLU,
        norm=True,
        residual: bool = False,
        attn_on: list = [0, 0, 0, 0, 0, 0, 0],  # List of 0s and 1s indicating whether attention is applied in each layer
        attn_depth: int = 1,
        attn_heads: int = 1,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.inplanes = channels[0]
        self.attn_on = attn_on

        self.layers = nn.ModuleDict({})
        for i in range(0, self.depth):
            attn_enabled = False if self.attn_on is None else bool(self.attn_on[i])
            _kernel = (pixel_kernels[i], pixel_kernels[i])
            _stride = (pixel_strides[i], pixel_strides[i])
            self.layers[str(i)] = self._make_layer(
                block,
                channels[i + 1],
                kernel=_kernel,
                stride=_stride,
                dropout=dropout,
                activation=activation,
                norm=norm,
                residual=residual,
                attn_on=attn_enabled,
                attn_depth=attn_depth,
                attn_heads=attn_heads,
            )

    def _make_layer(
        self, block, planes, kernel, stride, dropout, activation, norm, residual, attn_on, attn_depth, attn_heads
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=not norm),
                nn.BatchNorm2d(planes) if norm else nn.Identity(),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                kernel,
                stride,
                downsample,
                activation,
                dropout,
                norm,
                residual,
                attn_on=attn_on,
                attn_depth=attn_depth,
                attn_heads=attn_heads,
            )
        )
        self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        residuals = []
        for i in range(0, self.depth):
            x, res = self.layers[str(i)](x)
            residuals.append(res)

        return x, residuals


""" MODELS """
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
        lr_schedule: str = None,
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
        X = self.encode(X)
        X = self.recode(X)
        X = self.decode(X)
        return X, 1

    def step(self, batch, batch_idx):
        X, Y = batch
        pred_Y, _ = self(X)

        recon = self.metric(pred_Y, Y)  # pixel-wise recon loss
        ssim = 1 - self.ssim(pred_Y.unsqueeze(1), Y.unsqueeze(1))  # SSIM loss
        gdl = self.gdl(pred_Y.unsqueeze(1), Y.unsqueeze(1))  # gradient difference loss

        loss = self.recon_weight * recon + self.ssim_weight * ssim + self.gdl_weight * gdl

        log = ({"recon": recon, "ssim": ssim, "gdl": gdl})

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
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, patience=10, factor=0.5, verbose=False, min_lr=5e-4,
        #     # optimizer, patience=10, factor=0.5, verbose=True, # original params that worked okay
        # )

        if self.hparams.lr_schedule == 'Cyclic':
            num_cycles = 1
            max_steps = 15000
            step_size = max_steps // 2 // num_cycles

            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=1e-4, max_lr=1e-2, cycle_momentum=False, step_size_up=7500, step_size_down=7500, mode="triangular2"
                # optimizer, base_lr=1e-4, max_lr=1e-2, cycle_momentum=False, step_size_up=5000, step_size_down=5000, mode="triangular2"
                # optimizer, base_lr=1e-4, max_lr=1e-2, cycle_momentum=False, step_size_up=step_size, step_size_down=step_size, mode="triangular2"
            )
            scheduler._scale_fn_custom = scheduler._scale_fn_ref()
            scheduler._scale_fn_ref = None

            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "step",
                "frequency": 1
            }

        elif self.hparams.lr_schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=680, gamma=0.2
            )

            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }

        elif self.hparams.lr_schedule == None:
            lr_scheduler = None

        else:
            raise ValueError("Not a valid learning rate scheduler.")

        if lr_scheduler is None:
            return optimizer
        else:
            return [optimizer], [lr_scheduler]


    def on_train_epoch_end(self) -> None:
        self.epoch_plotted = False

    # @plot_with_agg_backend
    def plot_training_results(self, X, Y, pred_Y):
        X = X.cpu()
        Y = Y.cpu()
        pred_Y = pred_Y.cpu()

        if X.ndim == 4:  # frames x Xpix x Ypix
            fig, ax = plt.subplots(ncols=3, nrows=1, dpi=150, figsize=(7.5, 2.5))
            idx = random.randint(0, Y.shape[0] - 1)
            frame_idx = random.randint(0, X.shape[1] - 1)
            ax0 = ax[0].imshow(X[idx, frame_idx, :, :], cmap="gray")
            ax[0].set_title("Input")
            add_colorbar(ax0)
            ax1 = ax[1].imshow(pred_Y[idx, :, :], cmap="twilight_shifted")
            ax[1].set_title("Prediction")
            add_colorbar(ax1)
            ax2 = ax[2].imshow(Y[idx, :, :], cmap="twilight_shifted")
            ax[2].set_title("Truth")
            add_colorbar(ax2)

            plt.tight_layout()
            # dress_fig(tight=True, xlabel="x pixels", ylabel="y pixels", legend=False)

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

    # def _init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
    #             nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0.0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
    #             nn.init.constant_(m.bias, 0.0)
    #         # elif isinstance(m, nn.LayerNorm):
    #         #     nn.init.constant_(m.bias, 0)
    #         #     nn.init.constant_(m.weight, 1.0)


    def _init_weights(self):
        """ Gen 2"""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                # Use He initialization for convolutional layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

                # Zero-initialize the biases
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.Linear):
                # Use He initialization for linear layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

                # Zero-initialize the biases
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
                # Initialize BatchNorm with small positive values to prevent division by zero
                nn.init.normal_(module.weight, mean=1, std=0.02)
                nn.init.constant_(module.bias, 0)


    def count_parameters(self):
        return sum(param.numel() for param in self.parameters())


class PRAUNe(AutoEncoder):
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
        attn: list = [0, 0, 0, 0, 0, 0],
        attn_heads: int = 1,
        attn_depth: int = 1,
        dropout: float = 0.0,
        activation="GELU",
        norm=True,
        fwd_skip: bool = True,
        sym_skip: bool = True,
        lr: float = 5e-3,
        lr_schedule: str = None,
        weight_decay: float = 1e-6,
        metric=nn.MSELoss,
        recon_weight=1.0,
        ssim_weight=1.0,
        gdl_weight=1.0,
        window_size=15,
        plot_interval: int = 5,
        data_info: dict = None,
    ) -> None:
        super().__init__(lr, weight_decay, metric, plot_interval, lr_schedule)

        self.recon_weight = recon_weight
        self.ssim = SSIM(window_size=window_size)
        self.ssim_weight = ssim_weight
        self.gdl = GradientDifferenceLoss()
        self.gdl_weight = gdl_weight

        try:
            activation = getattr(nn, activation)
        except:
            activation = activation

        # If channels are given as a single int, generate a list of that int with length depth
        channels = [1] + [channels] * depth if isinstance(channels, int) else channels
        encoder_channels = [1] + [channels] * depth if isinstance(channels, int) else channels

        self.encoder_channels = encoder_channels[0: depth + 1]
        self.decoder_channels = list(reversed(self.encoder_channels))

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
            block=AttnResBlock3d,
            depth=depth,
            channels=self.encoder_channels,
            pixel_kernels=pixel_kernels,
            frame_kernels=frame_kernels,
            pixel_strides=pixel_strides,
            frame_strides=frame_strides,
            attn_on=attn,
            attn_depth=attn_depth,
            attn_heads=attn_heads,
            dropout=dropout,
            activation=activation,
            norm=norm,
            residual=fwd_skip,
        )

        self.decoder = AttnResNet2DT(
            block=AttnResBlock2dT,
            depth=depth,
            channels=self.decoder_channels,
            kernels=list(reversed(pixel_kernels)),
            strides=list(reversed(pixel_strides)),
            # attn_on=list(reversed(attn[0:depth])),
            attn_on=[0, 0, 0, 0, 0, 0, 0],
            attn_depth=attn_depth,
            attn_heads=attn_heads,
            dropout=dropout,
            activation=activation,
            norm=norm,
            sym_residual=sym_skip,
            fwd_residual=fwd_skip,
        )

        # self.output_activation = nn.Tanh()
        # self.output_activation = nn.Sigmoid()

        self._init_weights()
        self.data_info = data_info
        self.save_hyperparameters()

    def forward(self, X: torch.Tensor):
        X = X.unsqueeze(1) if X.ndim < 5 else X
        Z, res = self.encoder(X)
        D = self.decoder(Z.sum(dim=2), res).squeeze(1)
        del X, Z, res
        # D = self.output_activation(D)
        return D, 1


class SVDAE(AutoEncoder):
    """
    - An autoencoder for the SVD of a correlation matrix, with a 2D ResNet encoder and 2D ResNet decoder.
    - Used here to examine how well a 2D/2D U-Net can denoise the SVD to recover the phase image
    - Consider using just a frozen version of the encoder to encode an SVD latent space for the PRAUNe model
    """
    def __init__(
        self,
        depth: int = 6,
        channels: list = [1, 64, 128, 256, 256, 256, 256],
        pixel_kernels: tuple = (5, 3),
        pixel_downsample: int = 4,
        attn: list = [0, 0, 0, 0, 0, 0, 0, 0, 0],
        attn_heads: int = 1,
        attn_depth: int = 1,
        dropout: float = 0.0,
        activation="GELU",
        norm=True,
        fwd_skip: bool = True,
        sym_skip: bool = True,
        lr: float = 2e-4,
        lr_schedule: str = None,
        weight_decay: float = 1e-6,
        metric=nn.MSELoss,
        recon_weight=1.0,
        ssim_weight=1.0,
        gdl_weight=1.0,
        window_size=15,
        plot_interval: int = 5,
        data_info: dict = None,
    ) -> None:
        super().__init__(lr, weight_decay, metric, plot_interval, lr_schedule)

        self.recon_weight = recon_weight
        self.ssim = SSIM(window_size=window_size)
        self.ssim_weight = ssim_weight
        self.gdl = GradientDifferenceLoss()
        self.gdl_weight = gdl_weight

        try:
            activation = getattr(nn, activation)
        except:
            activation = activation

        encoder_channels = [2] + [channels] * depth if isinstance(channels, int) else channels
        encoder_channels = encoder_channels[0: depth + 1]
        decoder_channels = list(reversed(encoder_channels))
        decoder_channels[-1] = 1

        # # CCCC
        # decoder_channels = [n * 2 for n in decoder_channels]
        # decoder_channels[-1] = 2

        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels

        # Automatically calculate the strides for each layer
        pixel_strides = [
            2 if i < int(np.log2(pixel_downsample)) else 1 for i in range(depth)
        ]

        # And automatically fill the kernel sizes
        pixel_kernels = [
            pixel_kernels[0] if i == 0 else pixel_kernels[1] for i in range(depth)
        ]

        self.encoder = AttnResNet2D(
            block=AttnResBlock2d,
            depth=depth,
            channels=self.encoder_channels,
            pixel_kernels=pixel_kernels,
            pixel_strides=pixel_strides,
            attn_on=attn,
            attn_depth=attn_depth,
            attn_heads=attn_heads,
            dropout=dropout,
            activation=activation,
            norm=norm,
            residual=fwd_skip,
        )

        self.decoder = AttnResNet2DT(
            block=AttnResBlock2dT,
            depth=depth,
            channels=self.decoder_channels,
            kernels=list(reversed(pixel_kernels)),
            strides=list(reversed(pixel_strides)),
            # attn_on=list(reversed(attn[0:depth])),
            attn_on=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            attn_depth=attn_depth,
            attn_heads=attn_heads,
            dropout=dropout,
            activation=activation,
            norm=norm,
            sym_residual=sym_skip,
            fwd_residual=fwd_skip,
        )

        # self.output_activation = nn.Tanh()
        # self.output_activation = nn.Sigmoid()

        self._init_weights()
        self.data_info = data_info
        self.save_hyperparameters()

    def forward(self, X: torch.Tensor):
        X = X.unsqueeze(1) if X.ndim < 4 else X
        Z, res = self.encoder(X)
        D = self.decoder(Z, res).squeeze(1)
        del X, Z, res
        # D = self.output_activation(D)
        return D, 1


class FSVDAE(SVDAE):
    """
    - Same as SVDAE, but converts frames to eigenvalues zsin and zcos during training
    """
    def __init__(
        self,
        nbar_signal=(1e2, 1e5),
        nbar_bkgrnd=(0, 0),
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.frame_to_eigen = FramesToEigenvalues(nbar_signal, nbar_bkgrnd)

    def forward(self, X: torch.Tensor):
        X = self.frame_to_eigen(X)
        X = X.unsqueeze(1) if X.ndim < 4 else X
        Z, res = self.encoder(X)
        D = self.decoder(Z, res).squeeze(1)
        del X, Z, res
        # D = self.output_activation(D)
        return D, 1


class EPRAUNe(PRAUNe):
    """
    Phase Retrieval U-Net (PRUNe). 3D ResNet encoder, 2D ResNet decoder
    """
    def __init__(
        self,
        SVD_encoder = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.svd_encoder = SVD_encoder

        if self.svd_encoder is not None:
        #     for param in self.svd_encoder.parameters():
        #         param.requires_grad = False

            self.conv_fusion = nn.Conv2d(
                # in_channels=self.decoder_channels[0] * 2,
                in_channels=128+64,
                out_channels=self.decoder_channels[0],
                kernel_size=3,
                padding=1,
                stride=1
            )

    def forward(self, X: torch.Tensor, P: torch.Tensor):
        X = X.unsqueeze(1) if X.ndim < 5 else X

        # Forward pass of frames
        X, res = self.encoder(X)

        X = X.sum(dim=2)

        if self.svd_encoder is not None:
            # Get SVD latent
            P, _ = self.svd_encoder(P)

            # Concatenate with frames latent
            X = torch.cat((X, P), dim=1)

            # Combine with convolutional layer (which halves the number of channels)
            X = self.conv_fusion(X)

            del P

        # Decode latent
        X = self.decoder(X, res).squeeze(1)

        # X = self.output_activation(X)
        del res

        return X, 1

    def step(self, batch, batch_idx):
        if self.svd_encoder is not None:
            X, Y, P = batch
        else:
            X, Y = batch
            P = 1

        pred_Y, _ = self(X, P)

        recon = self.metric(pred_Y, Y)  # pixel-wise recon loss
        ssim = 1 - self.ssim(pred_Y.unsqueeze(1), Y.unsqueeze(1))  # SSIM loss
        gdl = self.gdl(pred_Y.unsqueeze(1), Y.unsqueeze(1))  # gradient difference loss

        loss = self.recon_weight * recon + self.ssim_weight * ssim + self.gdl_weight * gdl

        log = ({"recon": recon, "ssim": ssim, "gdl": gdl})

        return loss, log, X, Y, pred_Y

if __name__ == '__main__':

    X = torch.abs(torch.randn(10, 32, 64, 64))
    model = FSVDAE(
        depth=6,
        channels=10,
        pixel_kernels=(5, 3),
        pixel_downsample=4,
        activation="GELU",
        norm=True,
    )
    D, Z = model(X)
    # E, res = model.encoder(X)
    # Count model parameters
    print(model.count_parameters())


# Old
"""

class ConvAttention3D(nn.Module):
    def __init__(self, in_channels, out_channels, attn_depth=1, attn_heads=1, dropout=0.0):
        super(ConvAttention3D, self).__init__()

        self.query_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        self.attn_depth = attn_depth
        self.attn_heads = attn_heads

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, D, H, W = x.size()

        proj_query = self.query_conv(x).view(B, self.attn_heads, -1, D, H, W)
        proj_key = self.key_conv(x).view(B, self.attn_heads, -1, D, H, W)
        proj_value = self.value_conv(x).view(B, self.attn_heads, -1, D, H, W)

        energy = torch.matmul(proj_query, proj_key.transpose(-2, -1))
        attention = self.softmax(energy)
        attention = self.dropout(attention)

        out = torch.matmul(attention, proj_value)
        out = out.view(B, -1, D, H, W)
        out = self.gamma * out + x

        return out


class ConvAttention2D(nn.Module):
    def __init__(self, in_channels, out_channels, attn_depth=1, attn_heads=1, dropout=0.0):
        super(ConvAttention2D, self).__init__()

        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.attn_depth = attn_depth
        self.attn_heads = attn_heads

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.size()

        proj_query = self.query_conv(x).view(B, self.attn_heads, -1, H, W)
        proj_key = self.key_conv(x).view(B, self.attn_heads, -1, H, W)
        proj_value = self.value_conv(x).view(B, self.attn_heads, -1, H, W)

        energy = torch.matmul(proj_query, proj_key.transpose(-2, -1))
        attention = self.softmax(energy)
        attention = self.dropout(attention)

        out = torch.matmul(attention, proj_value)
        out = out.view(B, -1, H, W)
        out = self.gamma * out + x

        return out
        
"""