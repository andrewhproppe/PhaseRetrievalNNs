import torch
import pytorch_lightning as pl
import wandb
import random
import torch.nn as nn
import torch.nn.functional as F

from PRNN.visualization.figure_utils import *
from PRNN.models.utils import SSIM, GradientDifferenceLoss, CircularMSELoss
from typing import Optional, Type

# TODO: Implement a ResVANet3D model that uses the VAN block of OverlapPatchEmbed->Transformer->PatchMerging
# https://github.com/Visual-Attention-Network/VAN-Classification/blob/main/models/van.py

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
        self.residual_scale = nn.Parameter(torch.tensor([1e-1]), requires_grad=True)
        self.activation = nn.Identity() if activation is None else activation()
        padding = kernel // 2

        self.convt1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                in_channels,
                kernel_size=kernel,
                stride=1,
                padding=padding,
                output_padding=0,
                bias=not norm,
            ),
            nn.BatchNorm2d(in_channels) if norm else nn.Identity(),
            self.activation,
        )

        # Add or skip attention layer based on the use_attention flag
        self.attention = Attention2D(in_channels) if attn_on else nn.Identity()

        self.convt2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                output_padding=stride - 1,
                bias=not norm,
            ),
            nn.BatchNorm2d(out_channels) if norm else nn.Identity(),
            nn.Dropout(dropout),
        )
        self.upsample = upsample
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.convt1(x)

        # Apply attention if available
        out = self.attention(out)

        out = self.convt2(out)
        if self.upsample:
            residual = self.upsample(x)
        if self.residual:
            out += residual*self.residual_scale
        out = self.activation(out)
        return out


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

        self.residual = residual
        self.residual_scale = nn.Parameter(torch.tensor([1e-1]), requires_grad=True)
        self.activation = nn.Identity() if activation is None else activation()
        padding = tuple(k // 2 for k in kernel)

        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                bias=not norm
            ),
            nn.BatchNorm3d(out_channels) if norm else nn.Identity(),
            self.activation,
        )

        # Add or skip attention layer based on the use_attention flag
        self.attention = Attention3D(out_channels) if attn_on else nn.Identity()

        self.conv2 = nn.Sequential(
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=kernel,
                stride=1,
                padding=padding,
                bias=not norm
            ),
            nn.BatchNorm3d(out_channels) if norm else nn.Identity(),
            nn.Dropout(dropout),
        )
        self.downsample = downsample
        self.out_channels = out_channels

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]  # get only x, ignore residual that is fed back into forward pass
        residual = x
        out = self.conv1(x)

        # Apply attention if available
        out = self.attention(out)

        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        if self.residual:  # forward skip connection
            out += residual*self.residual_scale
        out = self.activation(out)
        return out, residual


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
        attn_on: list = [0, 0, 0, 0, 0, 0, 0],  # List of 0s and 1s indicating whether attention is applied in each layer
        attn_depth: int = 1,
        attn_heads: int = 1,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.inplanes = channels[0]
        self.sym_residual = sym_residual  # for symmetric skip connections
        self.fwd_residual = fwd_residual  # for forward (normal) skip connections
        self.attn_on = attn_on
        self.res_scalars = nn.ParameterList([nn.Parameter(torch.tensor([1e-1]), requires_grad=True) for _ in range(depth)])

        self.layers = nn.ModuleDict({})
        for i in range(0, self.depth):
            attn_enabled = False if self.attn_on is None else bool(self.attn_on[i])
            self.layers[str(i)] = self._make_layer(
                block,
                channels[i + 1],
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
        self.inplanes = planes

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
                x = x + res * self.res_scalars[i]
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
            # elif isinstance(m, nn.LayerNorm):
            #     nn.init.constant_(m.bias, 0)
            #     nn.init.constant_(m.weight, 1.0)


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
        pixel_kernels: tuple = (3, 3),
        frame_kernels: tuple = (3, 3),
        pixel_downsample: int = 4,
        frame_downsample: int = 32,
        attn: list = [0, 0, 0, 0, 0, 0],
        attn_heads: int = 1,
        attn_depth: int = 1,
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

        self.encoder = AttnResNet3D(
            block=AttnResBlock3d,
            depth=depth,
            channels=channels[0 : depth + 1],
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
            channels=list(reversed(channels[0 : depth + 1])),
            kernels=list(reversed(pixel_kernels)),
            strides=list(reversed(pixel_strides)),
            attn_on=list(reversed(attn[0:depth])),
            attn_depth=attn_depth,
            attn_heads=attn_heads,
            dropout=dropout,
            activation=activation,
            norm=norm,
            sym_residual=sym_skip,
            fwd_residual=fwd_skip,
        )

        self.output_activaiton = nn.Tanh()

        self._init_weights()
        self.save_hyperparameters()

    def forward(self, X: torch.Tensor):
        X = X.unsqueeze(1) if X.ndim < 5 else X
        Z, res = self.encoder(X)
        D = self.decoder(Z.sum(dim=2), res).squeeze(1)
        D = self.output_activaiton(D)
        return D, 1


if __name__ == '__main__':
    X = torch.randn(10, 32, 64, 64)
    model = PRAUNe(
        depth=6,
        channels=16,
        pixel_kernels=(5, 3),
        frame_kernels=(5, 3),
        attn=[0, 0, 0, 0, 0, 0],
        pixel_downsample=4,
        frame_downsample=32,
        activation="GELU",
        norm=True,
    )
    D, Z = model(X)

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