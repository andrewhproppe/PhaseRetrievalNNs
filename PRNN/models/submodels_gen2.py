import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Type
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn as nn

from PRNN.pipeline.transforms import TensorNormalize


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


### MISC ###
class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


### MLP ###
class MLPBlock(nn.Module):
    def __init__(
        self,
        out_dim: int,
        activation: Optional[Type[nn.Module]] = None,
        residual: bool = False,
    ) -> None:
        super().__init__()
        linear = nn.LazyLinear(out_dim)
        norm = nn.LazyBatchNorm1d()
        activation = nn.Identity() if activation is None else activation()
        self.model = nn.Sequential(linear, norm, activation)
        self.residual = residual

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        output = self.model(data)
        if self.residual:
            # output += data
            output = output + data
        return output


class MLPStack(nn.Module):
    def __init__(
        self,
        out_dim: int,
        depth: int,
        activation: Optional[Type[nn.Module]] = None,
        output_activation: Optional[Type[nn.Module]] = None,
        residual: bool = False,
    ) -> None:
        super().__init__()
        blocks = [MLPBlock(out_dim, activation, residual=False)]
        for _ in range(depth):
            blocks.append(MLPBlock(out_dim, activation, residual=True))
        blocks.append(MLPBlock(out_dim, output_activation, residual=False))
        self.model = nn.Sequential(*blocks)
        self.residual = residual
        self.norm = nn.LazyBatchNorm1d()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        output = self.model(data)
        # skip connection through the full model
        if self.residual:
            # output += data
            output = output + data
        # normalize the result to prevent exploding values
        output = self.norm(output)
        return output


### RESNET BLOCKS ###
class ResBlock2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=(3, 3),
        stride=1,
        downsample=None,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        dropout=0.0,
        norm=True,
        residual: bool = True,
    ) -> None:
        super(ResBlock2d, self).__init__()

        self.residual = residual
        self.activation = nn.Identity() if activation is None else activation()
        padding = tuple(k // 2 for k in kernel)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                bias=not norm
            ),
            nn.BatchNorm2d(out_channels) if norm else nn.Identity(),
            self.activation,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel,
                stride=1,
                padding=padding,
                bias=not norm
            ),
            nn.BatchNorm2d(out_channels) if norm else nn.Identity(),
            nn.Dropout(dropout),
        )
        self.downsample = downsample
        self.out_channels = out_channels

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]  # get only x, ignore residual that is fed back into forward pass
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        if self.residual:  # forward skip connection
            out += residual
        out = self.activation(out)
        return out, residual


class ResBlock2dT(nn.Module):
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
    ) -> None:
        super(ResBlock2dT, self).__init__()

        self.residual = residual
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
        out = self.convt2(out)
        if self.upsample:
            residual = self.upsample(x)
        if self.residual:
            out += residual
        out = self.activation(out)
        return out


class ResBlock3d(nn.Module):
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
    ) -> None:
        super(ResBlock3d, self).__init__()

        self.residual = residual
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
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        if self.residual:  # forward skip connection
            out += residual
        out = self.activation(out)
        return out, residual



### RESNETS ###
class ResNet2D(nn.Module):
    def __init__(
        self,
        block: nn.Module = ResBlock2d,
        first_layer_args: dict = {
            "kernel": (7, 7),
            "stride": (2, 2),
            "padding": (3, 3),
        },
        depth: int = 4,
        channels: list = [1, 64, 128, 256, 512],
        strides: list = [1, 1, 1, 1, 1],
        layers: list = [1, 1, 1, 1],
        dropout: list = [0.0, 0.0, 0.0, 0.0],
        activation: nn.Module = nn.ReLU,
        residual: bool = False,
    ) -> None:
        super(ResNet2D, self).__init__()
        self.depth = depth
        self.inplanes = channels[1]

        # First layer with different kernel and stride; gives some extra control over frame dimension versus image XY dimensions
        self.conv_in = nn.Sequential(
            nn.Conv2d(
                channels[0],
                channels[1],
                kernel_size=first_layer_args["kernel"],
                stride=first_layer_args["stride"],
                padding=first_layer_args["padding"]
                # padding=tuple(k//2 for k in first_layer_args['kernel'])
            ),
            nn.BatchNorm2d(channels[1]),
            activation(),
        )

        self.layers = nn.ModuleDict({})
        for i in range(0, self.depth):
            self.layers[str(i)] = self._make_layer(
                block,
                channels[i + 1],
                layers[i],
                kernel=(3, 3),
                stride=strides[i],
                activation=activation,
                dropout=dropout[i],
                residual=residual,
            )

    def _make_layer(
        self, block, planes, blocks, kernel, stride, activation, dropout, residual
    ):
        """Modified from Nourman (https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/)"""
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
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
                residual,
            )
        )
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    kernel,
                    1,
                    None,
                    activation,
                    dropout,
                    residual,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_in(x)
        residuals = []
        for i in range(0, self.depth):
            x, res = self.layers[str(i)](x)
            residuals.append(res)

        return x, residuals


class ResNet2DT(nn.Module):
    def __init__(
        self,
        block: nn.Module = ResBlock2dT,
        depth: int = 4,
        channels: list = [512, 256, 128, 64, 1],
        kernels: list = [3, 3, 3, 3, 3],
        strides: list = [1, 1, 1, 1, 1],
        layers: list = [1, 1, 1, 1],
        dropout: float = 0.0,
        activation=nn.ReLU,
        norm=True,
        sym_residual: bool = True,
        fwd_residual: bool = True,
    ) -> None:
        super(ResNet2DT, self).__init__()
        self.depth = depth
        self.inplanes = channels[0]
        self.sym_residual = sym_residual  # for symmetric skip connections
        self.fwd_residual = fwd_residual  # for forward (normal) skip connections

        self.layers = nn.ModuleDict({})
        for i in range(0, self.depth):
            self.layers[str(i)] = self._make_layer(
                block,
                channels[i + 1],
                layers[i],
                kernel=kernels[i],
                stride=strides[i],
                dropout=dropout,
                activation=activation,
                norm=norm,
                residual=fwd_residual,
            )

    def _make_layer(
        self, block, planes, blocks, kernel, stride, dropout, activation, norm, residual
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
            )
        )
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    kernel,
                    1,
                    None,
                    activation,
                    dropout,
                    norm,
                    residual,
                )
            )

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
                x = x + res
            x = self.layers[str(i)](x)
        return x


class ResNet3D(nn.Module):
    def __init__(
        self,
        block: nn.Module = ResBlock3d,
        depth: int = 4,
        channels: list = [1, 64, 128, 256, 512],
        pixel_kernels: list = [3, 3, 3, 3, 3],
        frame_kernels: list = [3, 3, 3, 3, 3],
        pixel_strides: list = [1, 1, 1, 1, 1],
        frame_strides: list = [1, 1, 1, 1, 1],
        layers: list = [1, 1, 1, 1],
        dropout: float = 0.0,
        activation=nn.ReLU,
        norm=True,
        residual: bool = False,
    ) -> None:
        super(ResNet3D, self).__init__()
        self.depth = depth
        self.inplanes = channels[0]

        self.layers = nn.ModuleDict({})
        for i in range(0, self.depth):
            _kernel = (frame_kernels[i], pixel_kernels[i], pixel_kernels[i])
            _stride = (frame_strides[i], pixel_strides[i], pixel_strides[i])
            self.layers[str(i)] = self._make_layer(
                block,
                channels[i + 1],
                layers[i],
                kernel=_kernel,
                stride=_stride,
                dropout=dropout,
                activation=activation,
                norm=norm,
                residual=residual,
            )

    def _make_layer(
        self, block, planes, blocks, kernel, stride, dropout, activation, norm, residual
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
            )
        )
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    kernel,
                    1,
                    None,
                    activation,
                    dropout,
                    norm,
                    residual,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        residuals = []
        for i in range(0, self.depth):
            x, res = self.layers[str(i)](x)
            residuals.append(res)

        return x, residuals


class MultiScaleCNN(nn.Module):
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
        layer = ResBlock2D( # deleted this old class, will replace with ResBlock2d but may be bugged
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


### MISC ###
class InterpolateUpsample(nn.Module):
    def __init__(self, scale_factor, mode="nearest"):
        super(InterpolateUpsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class UpsampleConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        scale_factor=2,
        mode="nearest",
        activation: nn.Module = nn.ReLU,
    ):
        super(UpsampleConvBlock, self).__init__()
        padding = kernel_size // 2
        self.activation = nn.Identity() if activation is None else activation()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        self.up = InterpolateUpsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.activation(x)
        return x


class UpsampleConvStack(nn.Module):
    def __init__(
        self,
        channels: list = [1, 16, 32, 64, 128],
        depth: int = 2,
        kernel_size: int = 3,
        activation: nn.Module = nn.ReLU,
        scale_factors: list = [2, 2, 2],
        mode="nearest",
    ):
        super(UpsampleConvStack, self).__init__()

        activation = nn.Identity if activation is None else activation
        layers = []
        for i in range(0, depth - 1):
            layers.append(
                UpsampleConvBlock(
                    channels[i],
                    channels[i + 1],
                    kernel_size,
                    scale_factors[i],
                    mode,
                    activation,
                )
            )
        layers.append(
            UpsampleConvBlock(
                channels[depth - 1],
                1,
                kernel_size,
                scale_factors[depth - 1],
                mode,
                activation,
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x).squeeze(1)


class NormalizeAndRescale(nn.Module):
    def __init__(self, nbar_signal, nbar_bkgrnd):
        super(NormalizeAndRescale, self).__init__()
        self.nbar_signal = nbar_signal
        self.nbar_bkgrnd = nbar_bkgrnd

    def forward(self, X):
        # Normalize by mean of sum of frames
        X = X / torch.mean(torch.sum(X, axis=(-2, -1)))

        # Scale to nbar total counts each frame
        signal_levels = torch.randint(low=int(self.nbar_signal[0]), high=int(self.nbar_signal[1]) + 1, size=(X.shape[0],),
                                      device=X.device)
        X = X * signal_levels.view(X.shape[0], 1, 1, 1)

        # Add flat background to all frames
        bkgrnd_levels = torch.randint(low=int(self.nbar_bkgrnd[0]), high=int(self.nbar_bkgrnd[1]) + 1, size=(X.shape[0],),
                                      device=X.device) / (X.shape[-2] * X.shape[-1])
        X = X + bkgrnd_levels.view(X.shape[0], 1, 1, 1)

        return X


class PoissonSampling(nn.Module):
    def forward(self, X):
        return torch.poisson(X)


class SVDlayer(nn.Module):
    def forward(self, X, num_minibatches=2):
        batch_size, nframes, Nx, Ny = X.shape

        minibatch_size = X.shape[0] // num_minibatches
        X_temp_list = []
        for i in range(0, num_minibatches):
            X_temp = torch.flatten(X[i*minibatch_size:(i+1)*minibatch_size], start_dim=2)
            _, _, X_temp = torch.linalg.svd(X_temp, full_matrices=False)
            X_temp_list.append(X_temp)
        X = torch.concat(X_temp_list, dim=0)
        # X = torch.flatten(X, start_dim=2)
        # _, _, X = torch.linalg.svd(X)

        zsin = torch.reshape(X[:, 1, :], (batch_size, Nx, Ny))
        zcos = torch.reshape(X[:, 2, :], (batch_size, Nx, Ny))
        Z = torch.concat((zcos.unsqueeze(1), zsin.unsqueeze(1)), dim=1)

        return Z


class FramesToEigenvalues(nn.Module):
    def __init__(
        self,
        nbar_signal,
        nbar_bkgrnd
    ):
        super(FramesToEigenvalues, self).__init__()
        self.signal_scaling = NormalizeAndRescale(nbar_signal, nbar_bkgrnd)
        self.poisson_layer = PoissonSampling()
        self.svdlayer = SVDlayer()
        self.normalize = TensorNormalize(minmax=(-1, 1))

    def forward(self, X):
        X = self.signal_scaling(X)
        X = self.poisson_layer(X)
        X = self.svdlayer(X)
        X = self.normalize(X)
        return X


if __name__ == "__main__":
    nchannels = 1
    input_tensor = torch.randn(2, 1, 32, 64, 64)
    #
    # attn_args = {
    #     "image_patch_size": 4,
    #     "frame_patch_size": 4,
    #     "embedding_size": 64,
    #     "hidden_size": 128,
    #     "head_size": 128,
    #     "depth": 2,
    #     "nheads": 4,
    #     "dropout": 0.0,
    # }
    #
    # model = AttentionResNet3D(
    #     input_shape=input_tensor.shape,
    #     depth=3,
    #     channels=[1, 16, 16, 16, 16, 16],
    #     pixel_kernels=[3, 3, 3, 3],
    #     frame_kernels=[3, 3, 3, 3],
    #     pixel_strides=[2, 2, 1, 1],
    #     frame_strides=[2, 2, 2, 2],
    #     attn_on=[1, 1, 1, 1],
    #     attn_args=attn_args,
    #     residual=True,
    # )
    #
    # y, z = model(input_tensor)
    # print(y.shape)

