import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Type
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


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


class AttentionBlock(nn.Module):
    def __init__(
        self,
        out_dim: int,
        depth: int,
        num_heads: int,
        activation: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.projection_layer = nn.LazyLinear(out_dim)
        # create some attention heads
        self.heads = nn.ModuleList(
            [
                MLPStack(
                    out_dim,
                    depth,
                    activation,
                    output_activation=activation,
                    residual=True,
                )
                for _ in range(num_heads)
            ]
        )
        self.attention = nn.Sequential(nn.LazyLinear(out_dim), nn.Softmax())
        self.transform_layers = MLPStack(out_dim, depth * 2, activation, residual=False)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # project so we can use residual connections
        projected_values = self.projection_layer(data)
        # stack up the results of each head
        outputs = torch.stack([head(projected_values) for head in self.heads], dim=1)
        weights = self.attention(outputs)
        weighted_values = (weights * outputs).flatten(1)
        return self.transform_layers(weighted_values)


class Conv2DBlock(nn.Module):
    def __init__(
        self,
        out_dim: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: Optional[Type[nn.Module]] = None,
        norm: Optional[Type[nn.Module]] = None,
        residual: bool = False,
        trans: bool = False,
    ) -> None:
        super().__init__()

        if trans:
            self.conv = nn.LazyConvTranspose2d(
                out_dim, kernel, stride, padding, output_padding=padding
            )
        else:
            self.conv = nn.LazyConv2d(out_dim, kernel, stride, padding)

        self.norm = nn.Identity() if norm is None else nn.LazyBatchNorm2d()
        self.activation = nn.Identity() if activation is None else activation()
        self.model = nn.Sequential(self.conv, self.norm, self.activation)
        self.residual = residual

        if self.residual:
            self.downsample = nn.Sequential(
                nn.LazyConv2d(out_dim, 1, stride, bias=False),
                self.norm,
            )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Y = self.model(X)
        if self.residual:
            Y += self.downsample(X)
        return Y


class Conv2DStack(nn.Module):
    def __init__(
        self,
        channels,
        kernel: int = 3,
        stride: int = 1,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        output_activation: Optional[Type[nn.Module]] = None,
        norm: Optional[Type[nn.Module]] = None,
        residual: bool = False,
        trans: bool = False,
    ) -> None:
        super().__init__()

        self.norm = nn.Identity() if norm is None else nn.LazyBatchNorm2d()

        blocks = [
            Conv2DBlock(
                channels[1],
                kernel,
                stride,
                norm=norm,
                activation=activation,
                residual=False,
                trans=trans,
            )
        ]
        for idx in range(1, len(channels) - 1):
            blocks.append(
                Conv2DBlock(
                    channels[idx + 1],
                    kernel,
                    stride,
                    norm=norm,
                    activation=activation,
                    residual=residual,
                    trans=trans,
                )
            )
        blocks.append(
            Conv2DBlock(
                channels[-1],
                kernel,
                stride,
                norm=norm,
                activation=output_activation,
                residual=False,
                trans=trans,
            )
        )

        self.model = nn.Sequential(*blocks)
        self.residual = residual

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Y = self.model(X)
        Y = self.norm(Y)  # normalize the result to prevent exploding values
        return Y


class Conv3DBlock(nn.Module):
    def __init__(
        self,
        out_dim: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: Optional[Type[nn.Module]] = None,
        norm: Optional[Type[nn.Module]] = None,
        residual: bool = False,
        trans: bool = False,
    ) -> None:
        super().__init__()

        if trans:
            self.conv = nn.LazyConvTranspose3d(
                out_dim, kernel, stride, padding, output_padding=padding
            )
        else:
            self.conv = nn.LazyConv3d(out_dim, kernel, stride, padding)

        self.norm = nn.Identity() if norm is None else nn.LazyBatchNorm3d()
        self.activation = nn.Identity() if activation is None else activation()
        self.model = nn.Sequential(self.conv, self.norm, self.activation)
        self.residual = residual

        if self.residual:
            self.downsample = nn.Sequential(
                nn.LazyConv3d(out_dim, 1, stride, bias=False),
                self.norm,
            )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Y = self.model(X)
        if self.residual:
            Y += self.downsample(X)
        return Y


class Conv3DStack(nn.Module):
    def __init__(
        self,
        channels,
        kernel: int = 3,
        stride: int = 1,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        output_activation: Optional[Type[nn.Module]] = None,
        norm: Optional[Type[nn.Module]] = None,
        residual: bool = False,
        trans: bool = False,
    ) -> None:
        super().__init__()

        self.norm = nn.Identity() if norm is None else nn.LazyBatchNorm3d()

        blocks = [
            Conv3DBlock(
                channels[1],
                kernel,
                stride,
                norm=norm,
                activation=activation,
                residual=False,
                trans=trans,
            )
        ]
        for idx in range(1, len(channels) - 1):
            blocks.append(
                Conv3DBlock(
                    channels[idx + 1],
                    kernel,
                    stride,
                    norm=norm,
                    activation=activation,
                    residual=residual,
                    trans=trans,
                )
            )
        blocks.append(
            Conv3DBlock(
                channels[-1],
                kernel,
                stride,
                norm=norm,
                activation=output_activation,
                residual=False,
                trans=trans,
            )
        )

        self.model = nn.Sequential(*blocks)
        self.residual = residual

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Y = self.model(X)
        Y = self.norm(Y)  # normalize the result to prevent exploding values
        return Y


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


class ResNet2D_new(nn.Module):
    def __init__(
        self,
        block: nn.Module = ResBlock2d,
        depth: int = 4,
        channels: list = [1, 64, 128, 256, 512],
        pixel_kernels: list = [3, 3, 3, 3, 3],
        pixel_strides: list = [1, 1, 1, 1, 1],
        layers: list = [1, 1, 1, 1],
        dropout: float = 0.0,
        activation=nn.ReLU,
        norm=True,
        residual: bool = False,
    ) -> None:
        super(ResNet2D_new, self).__init__()
        self.depth = depth
        self.inplanes = channels[0]

        self.layers = nn.ModuleDict({})
        for i in range(0, self.depth):
            _kernel = (pixel_kernels[i], pixel_kernels[i])
            _stride = (pixel_strides[i], pixel_strides[i])
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


class ResNet3D_original(nn.Module):
    def __init__(
        self,
        block: nn.Module = ResBlock3d,
        first_layer_args: dict = {
            "kernel": (7, 7, 7),
            "stride": (2, 2, 2),
            "padding": (3, 3, 3),
        },
        depth: int = 4,
        channels: list = [1, 64, 128, 256, 512],
        pixel_strides: list = [1, 1, 1, 1, 1],
        frame_strides: list = [1, 1, 1, 1, 1],
        layers: list = [1, 1, 1, 1],
        dropout: list = [0.0, 0.0, 0.0, 0.0],
        activation=nn.ReLU,
        residual: bool = False,
    ) -> None:
        super(ResNet3D_original, self).__init__()
        self.depth = depth
        self.inplanes = channels[1]

        # First layer with different kernel and stride; gives some extra control over frame dimension versus image XY dimensions
        self.conv_in = nn.Sequential(
            nn.Conv3d(
                channels[0],
                channels[1],
                kernel_size=first_layer_args["kernel"],
                stride=first_layer_args["stride"],
                padding=first_layer_args["padding"]
                # padding=tuple(k//2 for k in first_layer_args['kernel'])
            ),
            nn.BatchNorm3d(channels[1]),
            activation(),
        )

        self.layers = nn.ModuleDict({})
        for i in range(0, self.depth):
            _stride = (frame_strides[i], pixel_strides[i], pixel_strides[i])
            self.layers[str(i)] = self._make_layer(
                block,
                channels[i + 1],
                layers[i],
                kernel=(3, 3, 3),
                stride=_stride,
                activation=activation,
                dropout=dropout[i],
                residual=residual,
            )

    def _make_layer(
        self, block, planes, blocks, kernel, stride, activation, dropout, residual
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm3d(planes),
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


class DeconvBlock2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        stride,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
    ) -> None:
        super(DeconvBlock2d, self).__init__()

        self.activation = nn.Identity() if activation is None else activation()
        padding = kernel // 2

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                in_channels,
                kernel_size=kernel,
                stride=1,
                padding=padding,
                output_padding=0,
            ),
            nn.BatchNorm2d(in_channels),
            self.activation,
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                output_padding=stride - 1,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.out_channels = out_channels

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.activation(out)
        return out


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


class DeconvNet2D(nn.Module):
    def __init__(
        self,
        block: nn.Module = DeconvBlock2d,
        last_layer_args: dict = {
            "kernel": (7, 7),
            "stride": (2, 2),
            "padding": (3, 3, 3),
        },
        depth: int = 4,
        channels: list = [512, 256, 128, 64, 1],
        strides: list = [1, 1, 1, 1, 1],
        layers: list = [1, 1, 1, 1],
        activation=nn.ReLU,
        residual: bool = True,
    ) -> None:
        super(DeconvNet2D, self).__init__()
        self.depth = depth
        self.inplanes = channels[0]
        self.residual = residual
        self.layers = nn.ModuleDict({})

        for i in range(0, self.depth):
            if i == self.depth - 1:
                self.layers[str(i)] = self._make_layer(
                    block,
                    channels[i],
                    layers[i],
                    stride=strides[i],
                    activation=activation,
                )
            else:
                self.layers[str(i)] = self._make_layer(
                    block,
                    channels[i + 1],
                    layers[i],
                    stride=strides[i],
                    activation=activation,
                )

        # Mirrors conv_in of encoder
        self.conv_out = nn.Sequential(
            nn.ConvTranspose2d(
                channels[-2],
                channels[-1],
                kernel_size=last_layer_args["kernel"],
                stride=last_layer_args["stride"],
                padding=last_layer_args["padding"],
                # padding=tuple(k//2 for k in last_layer_args['kernel']),
                output_padding=tuple(s - 1 for s in last_layer_args["stride"]),
            ),
            nn.BatchNorm2d(channels[-1]),
            activation(),
        )

    def _make_layer(
        self, block, planes, blocks, kernel=3, stride=1, activation=nn.ReLU
    ):
        layers = []
        layers.append(block(self.inplanes, planes, kernel, stride, activation))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel, stride, activation))

        return nn.Sequential(*layers)

    def forward(self, x, residuals):
        for i in range(0, self.depth):
            res = residuals[-1 - i]
            if res.ndim > x.ndim:  # for 3D to 2D
                res = torch.mean(res, dim=2)
            if res.shape != x.shape:  # for 2D to 2D with correlation matrix
                res = F.interpolate(
                    res, size=x.shape[2:], mode="bilinear", align_corners=True
                )
            if self.residual:  # symmetric skip connection
                x = x + res
            x = self.layers[str(i)](x)
        x = self.conv_out(x)
        return x


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


class ResBlock2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=(3, 3),
        stride=1,
        dilation=1,
        downsample=None,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        dropout=0.1,
        residual: bool = True,
    ) -> None:
        super(ResBlock2D, self).__init__()

        self.residual = residual
        self.activation = nn.Identity() if activation is None else activation()
        if isinstance(kernel, int):
            # padding = kernel//2
            padding = (
                (stride // 4) + dilation * (kernel - 1)
            ) // 2  # crazy but works for stride 2 and 4
        else:
            padding = tuple(k // 2 for k in kernel)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            # nn.BatchNorm2d(out_channels),
            self.activation,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel,
                stride=1,
                padding=padding,
                dilation=dilation,
            ),
            # nn.BatchNorm2d(out_channels),
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


class DeconvolutionNetwork(nn.Module):
    def __init__(
        self,
        channels: list = [1, 16, 32, 64, 128],
        depth: int = 2,
        kernel_size: int = 3,
        stride: int = 2,
        activation: nn.Module = nn.ReLU,
    ):
        super(DeconvolutionNetwork, self).__init__()

        activation = nn.Identity if activation is None else activation
        layers = []
        layers.append(
            DeconvBlock2d(channels[0], channels[1], kernel_size, stride, activation)
        )
        for i in range(1, depth - 1):
            layers.append(
                DeconvBlock2d(
                    channels[i], channels[i + 1], kernel_size, stride, activation
                )
            )
        layers.append(
            DeconvBlock2d(channels[depth - 1], 1, kernel_size, stride, activation)
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x).squeeze(1)


class DeconvNet(nn.Module):
    """
    - Deconvolutional network with residual connections which automatically determines the strides needed
    to upsample the image to the desired size, based on the size_ratio. The last_layer_args can be used
    to ensure the kernel is divisible by the stride, which helps prevent checkerboard artifacts in
    the reconstructed image.
    - The kernel size should be equal to the stride in the last layer, which is intended to resemble the reverse process
    of making image patches using convolutional embedding (where kernel size = stride).
    """

    def __init__(
        self,
        channels,
        size_ratio,
        depth: int = 2,
        kernel_size: int = 3,
        strides: list = [1, 1, 1, 1, 1],
        last_layer_args: dict = {"kernel": 2, "stride": 2},
        activation: nn.Module = nn.ReLU,
    ):
        super(DeconvNet, self).__init__()

        ndouble = int(np.log2(size_ratio)) - int(
            np.log2(last_layer_args["stride"])
        )  # This determines how many times the image needs to be doubled
        for i in range(ndouble):
            strides[i] = 2

        activation = nn.Identity if activation is None else activation
        layers = []
        layers.append(
            DeconvBlock2d(channels[0], channels[1], kernel_size, strides[0], activation)
        )
        for i in range(1, depth - 1):
            layers.append(
                DeconvBlock2d(
                    channels[i], channels[i + 1], kernel_size, strides[i], activation
                )
            )
        layers.append(
            nn.ConvTranspose2d(
                channels[depth - 1],
                1,
                last_layer_args["kernel"],
                last_layer_args["stride"],
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x).squeeze(1)


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

class SelfAttention3d(nn.Module):
    def __init__(self, in_channels, num_heads=4, depth=1):
        super(SelfAttention3d, self).__init__()
        self.num_heads = num_heads
        self.depth = depth

        self.query_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        self.attention_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(in_channels, in_channels, kernel_size=1),
                    nn.ReLU(),
                    nn.Conv3d(in_channels, in_channels, kernel_size=1),
                )
                for _ in range(self.depth)
            ]
        )

        self.final_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)

        query = query.view(query.size(0), self.num_heads, -1, *query.shape[2:])
        key = key.view(key.size(0), self.num_heads, -1, *key.shape[2:])
        value = value.view(value.size(0), self.num_heads, -1, *value.shape[2:])

        attention_map = torch.matmul(query.transpose(-2, -1), key)
        attention_map = self.softmax(attention_map)

        out = torch.matmul(attention_map, value)
        out = out.view(x.size(0), -1, *out.shape[3:])

        for layer in self.attention_layers:
            out = layer(out)

        out = self.final_conv(out)
        return out


class AttnResBlock3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=(3, 3, 3),
        stride=1,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        dropout=0.0,
        norm=True,
        attn_on=0,
        attn_heads=2,
        attn_depth=2,
        residual: bool = True,
        downsample=None,
    ) -> None:
        super(AttnResBlock3d, self).__init__()

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
            ),
            nn.BatchNorm3d(out_channels),
            self.activation,
        )

        self.attention = (
            SelfAttention3d(out_channels, attn_heads, attn_depth)
            if attn_on
            else nn.Identity()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=kernel,
                stride=1,
                padding=padding,
            ),
            nn.BatchNorm3d(out_channels),
            nn.Dropout(dropout),
        )
        self.downsample = downsample
        self.out_channels = out_channels

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]  # get only x, ignore residual that is fed back into forward pass
        residual = x
        out = self.conv1(x)
        out = self.attention(out)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        if self.residual:  # forward skip connection
            out += residual
        out = self.activation(out)
        return out, residual


class AttnResNet3D(nn.Module):
    def __init__(
        self,
        depth: int = 4,
        channels: list = [1, 64, 128, 256, 512],
        pixel_kernels: list = [3, 3, 3, 3, 3],
        frame_kernels: list = [3, 3, 3, 3, 3],
        pixel_strides: list = [1, 1, 1, 1, 1],
        frame_strides: list = [1, 1, 1, 1, 1],
        attn_on: list = [0, 0, 0, 0],
        attn_heads: int = 2,
        attn_depth: int = 2,
        dropout: float = 0.0,
        activation=nn.ReLU,
        norm=True,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.inplanes = channels[0]

        self.layers = nn.ModuleDict({})
        for i in range(0, self.depth):
            _kernel = (frame_kernels[i], pixel_kernels[i], pixel_kernels[i])
            _stride = (frame_strides[i], pixel_strides[i], pixel_strides[i])
            self.layers[str(i)] = self._make_layer(
                planes=channels[i + 1],
                kernel=_kernel,
                stride=_stride,
                activation=activation,
                dropout=dropout,
                norm=norm,
                attn_on=attn_on[i],
                attn_heads=attn_heads,
                attn_depth=attn_depth,
                residual=residual,
            )

    def _make_layer(
        self,
        planes,
        kernel,
        stride,
        activation,
        dropout,
        norm,
        attn_on,
        attn_heads,
        attn_depth,
        residual,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm3d(planes) if norm else nn.Identity(),
            )
        layers = [
            AttnResBlock3d(
                in_channels=self.inplanes,
                out_channels=planes,
                kernel=kernel,
                stride=stride,
                activation=activation,
                dropout=dropout,
                norm=norm,
                attn_on=attn_on,
                attn_heads=attn_heads,
                attn_depth=attn_depth,
                residual=residual,
                downsample=downsample,
            )
        ]
        self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        residuals = []
        for i in range(0, self.depth):
            x, res = self.layers[str(i)](x)
            residuals.append(res)

        return x, residuals


class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, input_size, heads=8, head_size=64, dropout=0.0):
        super().__init__()
        inner_size = head_size * heads
        project_out = not (heads == 1 and head_size == input_size)

        self.heads = heads
        self.scale = head_size**-0.5

        self.norm = nn.LayerNorm(input_size)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(input_size, inner_size * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_size, input_size), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, input_size, depth, heads, head_size, hidden_size, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            input_size,
                            heads=heads,
                            head_size=head_size,
                            dropout=dropout,
                        ),
                        FeedForward(input_size, hidden_size, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SelfAttention3D(nn.Module):
    def __init__(
        self,
        input_shape,
        image_patch_size,
        frame_patch_size,
        embedding_size,
        hidden_size,
        head_size,
        depth,
        nheads,
        dropout=0.0,
    ):
        super().__init__()

        channels = input_shape[1]
        frames = input_shape[2]
        image_height = input_shape[3]
        image_width = input_shape[4]
        patch_height, patch_width = pair(image_patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."
        assert (
            frames % frame_patch_size == 0
        ), "Frames must be divisible by frame patch size"

        num_patches = (
            (image_height // patch_height)
            * (image_width // patch_width)
            * (frames // frame_patch_size)
        )
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embedding_size))

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)",
                p1=patch_height,
                p2=patch_width,
                pf=frame_patch_size,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embedding_size),
            nn.LayerNorm(embedding_size),
        )

        self.transformer = Transformer(
            embedding_size,
            depth,
            nheads,
            hidden_size,
            head_size,
            dropout,
        )

        self.from_patch_embedding = nn.Sequential(
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, patch_dim),
            nn.LayerNorm(patch_dim),
            Rearrange(
                "b (f h w) (p1 p2 pf c) -> b c (f pf) (h p1) (w p2)",
                f=frames // frame_patch_size,
                h=image_height // patch_height,
                w=image_width // patch_width,
                p1=patch_height,
                p2=patch_width,
                pf=frame_patch_size,
            ),
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        x += self.pos_embedding[:, : (n + 1)]
        x = self.transformer(x)
        x = self.from_patch_embedding(x)
        return x


class AttentionResNet3D(nn.Module):
    def __init__(
        self,
        input_shape: tuple,
        depth: int,
        channels: list,
        pixel_kernels: list,
        frame_kernels: list,
        pixel_strides: list,
        frame_strides: list,
        attn_on: list,
        attn_args: dict,
        dropout: float = 0.0,
        activation=nn.ReLU,
        norm=True,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.inplanes = channels[0]

        input_shapes = self.get_input_shapes(
            torch.randn(input_shape), channels[1:], pixel_strides, frame_strides
        )

        self.layers = nn.ModuleDict({})
        for i in range(0, self.depth):
            _kernel = (frame_kernels[i], pixel_kernels[i], pixel_kernels[i])
            _stride = (frame_strides[i], pixel_strides[i], pixel_strides[i])
            self.layers[str(i)] = self._make_layer(
                input_shape=input_shapes[i],
                planes=channels[i + 1],
                kernel=_kernel,
                stride=_stride,
                activation=activation,
                dropout=dropout,
                norm=norm,
                attn_on=attn_on[i],
                attn_args=attn_args,
                residual=residual,
            )

    def _make_layer(
        self,
        input_shape,
        planes,
        kernel,
        stride,
        activation,
        dropout,
        norm,
        attn_on,
        attn_args,
        residual,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm3d(planes) if norm else nn.Identity(),
            )

        attention = (
            SelfAttention3D(
                input_shape=input_shape,
                **attn_args,
            )
            if attn_on
            else nn.Identity()
        )

        layers = []
        layers.append(attention)
        layers.append(
            ResBlock3d(
                in_channels=self.inplanes,
                out_channels=planes,
                kernel=kernel,
                stride=stride,
                activation=activation,
                dropout=dropout,
                residual=residual,
                downsample=downsample,
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

    @staticmethod
    def get_input_shapes(input_tensor, channels, pixel_strides, frame_strides):
        input_shapes = [input_tensor.shape]
        for i, (ch, ps, fs) in enumerate(zip(channels, pixel_strides, frame_strides)):
            input_shape = input_shapes[i]
            input_shapes.append(
                (
                    input_shape[0],
                    ch,
                    input_shape[2] // fs,
                    input_shape[3] // ps,
                    input_shape[4] // ps,
                )
            )
        return input_shapes


if __name__ == "__main__":
    nchannels = 1
    input_tensor = torch.randn(2, 1, 32, 64, 64)

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

    model = AttentionResNet3D(
        input_shape=input_tensor.shape,
        depth=3,
        channels=[1, 16, 16, 16, 16, 16],
        pixel_kernels=[3, 3, 3, 3],
        frame_kernels=[3, 3, 3, 3],
        pixel_strides=[2, 2, 1, 1],
        frame_strides=[2, 2, 2, 2],
        attn_on=[1, 1, 1, 1],
        attn_args=attn_args,
        residual=True,
    )

    y, z = model(input_tensor)
    print(y.shape)
