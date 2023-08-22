import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Type

# from attention_augmented_conv import AugmentedConv
# from utils import calculate_layer_sizes


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
            ),
            nn.BatchNorm2d(out_channels),
            self.activation,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel,
                stride=1,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
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
            ),
            nn.BatchNorm3d(out_channels),
            self.activation,
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
                nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride),
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
            ),
            nn.BatchNorm2d(in_channels),
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
            ),
            nn.BatchNorm2d(out_channels),
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


# class AttnResBlock3d_bad(nn.Module):
#     def __init__(
#         self,
#         input_size,
#         in_channels,
#         out_channels,
#         kernel=(3, 3, 3),
#         stride=1,
#         downsample=None,
#         activation: Optional[Type[nn.Module]] = nn.ReLU,
#         dropout=0.0,
#         residual: bool = True,
#     ) -> None:
#         super(AttnResBlock3d, self).__init__()
#
#         self.residual = residual
#         self.activation = nn.Identity() if activation is None else activation()
#         padding = tuple(k // 2 for k in kernel)
#
#         self.conv1 = nn.Sequential(
#             AugmentedConv(
#                 in_channels,
#                 out_channels,
#                 kernel_size=kernel,
#                 dk=4,
#                 dv=4,
#                 Nh=2,
#                 relative=True,
#                 stride=stride,
#                 shape=input_size // stride,
#             ),
#             nn.BatchNorm3d(out_channels),
#             self.activation,
#         )
#
#         self.conv2 = nn.Sequential(
#             nn.Conv3d(
#                 out_channels,
#                 out_channels,
#                 kernel_size=kernel,
#                 stride=1,
#                 padding=padding,
#             ),
#             nn.BatchNorm3d(out_channels),
#             nn.Dropout(dropout),
#         )
#         self.downsample = downsample
#         self.out_channels = out_channels
#
#     def forward(self, x):
#         if isinstance(x, tuple):
#             x = x[0]  # get only x, ignore residual that is fed back into forward pass
#         residual = x
#         out = self.conv1(x)
#         out = self.conv2(out)
#         if self.downsample:
#             residual = self.downsample(x)
#         if self.residual:  # forward skip connection
#             out += residual
#         out = self.activation(out)
#         return out, residual


# class AttnResBlock3d(nn.Module):
#     def __init__(
#         self,
#         input_size,
#         in_channels,
#         out_channels,
#         kernel=(3, 3, 3),
#         stride=1,
#         downsample=None,
#         activation: Optional[Type[nn.Module]] = nn.ReLU,
#         dropout=0.0,
#         residual: bool = True,
#     ) -> None:
#         super(AttnResBlock3d, self).__init__()
#
#         self.residual = residual
#         self.activation = nn.Identity() if activation is None else activation()
#         padding = tuple(k // 2 for k in kernel)
#
#         self.conv1 = nn.Sequential(
#             nn.Conv3d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=kernel,
#                 stride=stride,
#                 padding=padding,
#             ),
#             nn.BatchNorm3d(out_channels),
#             self.activation,
#         )
#         self.conv2 = nn.Sequential(
#             AugmentedConv(
#                 in_channels,
#                 out_channels,
#                 kernel_size=3,
#                 dk=4,
#                 dv=4,
#                 Nh=2,
#                 relative=True,
#                 stride=1,
#                 shape=input_size,
#             ),
#             nn.BatchNorm3d(out_channels),
#             nn.Dropout(dropout),
#         )
#         self.downsample = downsample
#         self.out_channels = out_channels
#
#     def forward(self, x):
#         if isinstance(x, tuple):
#             x = x[0]  # get only x, ignore residual that is fed back into forward pass
#         residual = x
#         out = self.conv1(x)
#         out = self.conv2(out)
#         if self.downsample:
#             residual = self.downsample(x)
#         if self.residual:  # forward skip connection
#             out += residual
#         out = self.activation(out)
#         return out, residual
#
#
# class AttnResNet3D(nn.Module):
#     def __init__(
#         self,
#         block: nn.Module = AttnResBlock3d,
#         input_size: int = 64,
#         depth: int = 4,
#         channels: list = [1, 64, 128, 256, 512],
#         pixel_kernels: list = [3, 3, 3, 3, 3],
#         frame_kernels: list = [3, 3, 3, 3, 3],
#         pixel_strides: list = [1, 1, 1, 1, 1],
#         frame_strides: list = [1, 1, 1, 1, 1],
#         dropout: float = 0.0,
#         activation=nn.ReLU,
#         norm=True,
#         residual: bool = False,
#     ) -> None:
#         super(AttnResNet3D, self).__init__()
#         self.depth = depth
#         self.inplanes = channels[0]
#         _input_sizes = calculate_layer_sizes(input_size, pixel_strides, depth)
#
#         self.layers = nn.ModuleDict({})
#         for i in range(0, self.depth):
#             _kernel = (frame_kernels[i], pixel_kernels[i], pixel_kernels[i])
#             _stride = (frame_strides[i], pixel_strides[i], pixel_strides[i])
#             self.layers[str(i)] = self._make_layer(
#                 block,
#                 input_size,
#                 channels[i + 1],
#                 kernel=_kernel,
#                 stride=_stride,
#                 dropout=dropout,
#                 activation=activation,
#                 norm=norm,
#                 residual=residual,
#             )
#
#     def _make_layer(
#         self,
#         block,
#         input_size,
#         planes,
#         kernel,
#         stride,
#         dropout,
#         activation,
#         norm,
#         residual,
#     ):
#         downsample = None
#         if stride != 1 or self.inplanes != planes:
#             downsample = nn.Sequential(
#                 nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride),
#                 nn.BatchNorm3d(planes) if norm else nn.Identity(),
#             )
#         layers = []
#         layers.append(
#             block(
#                 input_size,
#                 self.inplanes,
#                 planes,
#                 kernel,
#                 stride,
#                 downsample,
#                 activation,
#                 dropout,
#                 residual,
#             )
#         )
#         self.inplanes = planes
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         residuals = []
#         for i in range(0, self.depth):
#             x, res = self.layers[str(i)](x)
#             residuals.append(res)
#
#         return x, residuals


# if __name__ == "__main__":
#     model = AttnResNet3D(
#         input_size=64,
#         depth=4,
#         channels=[1, 4, 8, 16, 32],
#         pixel_kernels=[3, 3, 3, 3, 3],
#         frame_kernels=[3, 3, 3, 3, 3],
#         pixel_strides=[1, 1, 1, 1, 1],
#         frame_strides=[1, 1, 1, 1, 1],
#         dropout=0.0,
#         activation=nn.ReLU,
#         norm=True,
#         residual=True,
#     )
#
#     input_tensor = torch.randn(1, 1, 32, 64, 64)
#     model(input_tensor)
