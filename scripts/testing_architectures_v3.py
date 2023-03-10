from typing import Optional, Type
from QIML.models.base import init_fc_layers, get_conv_output_shape, get_conv_flat_shape, Reshape
import numpy as np
import torch
from torch import nn
from QIML.pipeline.QI_data import QIDataModule

import pytorch_lightning as pl
import wandb
import random
from matplotlib import pyplot as plt

class ResBlock3d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel=3,
            stride=1,
            downsample=None,
            activation: Optional[Type[nn.Module]] = nn.ReLU,
    ) -> None:
        super(ResBlock3d, self).__init__()

        self.activation = nn.Identity() if activation is None else activation()
        padding = kernel // 2

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            self.activation)
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel, stride=1, padding=padding),
            nn.BatchNorm3d(out_channels))
        self.downsample = downsample
        self.out_channels = out_channels

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0] # get only x, ignore residual that is fed back into forward pass
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.activation(out)
        return out, residual


class ResNet3D(nn.Module):
    def __init__(
            self,
            block: nn.Module = ResBlock3d,
            depth: int = 4,
            channels: list = [1, 64, 128, 256, 512],
            strides: list = [1, 1, 1, 1, 1],
            layers: list = [1, 1, 1, 1],
            activation: nn.Module = nn.ReLU,
    ) -> None:
        super(ResNet3D, self).__init__()
        self.depth = depth
        self.inplanes = channels[1]
        self.convi = nn.Sequential(
            nn.Conv3d(channels[0], channels[1], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(channels[1]),
            nn.ReLU())

        self.layers = nn.ModuleDict({})
        for i in range(0, self.depth):
            self.layers[str(i)] = self._make_layer(block, channels[i+1], layers[i], stride=strides[i], activation=activation)

    def _make_layer(self, block, planes, blocks, kernel=3, stride=1, activation=nn.ReLU):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm3d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, kernel, stride, downsample, activation))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel, stride, downsample, activation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.convi(x)
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
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel, stride=1, padding=padding, output_padding=padding-1),
            nn.BatchNorm2d(in_channels),
            self.activation)
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, output_padding=stride-1),
            nn.BatchNorm2d(out_channels))
        self.out_channels = out_channels

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.activation(out)
        return out


class DeconvNet2D(nn.Module):
    def __init__(
            self,
            block: nn.Module = ResBlock3d,
            depth: int = 4,
            channels: list = [512, 256, 128, 64, 1],
            strides: list = [1, 1, 1, 1, 1],
            layers: list = [1, 1, 1, 1],
            activation: nn.Module = nn.ReLU,
    ) -> None:
        super(DeconvNet2D, self).__init__()
        self.depth = depth
        self.inplanes = channels[0]
        self.layers = nn.ModuleDict({})

        for i in range(0, self.depth):
            if i == self.depth-1:
                self.layers[str(i)] = self._make_layer(block, channels[i], layers[i], stride=strides[i], activation=activation)
            else:
                self.layers[str(i)] = self._make_layer(block, channels[i+1], layers[i], stride=strides[i], activation=activation)

        # Mirrors convi of encoder
        self.convf = nn.Sequential(
            nn.ConvTranspose2d(channels[-2], channels[-1], kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm2d(channels[-1]),
            nn.ReLU())

    def _make_layer(self, block, planes, blocks, kernel=3, stride=1, activation=nn.ReLU):
        layers = []
        layers.append(block(self.inplanes, planes, kernel, stride, activation))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel, stride, activation))

        return nn.Sequential(*layers)

    def forward(self, x, residuals):
        for i in range(0, self.depth):
            x += residuals[-1-i].mean(axis=2)
            x = self.layers[str(i)](x)
        x = self.convf(x)
        return x




batch = 128
channels = 1
nframes = 16
Xdim = 32
Ydim = Xdim
zdim = 32
data = torch.rand(batch, channels, nframes, Xdim, Ydim) # batch * channels * nframes * Xpixel * Ypixel

depth = 4
channels = [1, 4, 8, 16, 32, 64]
strides = [2, 2, 2, 1, 2, 1]
layers = [1, 1, 1, 1, 1]

encoder = ResNet3D(
    block=ResBlock3d,
    depth=depth,
    channels=channels[0:depth+1],
    strides=strides[0:depth],
    layers=layers[0:depth],
)

decoder = DeconvNet2D(
    block=DeconvBlock2d,
    depth=depth,
    channels=list(reversed(channels[0:depth+1])),
    strides=list(reversed(strides[0:depth])),
    layers=list(reversed(layers[0:depth]))
)


z, res = encoder(data)
out = decoder(z.squeeze(2), res)


# print(z.shape)

# depth = 3
# channels = [1, 10, 10, 10, 10]
# kernel_size = (5, 5, 5)
# padding = tuple(k//2 for k in kernel_size)
