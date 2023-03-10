from typing import Tuple, Dict, Union, Optional, Any, Type
from functools import wraps
from argparse import ArgumentParser

import numpy as np
import torch
import random
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb
import matplotlib as mpl

# mpl.use("Agg")  # this forces a non-X server backend
from matplotlib import pyplot as plt

from QIML.utils import paths
from QIML.pipeline.data import H5Dataset
from QIML.visualization.AP_figs_funcs import *

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


def format_time_sequence(method):
    """
    Define a decorator that modifies the behavior of the
    forward call in a PyTorch model. This basically checks
    to see if the dimensions of the input data are [batch, time, features].
    In the case of 2D data, we'll automatically run the method
    with a view of the tensor assuming each element is an element
    in the sequence.
    """

    @wraps(method)
    def wrapper(model, X: torch.Tensor):
        if X.ndim == 2:
            batch_size, seq_length = X.shape
            output = method(model, X.view(batch_size, seq_length, -1))
        else:
            output = method(model, X)
        return output

    return wrapper


def init_rnn(module):
    for name, parameter in module.named_parameters():
        # use orthogonal initialization for RNNs
        if "weight" in name:
            try:
                nn.init.orthogonal_(parameter)
            # doesn't work for batch norm layers but that's fine
            except ValueError:
                pass
        # set biases to zero
        if "bias" in name:
            nn.init.zeros_(parameter)


def init_fc_layers(module):
    for name, parameter in module.named_parameters():
        if "weight" in name:
            try:
                nn.init.kaiming_uniform_(parameter)
            except ValueError:
                pass

        if "bias" in name:
            nn.init.zeros_(parameter)

def init_layers(module):
    for name, parameter in module.named_parameters():
        if "weight" in name:
            try:
                nn.init.kaiming_uniform_(parameter)
            except ValueError:
                pass

        if "bias" in name:
            nn.init.zeros_(parameter)

def get_conv_output_size(model, input_tensor: torch.Tensor):
    output = model(input_tensor)
    return output.size(-1)

def get_conv_output_shape(model, input_tensor: torch.Tensor):
    output = model(input_tensor)
    return torch.tensor(output.shape)

def get_conv_flat_shape(model, input_tensor: torch.Tensor):
    output = torch.flatten(model(input_tensor[0:1, :, :, :]))
    return output.shape

def get_conv1d_flat_shape(model, input_tensor: torch.Tensor):
    # output = torch.flatten(model(input_tensor[-1, :, :]))
    output = torch.flatten(model(input_tensor))
    return output.shape

def symmetry_loss(profile_output: torch.Tensor):
    """
    Computes a penalty for asymmetric profiles. Basically take
    the denoised profile, and fold half of it on itself and
    calculate the mean squared error. By minimizing this value
    we try to constrain its symmetry.
    Expected profile_output shape is [N, T, 2]

    Parameters
    ----------
    profile_output : torch.Tensor
        The output of the model, expected shape is [N, T, 2]
        for N batch size and T timesteps.

    Returns
    -------
    float
        MSE symmetry loss
    """
    half = profile_output.shape[-1]
    y_a = profile_output[:, :half]
    y_b = profile_output[:, -half:].flip(-1)
    return F.mse_loss(y_a, y_b)

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


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
            self.conv = nn.LazyConvTranspose2d(out_dim, kernel, stride, padding, output_padding=padding)
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

        blocks = [Conv2DBlock(channels[1], kernel, stride, norm=norm, activation=activation, residual=False, trans=trans)]
        for idx in range(1, len(channels)-1):
            blocks.append(Conv2DBlock(channels[idx+1], kernel, stride, norm=norm, activation=activation, residual=residual, trans=trans))
        blocks.append(Conv2DBlock(channels[-1], kernel, stride, norm=norm, activation=output_activation, residual=False, trans=trans))

        self.model = nn.Sequential(*blocks)
        self.residual = residual

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Y = self.model(X)
        Y = self.norm(Y) # normalize the result to prevent exploding values
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
            self.conv = nn.LazyConvTranspose3d(out_dim, kernel, stride, padding, output_padding=padding)
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

        blocks = [Conv3DBlock(channels[1], kernel, stride, norm=norm, activation=activation, residual=False, trans=trans)]
        for idx in range(1, len(channels)-1):
            blocks.append(Conv3DBlock(channels[idx+1], kernel, stride, norm=norm, activation=activation, residual=residual, trans=trans))
        blocks.append(Conv3DBlock(channels[-1], kernel, stride, norm=norm, activation=output_activation, residual=False, trans=trans))

        self.model = nn.Sequential(*blocks)
        self.residual = residual

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Y = self.model(X)
        Y = self.norm(Y) # normalize the result to prevent exploding values
        return Y


class QIAutoEncoder(pl.LightningModule):
    def __init__(
        self, lr: float = 1e-3, weight_decay: float = 0.0, plot_interval: int = 1000
    ) -> None:
        super().__init__()
        self.encoder = None
        self.decoder = None
        self.metric = nn.MSELoss()
        self.save_hyperparameters("lr", "weight_decay", "plot_interval")

    def encode(self, X: torch.Tensor):
        return self.encoder(X)

    def decode(self, Z: torch.Tensor):
        return self.decoder(Z)

    def forward(self, X: torch.Tensor):
        Z = self.encode(X)
        return self.decode(Z), Z

    def step(self, batch, batch_idx):
        X, Y = batch
        pred_Y, Z = self(X)
        recon = self.metric(Y, pred_Y)
        loss = recon
        log = {"recon": recon}
        return loss, log, X, Y, pred_Y

    def training_step(self, batch, batch_idx):
        loss, log, X, Y, pred_Y = self.step(batch, batch_idx)
        self.log("training_loss", log)

        if self.current_epoch > 0 and self.current_epoch % self.hparams.plot_interval == 0 and self.epoch_plotted == False:
            self.epoch_plotted = True # don't plot again in this epoch
            with torch.no_grad():
                fig = self.plot_training_results(X, Y, pred_Y)
                log.update({"plot": fig})
                self.logger.experiment.log(log)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, log, _, _, _ = self.step(batch, batch_idx)
        self.log("validiation_loss", log)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        return optimizer

    def on_train_epoch_end(self) -> None:
        self.epoch_plotted = False

    def plot_training_results(self, X, Y, pred_Y):
        fig, ax = plt.subplots(ncols=2, nrows=1, dpi=150, figsize=(5, 2.5))
        X = X.cpu()
        Y = Y.cpu()
        pred_Y = pred_Y.cpu()
        ax[0].imshow(Y[0, :, :])
        ax[0].set_title('Truth')
        ax[1].imshow(pred_Y[0, :, :])
        ax[1].set_title('Prediction')
        dress_fig(tight=True, xlabel='x pixels', ylabel='y pixels', legend=True)
        wandb.Image(plt)

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

    def initialize_weights(self, μ=0, σ=0.1):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=μ, std=σ)
                nn.init.normal_(m.bias, mean=μ, std=σ)


class QI3Dto2DConvAE(QIAutoEncoder):
    def __init__(
        self,
        input_dim: int = (1, 1, 16, 32, 32),
        channels=[0, 10, 10, 10, 10],
        zdim: int = 32,
        kernel: int = 3,
        stride: int = 2,
        flat_bottleneck: bool = False,
        residual: bool = False,
        norm=None,
        dropout: float = 0.0,
        lr: float = 2e-4,
        weight_decay: float = 0.0,
        plot_interval=50,
    ):
        super().__init__(lr, weight_decay, plot_interval)

        k = kernel
        s = stride
        frame_kernel = input_dim[2] // len(channels)

        self.encoder = Conv3DStack(
            channels=channels,
            kernel=(frame_kernel, k, k),
            stride=(s, s, s),
            residual=residual,
            norm=norm
        )

        self.decoder = Conv2DStack(
            channels=np.flip(channels),
            kernel=(k, k),
            stride=(s, s),
            trans=True,
            norm=norm,
        )

        dummy_input = torch.ones(*input_dim)

        if flat_bottleneck:
            with torch.no_grad():
                conv_shape = get_conv_output_shape(self.encoder, dummy_input)
                flat_shape = get_conv_flat_shape(self.encoder, dummy_input)
            conv_shape[0] = -1

            self.linear_bottleneck = nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(zdim),
                nn.LazyLinear(flat_shape[0]),
                Reshape(*conv_shape)
            )

        else:
            self.linear_bottleneck = nn.Identity()

        with torch.no_grad():
            _ = self(dummy_input)  # this initializes the shapes of the lazy modules

        # self.initialize_weights()


    def forward(self, X: torch.Tensor):
        if X.ndim < 5:
            X = X.unsqueeze(1) # adds the channel dimension
        Z = self.encoder(X)
        F = self.linear_bottleneck(Z)
        D = self.decoder(F.squeeze(2))
        D = D.squeeze(1) # removes the channel dimension
        return D, Z


class Res3Dto2DConvAE(QIAutoEncoder):
    def __init__(
        self,
        input_dim: int = (1, 1, 16, 32, 32),
        channels=[0, 10, 10, 10, 10],
        zdim: int = 32,
        kernel: int = 3,
        stride: int = 2,
        residual: bool = False,
        norm=None,
        dropout: float = 0.0,
        lr: float = 2e-4,
        weight_decay: float = 0.0,
        plot_interval=50,
    ):
        super().__init__(lr, weight_decay, plot_interval)

        k = kernel
        s = stride
        frame_kernel = input_dim[2] // len(channels)

        self.encoder = Conv3DStack(
            channels=channels,
            kernel=(frame_kernel, k, k),
            stride=(s, s, s),
            residual=residual,
            norm=norm
        )

        self.decoder = Conv2DStack(
            channels=np.flip(channels),
            kernel=(k, k),
            stride=(s, s),
            trans=True,
            norm=norm,
        )

        dummy_input = torch.ones(*input_dim)

        with torch.no_grad():
            _ = self(dummy_input)  # this initializes the shapes of the lazy modules

        # self.initialize_weights()


    def forward(self, X: torch.Tensor):
        if X.ndim < 5:
            X = X.unsqueeze(1) # adds the channel dimension
        Z = self.encoder(X)
        D = self.decoder(Z.squeeze(2))
        D = D.squeeze(1) # removes the channel dimension
        return D, Z


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet3D(nn.Module):
    def __init__(self, block, layers):
        super(ResNet3D, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)

        return x


class ResBlock3d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride=1,
            downsample=None
    ) -> None:
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out, residual


class ResNet3D(nn.Module):
    def __init__(
            self,
            block: nn.Module = ResBlock3d,
            depth: int = 4,
            channels: tuple = (1, 64, 128, 256, 512),
            strides: tuple = (1, 1, 1, 1, 1),
            layers: tuple = (1, 1, 1, 1)
    ) -> None:
        super(ResNet3D, self).__init__()
        self.depth = depth
        self.inplanes = channels[1]
        self.conv1 = nn.Sequential(
            nn.Conv3d(channels[0], channels[1], kernel_size=7, stride=strides[0], padding=3),
            nn.BatchNorm3d(channels[1]),
            nn.ReLU())
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleDict({})
        for i in range(0, self.depth):
            self.layers[str(i)] = self._make_layer(block, channels[i+1], layers[i], stride=strides[i])

        # self.layer0 = self._make_layer(block, channels[1], layers[0], stride=strides[1])
        # self.layer1 = self._make_layer(block, channels[2], layers[1], stride=strides[2])
        # self.layer2 = self._make_layer(block, channels[3], layers[2], stride=strides[3])
        # self.layer3 = self._make_layer(block, channels[4], layers[3], stride=strides[4])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm3d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        residuals = []
        for i in range(0, self.depth):
            x, res = self.layers[str(i)](x)
            residuals.append(res)

        # x, res1 = self.layer0(x)
        # x, res2 = self.layer1(x)
        # x, res3 = self.layer2(x)
        # x, res4 = self.layer3(x)

        return x, residuals

models = {
    "QI3Dto2DConvAE": QI3Dto2DConvAE
}

