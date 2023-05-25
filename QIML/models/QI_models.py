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

# mpl.use("TkAgg")  # this forces a non-X server backend
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
            output = output+data
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
            output = output+data
        # normalize the result to prevent exploding values
        output = self.norm(output)
        return output


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
        self.log("train_loss", loss, prog_bar=True)

        # if self.current_epoch > 0 and self.current_epoch % self.hparams.plot_interval == 0 and self.epoch_plotted == False:
        #     self.epoch_plotted = True # don't plot again in this epoch
        #     with torch.no_grad():
        #         fig = self.plot_training_results(X, Y, pred_Y)
        #         log.update({"plot": fig})
        #         self.logger.experiment.log(log)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, log, X, Y, pred_Y = self.step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)

        if self.current_epoch > 0 and self.current_epoch % self.hparams.plot_interval == 0 and self.epoch_plotted == False:
            self.epoch_plotted = True # don't plot again in this epoch
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

        if X.ndim == 4: # frames x Xpix x Ypix
            fig, ax = plt.subplots(ncols=3, nrows=1, dpi=150, figsize=(5, 2.5))
            idx = random.randint(0, Y.shape[0]-1)
            frame_idx = random.randint(0, X.shape[1]-1)
            ax[0].imshow(X[idx, frame_idx, :, :])
            ax[0].set_title('Input')
            ax[1].imshow(pred_Y[idx, :, :])
            ax[1].set_title('Prediction')
            ax[2].imshow(Y[idx, :, :])
            ax[2].set_title('Truth')
            dress_fig(tight=True, xlabel='x pixels', ylabel='y pixels', legend=False)
        elif X.ndim == 3: # correlation matrix
            fig, ax = plt.subplots(ncols=2, nrows=1, dpi=150, figsize=(5, 2.5))
            idx = random.randint(0, Y.shape[0]-1)
            ax[0].imshow(pred_Y[idx, :, :])
            ax[0].set_title('Prediction')
            ax[1].imshow(Y[idx, :, :])
            ax[1].set_title('Truth')
            dress_fig(tight=True, xlabel='x pixels', ylabel='y pixels', legend=False)

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


class ResBlock2d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel=(3, 3),
            stride=1,
            downsample=None,
            activation: Optional[Type[nn.Module]] = nn.ReLU,
            dropout=0.,
            residual: bool = True
    ) -> None:
        super(ResBlock2d, self).__init__()

        self.residual = residual
        self.activation = nn.Identity() if activation is None else activation()
        padding = tuple(k//2 for k in kernel)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            self.activation
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout)
        )
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
        if self.residual: # forward skip connection
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
            dropout=0.,
            residual: bool = True
    ) -> None:
        super(ResBlock3d, self).__init__()

        self.residual = residual
        self.activation = nn.Identity() if activation is None else activation()
        padding = tuple(k//2 for k in kernel)

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            self.activation
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel, stride=1, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.Dropout(dropout)
        )
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
        if self.residual: # forward skip connection
            out += residual
        out = self.activation(out)
        return out, residual


class ResNet2D(nn.Module):
    def __init__(
            self,
            block: nn.Module = ResBlock2d,
            first_layer_args: dict = {'kernel': (7, 7), 'stride': (2, 2), 'padding': (3, 3)},
            depth: int = 4,
            channels: list = [1, 64, 128, 256, 512],
            strides: list = [1, 1, 1, 1, 1],
            layers: list = [1, 1, 1, 1],
            dropout: list = [0., 0., 0., 0.],
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
                kernel_size=first_layer_args['kernel'],
                stride=first_layer_args['stride'],
                padding=first_layer_args['padding']
                # padding=tuple(k//2 for k in first_layer_args['kernel'])
            ),
            nn.BatchNorm2d(channels[1]),
            activation()
        )

        self.layers = nn.ModuleDict({})
        for i in range(0, self.depth):
            self.layers[str(i)] = self._make_layer(block, channels[i+1], layers[i], kernel=(3, 3), stride=strides[i], activation=activation, dropout=dropout[i], residual=residual)

    def _make_layer(self, block, planes, blocks, kernel, stride, activation, dropout, residual):
        """ Modified from Nourman (https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/) """
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, kernel, stride, downsample, activation, dropout, residual))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel, 1, None, activation, dropout, residual))

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
            first_layer_args: dict = {'kernel': (7, 7, 7), 'stride': (2, 2, 2), 'padding': (3, 3, 3)},
            depth: int = 4,
            channels: list = [1, 64, 128, 256, 512],
            pixel_strides: list = [1, 1, 1, 1, 1],
            frame_strides: list = [1, 1, 1, 1, 1],
            layers: list = [1, 1, 1, 1],
            dropout: list = [0., 0., 0., 0.],
            activation: nn.Module = nn.ReLU,
            residual: bool = False,
    ) -> None:
        super(ResNet3D, self).__init__()
        self.depth = depth
        self.inplanes = channels[1]

        # First layer with different kernel and stride; gives some extra control over frame dimension versus image XY dimensions
        self.conv_in = nn.Sequential(
            nn.Conv3d(
                channels[0],
                channels[1],
                kernel_size=first_layer_args['kernel'],
                stride=first_layer_args['stride'],
                padding=first_layer_args['padding']
                # padding=tuple(k//2 for k in first_layer_args['kernel'])
            ),
            nn.BatchNorm3d(channels[1]),
            activation()
        )

        self.layers = nn.ModuleDict({})
        for i in range(0, self.depth):
            _stride = (frame_strides[i], pixel_strides[i], pixel_strides[i])
            self.layers[str(i)] = self._make_layer(block, channels[i+1], layers[i], kernel=(3, 3, 3), stride=_stride, activation=activation, dropout=dropout[i], residual=residual)

    def _make_layer(self, block, planes, blocks, kernel, stride, activation, dropout, residual):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm3d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, kernel, stride, downsample, activation, dropout, residual))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel, 1, None, activation, dropout, residual))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_in(x)
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
            # nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel, stride=1, padding=padding, output_padding=padding-1),
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel, stride=1, padding=padding, output_padding=0),
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
            block: nn.Module = DeconvBlock2d,
            last_layer_args: dict = {'kernel': (7, 7), 'stride': (2, 2), 'padding': (3, 3, 3)},
            depth: int = 4,
            channels: list = [512, 256, 128, 64, 1],
            strides: list = [1, 1, 1, 1, 1],
            layers: list = [1, 1, 1, 1],
            activation: nn.Module = nn.ReLU,
            residual: bool = True,
    ) -> None:
        super(DeconvNet2D, self).__init__()
        self.depth = depth
        self.inplanes = channels[0]
        self.residual = residual
        self.layers = nn.ModuleDict({})

        for i in range(0, self.depth):
            if i == self.depth-1:
                self.layers[str(i)] = self._make_layer(block, channels[i], layers[i], stride=strides[i], activation=activation)
            else:
                self.layers[str(i)] = self._make_layer(block, channels[i+1], layers[i], stride=strides[i], activation=activation)

        # Mirrors conv_in of encoder
        self.conv_out = nn.Sequential(
            nn.ConvTranspose2d(
                channels[-2],
                channels[-1],
                kernel_size=last_layer_args['kernel'],
                stride=last_layer_args['stride'],
                padding=last_layer_args['padding'],
                # padding=tuple(k//2 for k in last_layer_args['kernel']),
                output_padding=tuple(s-1 for s in last_layer_args['stride'])
            ),
            nn.BatchNorm2d(channels[-1]),
            activation()
        )

    def _make_layer(self, block, planes, blocks, kernel=3, stride=1, activation=nn.ReLU):
        layers = []
        layers.append(block(self.inplanes, planes, kernel, stride, activation))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel, stride, activation))

        return nn.Sequential(*layers)

    def forward(self, x, residuals):
        for i in range(0, self.depth):
            res = residuals[-1-i]
            if res.ndim > x.ndim: # for 3D to 2D
                res = torch.mean(res, dim=2)
            if res.shape != x.shape: # for 2D to 2D with correlation matrix
                res = F.interpolate(res, size=x.shape[2:], mode='bilinear', align_corners=True)
            if self.residual: # symmetric skip connection
                x = x + res
            x = self.layers[str(i)](x)
        x = self.conv_out(x)
        return x


class SRN2D(QIAutoEncoder):
    """ Symmetric Resnet 2D-to-2D Convolutional Autoencoder """
    def __init__(
        self,
        depth: int = 4,
        first_layer_args={'kernel': (7, 7), 'stride': (2, 2), 'padding': (3, 3)},
        channels: list = [1, 4, 8, 16, 32, 64],
        strides: list = [2, 2, 2, 1, 2, 1],
        layers: list = [1, 1, 1, 1, 1],
        fwd_skip: bool = False,
        sym_skip: bool = True,
        dropout: float = [0., 0., 0., 0., 0.,],
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
            channels=channels[0:depth+1],
            strides=strides[0:depth],
            layers=layers[0:depth],
            dropout=dropout[0:depth],
            activation=enc_activation,
            residual=fwd_skip,
        )

        last_layer_args = {
            'kernel': first_layer_args['kernel'],
            'stride': tuple(np.sqrt(np.array(first_layer_args['stride'])).astype(int)), # quadratically smaller stride than encoder
            'padding': first_layer_args['padding']
        }

        self.decoder = DeconvNet2D(
            block=DeconvBlock2d,
            last_layer_args=last_layer_args,
            depth=depth,
            channels=list(reversed(channels[0:depth+1])),
            strides=list(reversed(np.sqrt(strides[0:depth]).astype(int))), # quadratically smaller stride than encoder
            layers=list(reversed(layers[0:depth])),
            activation=dec_activation,
            residual=sym_skip
        )

    def forward(self, X: torch.Tensor):
        if X.ndim < 4:
            X = X.unsqueeze(1)  # adds the channel dimension
        Z, res = self.encoder(X)
        D = self.decoder(Z, res)
        D = D.squeeze(1)  # removes the channel dimension
        return D, Z


class SRN3D(QIAutoEncoder):
    """ Symmetric Resnet 3D-to-2D Convolutional Autoencoder """
    def __init__(
        self,
        depth: int = 4,
        first_layer_args={'kernel': (9, 7, 7), 'stride': (6, 2, 2), 'padding': (0, 3, 3)},
        channels: list = [1, 4, 8, 16, 32, 64],
        pixel_strides: list = [2, 2, 2, 1, 2, 1],
        frame_strides: list = [2, 2, 2, 1, 2, 1],
        layers: list = [1, 1, 1, 1, 1],
        fwd_skip: bool = False,
        sym_skip: bool = True,
        dropout: float = [0., 0., 0., 0., 0.,],
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        plot_interval=50,
    ) -> None:
        """

        Returns
        -------
        object
        """
        super().__init__(lr, weight_decay, plot_interval)

        self.encoder = ResNet3D(
            block=ResBlock3d,
            first_layer_args=first_layer_args,
            depth=depth,
            channels=channels[0:depth+1],
            pixel_strides=pixel_strides[0:depth],
            frame_strides=frame_strides[0:depth],
            layers=layers[0:depth],
            dropout=dropout[0:depth],
            residual=fwd_skip,
        )

        # Remove first frame dimension from
        last_layer_args = dict((k, v[1:]) for k, v in first_layer_args.items())

        self.decoder = DeconvNet2D(
            block=DeconvBlock2d,
            last_layer_args=last_layer_args,
            depth=depth,
            channels=list(reversed(channels[0:depth+1])),
            strides=list(reversed(pixel_strides[0:depth])),
            layers=list(reversed(layers[0:depth])),
            residual=sym_skip
        )


    def forward(self, X: torch.Tensor):
        if X.ndim < 5:
            X = X.unsqueeze(1)  # adds the channel dimension
        Z, res = self.encoder(X)
        if Z.shape[2] > 1:
            print('Latent shape needs to be compressed down to 1')
            raise RuntimeError
        D = self.decoder(Z.squeeze(2), res)
        D = D.squeeze(1)  # removes the channel dimension
        return D, Z


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
            dropout=0.,
            residual: bool = True
    ) -> None:
        super(ResBlock2D, self).__init__()

        self.residual = residual
        self.activation = nn.Identity() if activation is None else activation()
        if isinstance(kernel, int):
            padding = kernel//2
        else:
            padding = tuple(k//2 for k in kernel)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            self.activation
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout)
        )
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
        if self.residual: # forward skip connection
            out += residual
        out = self.activation(out)
        return out, residual


class MultiScaleCNN(pl.LightningModule):
    def __init__(
            self,
            first_layer_args={'kernel_size': (7, 7), 'stride': (2, 2), 'padding': (3, 3)},
            nbranch: int = 3,
            branch_depth: int = 1,
            kernels: list = [3, 5, 7],
            channels: list = [4, 8, 16, 32, 64],
            strides: list = [2, 2, 2, 2, 2, 2],
            dilations: list = [1, 1, 1, 1, 1, 1],
            activation: nn.Module = nn.ReLU,
            residual: bool = False,
    ) -> None:
        super(MultiScaleCNN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(1, channels[0], **first_layer_args)
        self.actv1 = activation()

        self.branches = nn.ModuleList([])
        for i in range(0, nbranch):
            self.inchannels = channels[0]
            branch_layers = self._make_branch(branch_depth, channels, kernels[i], strides, dilations[i], activation, residual)
            self.branches.append(branch_layers)

        # Final convolutional layer for concatenated branch outputs
        self.conv3 = nn.Conv2d(
            nbranch*channels[branch_depth-1], # number of channels in concatenated branch outputs
            channels[branch_depth],
            kernel_size=3,
            stride=2,
            padding=1
        )

        # self.save_hyperparameters()

    def _make_branch(self, branch_depth, channels, kernel, strides, dilation, activation, residual):
        layers = []
        for i in range(0, branch_depth):
            layers.append(self._make_layer(channels[i], kernel=kernel, stride=strides[i], dilation=dilation, activation=activation, residual=residual))
        return nn.Sequential(*layers)


    def _make_layer(self, channels, kernel, stride, dilation, activation, residual):
        """ Modified from Nourman (https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/) """
        downsample = None
        if stride != 1 or self.inchannels != channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.inchannels, channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(channels),
            )
        layer = ResBlock2D(self.inchannels, channels, kernel, stride, dilation, downsample, activation, residual=residual)
        self.inchannels = channels
        # layers = []
        # layers.append(ResBlock2D(self.inchannels, channels, kernel, stride, downsample, activation, residual=residual))
        # return nn.Sequential(*layers)
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


class DeconvolutionNetwork(nn.Module):
    def __init__(
            self,
            channels: list = [1, 16, 32, 64, 128],
            depth: int = 2,
            kernel_size: int = 3,
            stride: int = 2,
            activation: nn.Module = nn.ReLU
    ):
        super(DeconvolutionNetwork, self).__init__()

        activation = nn.Identity if activation is None else activation
        layers = []
        layers.append(DeconvBlock2d(channels[0], channels[1], kernel_size, stride, activation))
        for i in range(1, depth-1):
            layers.append(DeconvBlock2d(channels[i], channels[i+1], kernel_size, stride, activation))
        layers.append(DeconvBlock2d(channels[depth-1], 1, kernel_size, stride, activation))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x).squeeze(1)


class MSRN2D(QIAutoEncoder):
    def __init__(
        self,
        encoder_args,
        decoder_args,
        z_size: int = 64,
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        plot_interval=50,
        init_lazy: bool = True # Set to false when testing encoded and decoded shapes; true for training
    ) -> None:
        super().__init__(lr, weight_decay, plot_interval)

        self.encoder = MultiScaleCNN(**encoder_args)
        self.decoder = DeconvolutionNetwork(**decoder_args)

        ## For a flattened bottleneck:
        # self.flatten = nn.Flatten()
        # self.linear1 = nn.LazyLinear(z_size)
        # self.reshape = Reshape(-1, 1, int(np.sqrt(z_size)), int(np.sqrt(z_size)))

        if init_lazy:
            self.initialize_lazy((2, 1, 1024, 1024))

    def forward(self, X: torch.Tensor):
        Z = self.encode(X)
        # Z = self.flatten(Z)
        # Z = self.linear1(Z)
        # Z = self.reshape(Z)
        del X # helps with memory allocation
        return self.decoder(Z), 1 # return a dummy Z, reduce memory load

""" For testing """
if __name__ == '__main__':

    from QIML.pipeline.QI_data import QIDataModule
    from data.utils import get_batch_from_dataset

    # data_fname = 'QIML_data_n1000_nbar10000_nframes32_npix32.h5'
    # data_fname = 'QIML_poisson_testset.h5'
    data_fname = 'QIML_mnist_data_n10_npix32.h5'

    data = QIDataModule(data_fname, batch_size=8, num_workers=0, nbar=1e4, nframes=64, corr_matrix=True)
    data.setup()
    batch = next(iter(data.train_dataloader()))
    X = batch[0]

    # Multiscale resnet using correlation matrix
    encoder_args = {
        'first_layer_args': {'kernel_size': (7, 7), 'stride': (2, 2), 'padding': (3, 3)},
        'nbranch': 3,
        'branch_depth': 5,
        'kernels': [3, 7, 21, 28, 56],
        'channels': [4, 8, 16, 32, 64],
        'strides': [2, 2, 2, 2, 2, 2],
        'dilations': [1, 2, 3, 4, 2, 2],
        'activation': nn.ReLU,
        'residual': False,
    }

    # Deconv decoder
    decoder_args = {
        'depth': 2
    }

    model = MSRN2D(
        encoder_args,
        decoder_args
    )

    from pytorch_lightning.loggers import WandbLogger

    logger = WandbLogger(
        entity="aproppe",
        project="MSRN2D",
        log_model=False,
        save_code=False,
        offline=True,
    )

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=logger,
        enable_checkpointing=False,
    )

    trainer.fit(model, data)

# Old
    # # For 2D to 2D
    # model = SRN2D(
    #     first_layer_args={'kernel': (7, 7), 'stride': (4, 4), 'padding': (2, 2)},
    #     depth=4,
    #     # channels=[1, 32, 64, 128, 256, 512],
    #     channels=[1, 16, 32, 64, 128, 256],
    #     strides=[4, 2, 2, 2, 1],
    #     layers=[1, 1, 1, 1, 1],
    #     dropout=[0.1, 0.1, 0.2, 0.3],
    #     lr=5e-4,
    #     weight_decay=1e-4,
    #     fwd_skip=True,
    #     sym_skip=True,
    #     plot_interval=5,  # training
    # )