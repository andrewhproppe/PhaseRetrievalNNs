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

        # if self.current_epoch > 0 and self.current_epoch % self.hparams.plot_interval == 0 and self.epoch_plotted == False:
        #     self.epoch_plotted = True # don't plot again in this epoch
        #     with torch.no_grad():
        #         fig = self.plot_training_results(X, Y, pred_Y)
        #         log.update({"plot": fig})
        #         self.logger.experiment.log(log)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, log, X, Y, pred_Y = self.step(batch, batch_idx)
        self.log("validiation_loss", log)

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
        fig, ax = plt.subplots(ncols=3, nrows=1, dpi=150, figsize=(5, 2.5))
        X = X.cpu()
        Y = Y.cpu()
        pred_Y = pred_Y.cpu()

        ax[0].imshow(X[0, 0, :, :])
        ax[0].set_title('Input')
        ax[1].imshow(pred_Y[0, :, :])
        ax[1].set_title('Prediction')
        ax[2].imshow(Y[0, :, :])
        ax[2].set_title('Truth')

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
        if self.residual:
            out += residual
        else:
            residual = torch.zeros_like(residual)
        out = self.activation(out)
        return out, residual


class ResNet3D(nn.Module):
    def __init__(
            self,
            block: nn.Module = ResBlock3d,
            first_layer_args: dict = {'kernel': (7, 7, 7), 'stride': (2, 2, 2), 'padding': (3, 3, 3)},
            depth: int = 4,
            channels: list = [1, 64, 128, 256, 512],
            strides: list = [1, 1, 1, 1, 1],
            layers: list = [1, 1, 1, 1],
            dropout: list = [0., 0., 0., 0.],
            activation: nn.Module = nn.ReLU,
            residual: bool = True,
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
            self.layers[str(i)] = self._make_layer(block, channels[i+1], layers[i], kernel=(3, 3, 3), stride=strides[i], activation=activation, dropout=dropout[i], residual=residual)

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
            layers.append(block(self.inplanes, planes, kernel, stride, downsample, activation, dropout, residual))

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
            last_layer_args: dict = {'kernel': (7, 7), 'stride': (2, 2), 'padding': (3, 3, 3)},
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
            res = residuals[-1-i].mean(axis=2)
            x = x + res
            x = self.layers[str(i)](x)
        x = self.conv_out(x)
        return x


class SRN3D(QIAutoEncoder):
    """ Symmetric Resnet 3D-to-2D Convolutional Autoencoder """
    def __init__(
        self,
        depth: int = 4,
        first_layer_args={'kernel': (9, 7, 7), 'stride': (6, 2, 2), 'padding': (0, 3, 3)},
        last_layer_args={'kernel': (7, 7), 'stride': (2, 2), 'padding': (3, 3)},
        channels: list = [1, 4, 8, 16, 32, 64],
        strides: list = [2, 2, 2, 1, 2, 1],
        layers: list = [1, 1, 1, 1, 1],
        residual: bool = True,
        fwd_skip: bool = False,
        sym_skip: bool = True,
        dropout: float = [0., 0., 0., 0., 0.,],
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        plot_interval=50,
    ):
        super().__init__(lr, weight_decay, plot_interval)

        self.encoder = ResNet3D(
            block=ResBlock3d,
            first_layer_args=first_layer_args,
            depth=depth,
            channels=channels[0:depth+1],
            strides=strides[0:depth],
            layers=layers[0:depth],
            dropout=dropout[0:depth],
            residual=residual
        )

        self.decoder = DeconvNet2D(
            block=DeconvBlock2d,
            last_layer_args=last_layer_args,
            depth=depth,
            channels=list(reversed(channels[0:depth+1])),
            strides=list(reversed(strides[0:depth])),
            layers=list(reversed(layers[0:depth]))
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



models = {
    "QI3Dto2DConvAE": QI3Dto2DConvAE,
    "SRN3D": SRN3D
}

""" For testing """
if __name__ == '__main__':

    from QIML.pipeline.QI_data import QIDataModule

    # data_fname = 'QIML_data_n100_nbar10000_nframes16_npix32.h5'
    # data_fname = 'QIML_data_n100_nbar10000_nframes16_npix32.h5'
    data_fname = 'QIML_data_n1000_nbar10000_nframes32_npix32.h5'
    data = QIDataModule(data_fname, batch_size=100)
    data.setup()

    # Loop to generate a batch of data taken from dataset
    for i in range(0, 12):
        if i == 0:
            X, _ = data.train_set.__getitem__(0)
            X = X.unsqueeze(0)
        else:
            Xtemp, _ = data.train_set.__getitem__(0)
            Xtemp = Xtemp.unsqueeze(0)
            X = torch.cat((X, Xtemp), dim=0)

    model = SRN3D(
        depth=4,
        first_layer_args={'kernel': (9, 7, 7), 'stride': (6, 2, 2), 'padding': (0, 3, 3)},
        channels=[1, 4, 8, 16, 32, 64],
        strides=[2, 2, 1, 1, 1, 1],
        layers=[1, 1, 1, 1, 1],
        plot_interval=10
    )

    # some shape tests before trying to actually train
    z, res = model.encoder(X.unsqueeze(1))
    # out = model(X)[0]
    print(z.shape)

    # raise RuntimeError

    from pytorch_lightning.loggers import WandbLogger

    # decide to train on GPU or CPU based on availability or user specified
    if not torch.cuda.is_available():
        GPU = 0
    else:
        GPU = 1

    logger = WandbLogger(
        entity="aproppe",
        project="SRN3D",
        log_model=False,
        save_code=False,
        offline=True,
    )

    trainer = pl.Trainer(
        max_epochs=100,
        gpus=int(torch.cuda.is_available()),
        logger=logger,
        checkpoint_callback=False,
        enable_checkpointing=False,
    )

    trainer.fit(model, data)

