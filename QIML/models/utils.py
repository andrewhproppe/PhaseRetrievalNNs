import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from functools import wraps


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


def phase_loss(pred, truth):
    # pred = (pred*2*torch.pi) - torch.pi
    truth = truth * 2 * torch.pi
    return F.mse_loss(pred, torch.cos(truth))


class BetaRateScheduler:
    def __init__(
        self,
        initial_beta: float = 0.0,
        end_beta: float = 4.0,
        cap_steps: int = 4000,
        hold_steps: int = 2000,
        log_steps: bool = False,
    ):
        self._initial_beta = initial_beta
        self._end_beta = end_beta
        self._cap_steps = cap_steps
        self._hold_steps = hold_steps
        self._log_steps = log_steps
        self.reset()

    @property
    def current_step(self):
        return self._current_step

    @current_step.setter
    def current_step(self, value: int):
        assert value >= 0
        self._current_step = value

    def reset(self):
        self.current_step = 0

    def __iter__(self):
        return self.beta()

    def beta(self):
        """
        Returns a generator that yields the next value of beta
        according to the scheduler. In the current implementation,
        the scheduler corresponds to a linear ramp up to `cap_steps`
        and subsequently holds the value of `end_beta` for another
        `hold_steps`. Once this is done, the value of `beta` is
        set back to zero, and the cycle begins anew.

        Yields
        -------
        float
            Value of beta at the current global step
        """
        if self._log_steps:
            cap = np.logspace(self._initial_beta, self._end_beta, self._cap_steps)
        else:
            cap = np.linspace(self._initial_beta, self._end_beta, self._cap_steps)
        hold = np.array([self._end_beta for _ in range(self._hold_steps)])

        beta_values = np.concatenate(
            [
                np.linspace(self._initial_beta, self._end_beta, self._cap_steps),
                np.array([self._end_beta for _ in range(self._hold_steps)]),
            ]
        )
        while self.current_step < self._cap_steps + self._hold_steps:
            self.current_step = self.current_step + 1
            yield beta_values[self.current_step - 1]
        self.reset()


def get_encoded_size(data, model):
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

    # some shape tests before trying to actually train
    z, res = model.encoder(X.unsqueeze(1))
    out = model(X.unsqueeze(1))
    return z, res, out


def calculate_layer_sizes(input_size, strides, depth):
    sizes = [input_size]
    size = input_size
    for s in strides[:depth]:
        size = size // s
        sizes.append(size)
    return sizes


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = self._build_feature_extractor()

        # We don't want to train the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss()

    def forward(self, input, target):
        input = input.unsqueeze(1)  # Add a channel dimension
        target = target.unsqueeze(1)
        input = input.repeat(1, 3, 1, 1)  # Convert single-channel input to 3-channel
        target = target.repeat(1, 3, 1, 1)

        input_features = self.feature_extractor(input)
        target_features = self.feature_extractor(target)
        loss = 0

        for input_feat, target_feat in zip(input_features, target_features):
            loss += self.criterion(input_feat, target_feat)

        return loss

    def _build_feature_extractor(self):
        vgg = models.vgg19(pretrained=True).features
        feature_extractor = nn.Sequential()

        for i, module in enumerate(vgg.children()):
            if isinstance(module, nn.Conv2d):
                feature_extractor.add_module(f"conv_{i}", module)
            if isinstance(module, nn.ReLU):
                feature_extractor.add_module(f"relu_{i}", module)
            if isinstance(module, nn.MaxPool2d):
                feature_extractor.add_module(f"pool_{i}", module)

        return feature_extractor


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(
                input, mode="bilinear", size=(224, 224), align_corners=False
            )
            target = self.transform(
                target, mode="bilinear", size=(224, 224), align_corners=False
            )
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


""" SSIM functions from https://github.com/Po-Hsun-Su/pytorch-ssim/tree/master """


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


class GradientDifferenceLoss(nn.Module):
    def __init__(self):
        super(GradientDifferenceLoss, self).__init__()

    def forward(self, output, target):
        target_gradient_x = F.conv2d(target, torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0), padding=1)
        target_gradient_y = F.conv2d(target, torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0), padding=1)
        output_gradient_x = F.conv2d(output, torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0), padding=1)
        output_gradient_y = F.conv2d(output, torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0), padding=1)

        gradient_diff_x = torch.abs(target_gradient_x - output_gradient_x)
        gradient_diff_y = torch.abs(target_gradient_y - output_gradient_y)

        gradient_diff_loss = gradient_diff_x.mean() + gradient_diff_y.mean()

        return gradient_diff_loss

