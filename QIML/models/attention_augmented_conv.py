"""
from https://github.com/leaderj1001/Attention-Augmented-Conv2d
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class AugmentedConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dk,
        dv,
        Nh,
        shape=0,
        relative=False,
        stride=1,
    ):
        super(AugmentedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.shape = shape
        self.relative = relative
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert (
            self.dk % self.Nh == 0
        ), "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert (
            self.dv % self.Nh == 0
        ), "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.conv_out = nn.Conv2d(
            self.in_channels,
            self.out_channels - self.dv,
            self.kernel_size,
            stride=stride,
            padding=self.padding,
        )

        self.qkv_conv = nn.Conv2d(
            self.in_channels,
            2 * self.dk + self.dv,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=self.padding,
        )

        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

        if self.relative:
            self.key_rel_w = nn.Parameter(
                torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True)
            )
            self.key_rel_h = nn.Parameter(
                torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True)
            )

    def forward(self, x):
        # Input x
        # (batch_size, channels, height, width)
        # batch, _, height, width = x.size()

        # conv_out
        # (batch_size, out_channels, height, width)
        conv_out = self.conv_out(x)
        batch, _, height, width = conv_out.size()

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, height * width, dvh or dkh)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v
        # (batch_size, Nh, height, width, dv or dk)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(
            x, self.dk, self.dv, self.Nh
        )
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits
        weights = F.softmax(logits, dim=-1)

        # attn_out
        # (batch, Nh, height * width, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(
            attn_out, (batch, self.Nh, self.dv // self.Nh, height, width)
        )
        # combine_heads_2d
        # (batch, out_channels, height, width)
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)
        return torch.cat((conv_out, attn_out), dim=1)

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        N, _, H, W = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        # q *= dkh ** -0.5
        q = q * (dkh ** -0.5)
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, H, W, Nh, "w")
        rel_logits_h = self.relative_logits_1d(
            torch.transpose(q, 2, 3), self.key_rel_h, W, H, Nh, "h"
        )

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum("bhxyd,md->bhxym", q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = (
                torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
            )
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1 :]
        return final_x


class AugmentedConv3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dk,
        dv,
        Nh,
        shape=0,
        relative=False,
        stride=1,
    ):
        super(AugmentedConv3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.shape = shape
        self.relative = relative
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert (
            self.dk % self.Nh == 0
        ), "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert (
            self.dv % self.Nh == 0
        ), "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.conv_out = nn.Conv3d(
            self.in_channels,
            self.out_channels - self.dv,
            self.kernel_size,
            stride=stride,
            padding=self.padding,
        )

        self.qkv_conv = nn.Conv3d(
            self.in_channels,
            2 * self.dk + self.dv,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=self.padding,
        )

        self.attn_out = nn.Conv3d(self.dv, self.dv, kernel_size=1, stride=1)

        if self.relative:
            self.key_rel_w = nn.Parameter(
                torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True)
            )
            self.key_rel_h = nn.Parameter(
                torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True)
            )
            self.key_rel_d = nn.Parameter(
                torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True)
            )

    def forward(self, x):
        conv_out = self.conv_out(x)
        batch, _, depth, height, width = conv_out.size()

        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(
            x, self.dk, self.dv, self.Nh
        )
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        if self.relative:
            h_rel_logits, w_rel_logits, d_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits
            logits += d_rel_logits
        weights = F.softmax(logits, dim=-1)

        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(
            attn_out, (batch, self.Nh, self.dv // self.Nh, depth, height, width)
        )
        attn_out = self.combine_heads_3d(attn_out)
        attn_out = self.attn_out(attn_out)
        return torch.cat((conv_out, attn_out), dim=1)

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        N, _, D, H, W = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_3d(q, Nh)
        k = self.split_heads_3d(k, Nh)
        v = self.split_heads_3d(v, Nh)

        dkh = dk // Nh
        q = q * (dkh ** -0.5)
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, D * H * W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, D * H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, D * H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_3d(self, x, Nh):
        batch, channels, depth, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, depth, height, width)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_3d(self, x):
        batch, Nh, dv, D, H, W = x.size()
        ret_shape = (batch, Nh * dv, D, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, D, H, W = q.size()
        q = torch.transpose(q, 2, 5).transpose(2, 4).transpose(2, 3)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, D, H, W, Nh, "w")
        rel_logits_h = self.relative_logits_1d(
            torch.transpose(q, 2, 3), self.key_rel_h, D, W, H, Nh, "h"
        )
        rel_logits_d = self.relative_logits_1d(
            torch.transpose(q, 2, 4), self.key_rel_d, H, W, D, Nh, "d"
        )

        return rel_logits_h, rel_logits_w, rel_logits_d

    def relative_logits_1d(self, q, rel_k, L, M, N, Nh, case):
        rel_logits = torch.einsum("bhwmd,md->bhw", q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * L * M, N, 2 * N - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, L, M, N, N))
        rel_logits = torch.unsqueeze(rel_logits, dim=4)
        rel_logits = rel_logits.repeat((1, 1, 1, 1, M, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 4, 5)
        elif case == "h":
            rel_logits = (
                torch.transpose(rel_logits, 3, 5).transpose(5, 6).transpose(4, 6)
            )
        elif case == "d":
            rel_logits = (
                torch.transpose(rel_logits, 2, 5).transpose(5, 6).transpose(4, 6)
            )
        rel_logits = torch.reshape(rel_logits, (-1, Nh, L * M * N, L * M * N))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, M, N = x.size()

        col_pad = torch.zeros((B, Nh, L, M, 1)).to(x)
        x = torch.cat((x, col_pad), dim=4)

        flat_x = torch.reshape(x, (B, Nh, L * M * 2 * N))
        flat_pad = torch.zeros((B, Nh, L * M - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L * M + 1, 2 * N - 1))
        final_x = final_x[:, :, : L * M, M - 1 :]
        return final_x


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    in_channels = 4
    npix = 16
    stride = 2

    tmp = torch.randn((2, in_channels, npix, npix)).to(device)
    augmented_conv1 = AugmentedConv(
        in_channels=in_channels,
        out_channels=32,
        kernel_size=3,
        dk=4,
        dv=4,
        Nh=2,
        relative=True,
        stride=stride,
        shape=npix // stride,
    ).to(device)
    conv_out1 = augmented_conv1(tmp)
    print(conv_out1.shape)

    tmp2 = torch.randn((2, in_channels, npix, npix, npix)).to(device)
    augmented_conv2 = AugmentedConv3D(
        in_channels=in_channels,
        out_channels=32,
        kernel_size=3,
        dk=4,
        dv=4,
        Nh=2,
        relative=True,
        stride=stride,
        shape=npix // stride,
    ).to(device)

    conv_out2 = augmented_conv2(tmp2)
    # depth = 5
    # strides = [2, 2, 1, 1, 1, 1]
    # input_size = 64
    #
    #
    # sizes = calculate_layer_sizes(input_size, strides, depth)
