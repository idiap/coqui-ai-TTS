"""Common layers for StyleTTS 2."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class UpSample1d(nn.Module):
    def __init__(self, upsample: bool = False):
        super().__init__()
        self.upsample = upsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            return F.interpolate(x, scale_factor=2, mode="nearest")
        else:
            return x


class AdaIN1d(nn.Module):
    """Adaptive instance normalization for 1D signals."""

    def __init__(self, style_dim: int, num_features: int):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdaINResBlk1d(nn.Module):
    """Adaptive Instance Normalization Residual Block 1D with upsampling."""

    def __init__(self, dim_in: int, dim_out: int, style_dim: int = 128, upsample: bool = False, dropout_p: float = 0.0):
        super().__init__()
        self.actv = nn.LeakyReLU(0.2)
        self.do_upsample = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self.dropout = nn.Dropout(dropout_p)

        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)

        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

        if upsample:
            self.pool = weight_norm(
                nn.ConvTranspose1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1, output_padding=1)
            )
        else:
            self.pool = nn.Identity()

    def _shortcut(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out
