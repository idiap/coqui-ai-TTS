"""Style encoders for StyleTTS 2."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def _raise_unexpected_layer_type(layer_type: str) -> None:
    msg = f"Unexpected layer type `{layer_type}`, expected [none, timepreserve, half]"
    raise ValueError(msg)


class DownSample(nn.Module):
    """Downsampling layer for StyleTTS 2."""

    def __init__(self, layer_type: str):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_type == "none":
            return x
        elif self.layer_type == "timepreserve":
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == "half":
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool2d(x, 2)
        else:
            _raise_unexpected_layer_type(self.layer_type)


class LearnedDownSample(nn.Module):
    """Learned downsampling layer for StyleTTS 2."""

    def __init__(self, layer_type: str, dim_in: int):
        super().__init__()
        self.layer_type = layer_type

        if self.layer_type == "none":
            self.conv = nn.Identity()
        elif self.layer_type == "timepreserve":
            self.conv = spectral_norm(
                nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=(3, 1),
                    stride=(2, 1),
                    groups=dim_in,
                    padding=(1, 0),
                )
            )
        elif self.layer_type == "half":
            self.conv = spectral_norm(
                nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    groups=dim_in,
                    padding=1,
                )
            )
        else:
            _raise_unexpected_layer_type(self.layer_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResBlk(nn.Module):
    """Residual block for StyleTTS 2 style encoder."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        actv: nn.Module | None = None,
        normalize: bool = False,
        downsample: str = "none",
    ):
        super().__init__()
        self.actv = actv if actv is not None else nn.LeakyReLU(0.2)
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.downsample_res = LearnedDownSample(downsample, dim_in)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in: int, dim_out: int):
        """Build convolutional weights."""
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))

        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)

        if self.learned_sc:
            self.conv1x1 = spectral_norm(nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x: torch.Tensor) -> torch.Tensor:
        """Shortcut connection."""
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x: torch.Tensor) -> torch.Tensor:
        """Residual connection."""
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample_res(x)

        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class StyleEncoder(nn.Module):
    """Style encoder for StyleTTS 2.

    This encoder processes mel-spectrograms and produces style embeddings.
    """

    def __init__(
        self,
        style_dim: int = 128,
        dim_in: int = 64,
        max_conv_dim: int = 512,
        repeat_num: int = 4,
    ):
        super().__init__()
        blocks = []
        blocks.append(spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1)))

        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks.append(ResBlk(dim_in, dim_out, downsample="half"))
            dim_in = dim_out

        blocks.append(nn.LeakyReLU(0.2))
        blocks.append(spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0)))
        blocks.append(nn.AdaptiveAvgPool2d(1))
        blocks.append(nn.LeakyReLU(0.2))

        self.shared = nn.Sequential(*blocks)
        self.unshared = nn.Linear(dim_out, style_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input mel-spectrogram

        Returns:
            Style embedding [B, style_dim]
        """
        h = self.shared(x)  # [B, dim_out, 1, 1]
        h = h.view(h.size(0), -1)  # [B, dim_out]
        s = self.unshared(h)  # [B, style_dim]

        return s
