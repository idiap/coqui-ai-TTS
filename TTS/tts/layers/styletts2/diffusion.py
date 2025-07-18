"""Original StyleTTS2 diffusion implementation.

This module implements the exact diffusion architecture from the original StyleTTS2,
using AudioDiffusionConditional, KDiffusion, and StyleTransformer1d.
"""

import math
from collections.abc import Callable
from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from torch import Tensor, einsum

from TTS.tts.layers.styletts2.utils import rand_bool


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def to_batch(
    batch_size: int,
    device: torch.device,
    x: float | None = None,
    xs: Tensor | None = None,
) -> Tensor:
    assert exists(x) ^ exists(xs), "Either x or xs must be provided"
    # If x provided use the same for all batch items
    if exists(x):
        xs = torch.full(size=(batch_size,), fill_value=x).to(device)
    assert exists(xs)
    return xs


class Distribution:
    def __call__(self, num_samples: int, device: torch.device):
        raise NotImplementedError()


class LogNormalDistribution(Distribution):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(self, num_samples: int, device: torch.device | str = "cpu") -> Tensor:
        normal = self.mean + self.std * torch.randn((num_samples,), device=device)
        return normal.exp()


class UniformDistribution(Distribution):
    def __call__(self, num_samples: int, device: torch.device | str = "cpu"):
        return torch.rand(num_samples, device=device)


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)

        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)

        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)


class RelativePositionBias(nn.Module):
    def __init__(self, num_buckets: int, max_distance: int, num_heads: int):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    @staticmethod
    def _relative_position_bucket(relative_position: Tensor, num_buckets: int, max_distance: int):
        num_buckets //= 2
        ret = (relative_position >= 0).to(torch.long) * num_buckets
        n = torch.abs(relative_position)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).long()
        )
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, num_queries: int, num_keys: int) -> Tensor:
        i, j, device = num_queries, num_keys, self.relative_attention_bias.weight.device
        q_pos = torch.arange(j - i, j, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, "j -> 1 j") - rearrange(q_pos, "i -> i 1")

        relative_position_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance
        )

        bias = self.relative_attention_bias(relative_position_bucket)
        bias = rearrange(bias, "m n h -> 1 h m n")
        return bias


def FeedForward(features: int, multiplier: int) -> nn.Module:
    mid_features = features * multiplier
    return nn.Sequential(
        nn.Linear(in_features=features, out_features=mid_features),
        nn.GELU(),
        nn.Linear(in_features=mid_features, out_features=features),
    )


class AttentionBase(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        use_rel_pos: bool,
        out_features: int | None = None,
        rel_pos_num_buckets: int | None = None,
        rel_pos_max_distance: int | None = None,
    ):
        super().__init__()
        self.scale = head_features**-0.5
        self.num_heads = num_heads
        self.use_rel_pos = use_rel_pos
        mid_features = head_features * num_heads

        if use_rel_pos:
            assert exists(rel_pos_num_buckets) and exists(rel_pos_max_distance)
            self.rel_pos = RelativePositionBias(
                num_buckets=rel_pos_num_buckets,
                max_distance=rel_pos_max_distance,
                num_heads=num_heads,
            )
        if out_features is None:
            out_features = features

        self.to_out = nn.Linear(in_features=mid_features, out_features=out_features)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Split heads
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.num_heads) for t in (q, k, v))
        # Compute similarity matrix
        sim = einsum("... n d, ... m d -> ... n m", q, k)
        sim = (sim + self.rel_pos(*sim.shape[-2:])) if self.use_rel_pos else sim
        sim = sim * self.scale
        # Get attention matrix with softmax
        attn = sim.softmax(dim=-1)
        # Compute values
        out = einsum("... n m, ... m d -> ... n d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        out_features: int | None = None,
        context_features: int | None = None,
        use_rel_pos: bool,
        rel_pos_num_buckets: int | None = None,
        rel_pos_max_distance: int | None = None,
    ):
        super().__init__()
        self.context_features = context_features
        mid_features = head_features * num_heads
        context_features = default(context_features, features)

        self.norm = nn.LayerNorm(features)
        self.norm_context = nn.LayerNorm(context_features)
        self.to_q = nn.Linear(in_features=features, out_features=mid_features, bias=False)
        self.to_kv = nn.Linear(in_features=context_features, out_features=mid_features * 2, bias=False)

        self.attention = AttentionBase(
            features,
            out_features=out_features,
            num_heads=num_heads,
            head_features=head_features,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )

    def forward(self, x: Tensor, *, context: Tensor | None = None) -> Tensor:
        assert_message = "You must provide a context when using context_features"
        assert not self.context_features or exists(context), assert_message
        # Use context if provided
        context = default(context, x)
        # Normalize then compute q from input and k,v from context
        x, context = self.norm(x), self.norm_context(context)
        q, k, v = (self.to_q(x), *torch.chunk(self.to_kv(context), chunks=2, dim=-1))
        # Compute and return attention
        return self.attention(q, k, v)


class StyleAttention(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        style_dim: int,
        head_features: int,
        num_heads: int,
        context_features: int | None = None,
        use_rel_pos: bool,
        rel_pos_num_buckets: int | None = None,
        rel_pos_max_distance: int | None = None,
    ):
        super().__init__()
        self.context_features = context_features
        mid_features = head_features * num_heads
        context_features = default(context_features, features)

        self.norm = AdaLayerNorm(style_dim, features)
        self.norm_context = AdaLayerNorm(style_dim, context_features)
        self.to_q = nn.Linear(in_features=features, out_features=mid_features, bias=False)
        self.to_kv = nn.Linear(in_features=context_features, out_features=mid_features * 2, bias=False)
        self.attention = AttentionBase(
            features,
            num_heads=num_heads,
            head_features=head_features,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )

    def forward(self, x: Tensor, s: Tensor, *, context: Tensor | None = None) -> Tensor:
        assert_message = "You must provide a context when using context_features"
        assert not self.context_features or exists(context), assert_message
        # Use context if provided
        context = default(context, x)
        # Normalize then compute q from input and k,v from context
        x, context = self.norm(x, s), self.norm_context(context, s)

        q, k, v = (self.to_q(x), *torch.chunk(self.to_kv(context), chunks=2, dim=-1))
        # Compute and return attention
        return self.attention(q, k, v)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        features: int,
        num_heads: int,
        head_features: int,
        multiplier: int,
        use_rel_pos: bool,
        rel_pos_num_buckets: int | None = None,
        rel_pos_max_distance: int | None = None,
        context_features: int | None = None,
    ):
        super().__init__()

        self.use_cross_attention = exists(context_features) and context_features > 0

        self.attention = Attention(
            features=features,
            num_heads=num_heads,
            head_features=head_features,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )

        if self.use_cross_attention:
            self.cross_attention = Attention(
                features=features,
                num_heads=num_heads,
                head_features=head_features,
                context_features=context_features,
                use_rel_pos=use_rel_pos,
                rel_pos_num_buckets=rel_pos_num_buckets,
                rel_pos_max_distance=rel_pos_max_distance,
            )

        self.feed_forward = FeedForward(features=features, multiplier=multiplier)

    def forward(self, x: Tensor, *, context: Tensor | None = None) -> Tensor:
        x = self.attention(x) + x
        if self.use_cross_attention:
            x = self.cross_attention(x, context=context) + x
        x = self.feed_forward(x) + x
        return x


class StyleTransformerBlock(nn.Module):
    def __init__(
        self,
        features: int,
        num_heads: int,
        head_features: int,
        style_dim: int,
        multiplier: int,
        use_rel_pos: bool,
        rel_pos_num_buckets: int | None = None,
        rel_pos_max_distance: int | None = None,
        context_features: int | None = None,
    ):
        super().__init__()

        self.use_cross_attention = exists(context_features) and context_features > 0

        self.attention = StyleAttention(
            features=features,
            style_dim=style_dim,
            num_heads=num_heads,
            head_features=head_features,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )

        if self.use_cross_attention:
            self.cross_attention = StyleAttention(
                features=features,
                style_dim=style_dim,
                num_heads=num_heads,
                head_features=head_features,
                context_features=context_features,
                use_rel_pos=use_rel_pos,
                rel_pos_num_buckets=rel_pos_num_buckets,
                rel_pos_max_distance=rel_pos_max_distance,
            )

        self.feed_forward = FeedForward(features=features, multiplier=multiplier)

    def forward(self, x: Tensor, s: Tensor, *, context: Tensor | None = None) -> Tensor:
        x = self.attention(x, s) + x
        if self.use_cross_attention:
            x = self.cross_attention(x, s, context=context) + x
        x = self.feed_forward(x) + x
        return x


class Transformer1d(nn.Module):
    def __init__(
        self,
        num_layers: int,
        channels: int,
        num_heads: int,
        head_features: int,
        multiplier: int,
        use_context_time: bool = True,
        use_rel_pos: bool = False,
        context_features_multiplier: int = 1,
        rel_pos_num_buckets: int | None = None,
        rel_pos_max_distance: int | None = None,
        context_features: int | None = None,
        context_embedding_features: int | None = None,
        embedding_max_length: int = 512,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    features=channels + context_embedding_features,
                    head_features=head_features,
                    num_heads=num_heads,
                    multiplier=multiplier,
                    use_rel_pos=use_rel_pos,
                    rel_pos_num_buckets=rel_pos_num_buckets,
                    rel_pos_max_distance=rel_pos_max_distance,
                )
                for i in range(num_layers)
            ]
        )

        self.to_out = nn.Sequential(
            Rearrange("b t c -> b c t"),
            nn.Conv1d(
                in_channels=channels + context_embedding_features,
                out_channels=channels,
                kernel_size=1,
            ),
        )

        use_context_features = exists(context_features)
        self.use_context_features = use_context_features
        self.use_context_time = use_context_time

        if use_context_time or use_context_features:
            context_mapping_features = channels + context_embedding_features

            self.to_mapping = nn.Sequential(
                nn.Linear(context_mapping_features, context_mapping_features),
                nn.GELU(),
                nn.Linear(context_mapping_features, context_mapping_features),
                nn.GELU(),
            )

        if use_context_time:
            assert exists(context_mapping_features)
            self.to_time = nn.Sequential(
                TimePositionalEmbedding(dim=channels, out_features=context_mapping_features),
                nn.GELU(),
            )

        if use_context_features:
            assert exists(context_features) and exists(context_mapping_features)
            self.to_features = nn.Sequential(
                nn.Linear(in_features=context_features, out_features=context_mapping_features),
                nn.GELU(),
            )

        self.fixed_embedding = FixedEmbedding(max_length=embedding_max_length, features=context_embedding_features)

    def get_mapping(self, time: Tensor | None = None, features: Tensor | None = None) -> Tensor | None:
        """Combines context time features and features into mapping"""
        items, mapping = [], None
        # Compute time features
        if self.use_context_time:
            assert_message = "use_context_time=True but no time features provided"
            assert exists(time), assert_message
            items += [self.to_time(time)]
        # Compute features
        if self.use_context_features:
            assert_message = "context_features exists but no features provided"
            assert exists(features), assert_message
            items += [self.to_features(features)]

        # Compute joint mapping
        if self.use_context_time or self.use_context_features:
            mapping = reduce(torch.stack(items), "n b m -> b m", "sum")
            mapping = self.to_mapping(mapping)

        return mapping

    def run(self, x, time, embedding, features):
        mapping = self.get_mapping(time, features)
        x = torch.cat([x.expand(-1, embedding.size(1), -1), embedding], dim=-1)
        mapping = mapping.unsqueeze(1).expand(-1, embedding.size(1), -1)

        for block in self.blocks:
            x = x + mapping
            x = block(x)

        x = x.mean(dim=1).unsqueeze(1)
        x = self.to_out(x)
        x = x.transpose(-1, -2)

        return x

    def forward(
        self,
        x: Tensor,
        time: Tensor,
        embedding_mask_proba: float = 0.0,
        embedding: Tensor | None = None,
        features: Tensor | None = None,
        embedding_scale: float = 1.0,
    ) -> Tensor:
        b, device = embedding.shape[0], embedding.device
        fixed_embedding = self.fixed_embedding(embedding)
        if embedding_mask_proba > 0.0:
            # Randomly mask embedding
            batch_mask = rand_bool(shape=(b, 1, 1), proba=embedding_mask_proba, device=device)
            embedding = torch.where(batch_mask, fixed_embedding, embedding)

        if embedding_scale != 1.0:
            # Compute both normal and fixed embedding outputs
            out = self.run(x, time, embedding=embedding, features=features)
            out_masked = self.run(x, time, embedding=fixed_embedding, features=features)
            # Scale conditional output using classifier-free guidance
            return out_masked + (out - out_masked) * embedding_scale
        else:
            return self.run(x, time, embedding=embedding, features=features)

        return x


class LearnedPositionalEmbedding(nn.Module):
    """Used for continuous time"""

    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


def TimePositionalEmbedding(dim: int, out_features: int) -> nn.Module:
    return nn.Sequential(
        LearnedPositionalEmbedding(dim),
        nn.Linear(in_features=dim + 1, out_features=out_features),
    )


class FixedEmbedding(nn.Module):
    def __init__(self, max_length: int, features: int):
        super().__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(max_length, features)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, length, device = *x.shape[0:2], x.device
        assert_message = "Input sequence length must be <= max_length"
        assert length <= self.max_length, assert_message
        position = torch.arange(length, device=device)
        fixed_embedding = self.embedding(position)
        fixed_embedding = repeat(fixed_embedding, "n d -> b n d", b=batch_size)
        return fixed_embedding


class StyleTransformer1d(nn.Module):
    def __init__(
        self,
        num_layers: int,
        channels: int,
        num_heads: int,
        head_features: int,
        multiplier: int,
        use_context_time: bool = True,
        use_rel_pos: bool = False,
        context_features_multiplier: int = 1,
        rel_pos_num_buckets: int | None = None,
        rel_pos_max_distance: int | None = None,
        context_features: int | None = None,
        context_embedding_features: int | None = None,
        embedding_max_length: int = 512,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                StyleTransformerBlock(
                    features=channels + context_embedding_features,
                    head_features=head_features,
                    num_heads=num_heads,
                    multiplier=multiplier,
                    style_dim=context_features,
                    use_rel_pos=use_rel_pos,
                    rel_pos_num_buckets=rel_pos_num_buckets,
                    rel_pos_max_distance=rel_pos_max_distance,
                )
                for i in range(num_layers)
            ]
        )

        self.to_out = nn.Sequential(
            Rearrange("b t c -> b c t"),
            nn.Conv1d(
                in_channels=channels + context_embedding_features,
                out_channels=channels,
                kernel_size=1,
            ),
        )

        use_context_features = exists(context_features)
        self.use_context_features = use_context_features
        self.use_context_time = use_context_time

        if use_context_time or use_context_features:
            context_mapping_features = channels + context_embedding_features

            self.to_mapping = nn.Sequential(
                nn.Linear(context_mapping_features, context_mapping_features),
                nn.GELU(),
                nn.Linear(context_mapping_features, context_mapping_features),
                nn.GELU(),
            )

        if use_context_time:
            assert exists(context_mapping_features)
            self.to_time = nn.Sequential(
                TimePositionalEmbedding(dim=channels, out_features=context_mapping_features),
                nn.GELU(),
            )

        if use_context_features:
            assert exists(context_features) and exists(context_mapping_features)
            self.to_features = nn.Sequential(
                nn.Linear(in_features=context_features, out_features=context_mapping_features),
                nn.GELU(),
            )

        self.fixed_embedding = FixedEmbedding(max_length=embedding_max_length, features=context_embedding_features)

    def get_mapping(self, time: Tensor | None = None, features: Tensor | None = None) -> Tensor | None:
        """Combines context time features and features into mapping"""
        items, mapping = [], None
        # Compute time features
        if self.use_context_time:
            assert_message = "use_context_time=True but no time features provided"
            assert exists(time), assert_message
            items += [self.to_time(time)]
        # Compute features
        if self.use_context_features:
            assert_message = "context_features exists but no features provided"
            assert exists(features), assert_message
            items += [self.to_features(features)]

        # Compute joint mapping
        if self.use_context_time or self.use_context_features:
            mapping = reduce(torch.stack(items), "n b m -> b m", "sum")
            mapping = self.to_mapping(mapping)

        return mapping

    def run(self, x, time, embedding, features):
        mapping = self.get_mapping(time, features)
        x = torch.cat([x.expand(-1, embedding.size(1), -1), embedding], dim=-1)
        mapping = mapping.unsqueeze(1).expand(-1, embedding.size(1), -1)

        for block in self.blocks:
            x = x + mapping
            x = block(x, features)

        x = x.mean(dim=1).unsqueeze(1)
        x = self.to_out(x)
        x = x.transpose(-1, -2)

        return x

    def forward(
        self,
        x: Tensor,
        time: Tensor,
        embedding_mask_proba: float = 0.0,
        embedding: Tensor | None = None,
        features: Tensor | None = None,
        embedding_scale: float = 1.0,
    ) -> Tensor:
        b, device = embedding.shape[0], embedding.device
        fixed_embedding = self.fixed_embedding(embedding)
        if embedding_mask_proba > 0.0:
            # Randomly mask embedding
            batch_mask = rand_bool(shape=(b, 1, 1), proba=embedding_mask_proba, device=device)
            embedding = torch.where(batch_mask, fixed_embedding, embedding)

        if embedding_scale != 1.0:
            # Compute both normal and fixed embedding outputs
            out = self.run(x, time, embedding=embedding, features=features)
            out_masked = self.run(x, time, embedding=fixed_embedding, features=features)
            # Scale conditional output using classifier-free guidance
            return out_masked + (out - out_masked) * embedding_scale
        else:
            return self.run(x, time, embedding=embedding, features=features)

        return x


# Diffusion utilities and samplers
class Diffusion(nn.Module):
    alias: str = ""

    """Base diffusion class"""

    def denoise_fn(
        self,
        x_noisy: Tensor,
        sigmas: Tensor | None = None,
        sigma: float | None = None,
        **kwargs,
    ) -> Tensor:
        raise NotImplementedError("Diffusion class missing denoise_fn")

    def forward(self, x: Tensor, noise: Tensor = None, **kwargs) -> Tensor:
        raise NotImplementedError("Diffusion class missing forward function")


class VDiffusion(Diffusion):
    alias = "v"

    def __init__(self, net: nn.Module, *, sigma_distribution: Distribution):
        super().__init__()
        self.net = net
        self.sigma_distribution = sigma_distribution

    def get_alpha_beta(self, sigmas: Tensor) -> tuple[Tensor, Tensor]:
        angle = sigmas * math.pi / 2
        alpha = torch.cos(angle)
        beta = torch.sin(angle)
        return alpha, beta

    def denoise_fn(
        self,
        x_noisy: Tensor,
        sigmas: Tensor | None = None,
        sigma: float | None = None,
        **kwargs,
    ) -> Tensor:
        batch_size, device = x_noisy.shape[0], x_noisy.device
        sigmas = to_batch(x=sigma, xs=sigmas, batch_size=batch_size, device=device)
        return self.net(x_noisy, sigmas, **kwargs)

    def forward(self, x: Tensor, noise: Tensor = None, **kwargs) -> Tensor:
        batch_size, device = x.shape[0], x.device

        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        sigmas_padded = rearrange(sigmas, "b -> b 1 1")

        # Get noise
        noise = default(noise, lambda: torch.randn_like(x))

        # Combine input and noise weighted by half-circle
        alpha, beta = self.get_alpha_beta(sigmas_padded)
        x_noisy = x * alpha + noise * beta
        x_target = noise * alpha - x * beta

        # Denoise and return loss
        x_denoised = self.denoise_fn(x_noisy, sigmas, **kwargs)
        return F.mse_loss(x_denoised, x_target)


class KDiffusion(Diffusion):
    """Elucidated Diffusion (Karras et al. 2022): https://arxiv.org/abs/2206.00364"""

    alias = "k"

    def __init__(
        self,
        net: nn.Module,
        *,
        sigma_distribution: Distribution,
        sigma_data: float,  # data distribution standard deviation
        dynamic_threshold: float = 0.0,
    ):
        super().__init__()
        self.net = net
        self.sigma_data = sigma_data
        self.sigma_distribution = sigma_distribution
        self.dynamic_threshold = dynamic_threshold

    def get_scale_weights(self, sigmas: Tensor) -> tuple[Tensor, ...]:
        sigma_data = self.sigma_data
        c_noise = torch.log(sigmas) * 0.25
        sigmas = rearrange(sigmas, "b -> b 1 1")
        c_skip = (sigma_data**2) / (sigmas**2 + sigma_data**2)
        c_out = sigmas * sigma_data * (sigma_data**2 + sigmas**2) ** -0.5
        c_in = (sigmas**2 + sigma_data**2) ** -0.5
        return c_skip, c_out, c_in, c_noise

    def denoise_fn(
        self,
        x_noisy: Tensor,
        sigmas: Tensor | None = None,
        sigma: float | None = None,
        **kwargs,
    ) -> Tensor:
        batch_size, device = x_noisy.shape[0], x_noisy.device
        sigmas = to_batch(x=sigma, xs=sigmas, batch_size=batch_size, device=device)

        # Predict network output and add skip connection
        c_skip, c_out, c_in, c_noise = self.get_scale_weights(sigmas)
        x_pred = self.net(c_in * x_noisy, c_noise, **kwargs)
        x_denoised = c_skip * x_noisy + c_out * x_pred

        return x_denoised

    def loss_weight(self, sigmas: Tensor) -> Tensor:
        # Computes weight depending on data distribution
        return (sigmas**2 + self.sigma_data**2) * (sigmas * self.sigma_data) ** -2

    def forward(self, x: Tensor, noise: Tensor = None, **kwargs) -> Tensor:
        batch_size, device = x.shape[0], x.device
        from einops import rearrange, reduce

        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        sigmas_padded = rearrange(sigmas, "b -> b 1 1")

        # Add noise to input
        noise = default(noise, lambda: torch.randn_like(x))
        x_noisy = x + sigmas_padded * noise

        # Compute denoised values
        x_denoised = self.denoise_fn(x_noisy, sigmas=sigmas, **kwargs)

        # Compute weighted loss
        losses = F.mse_loss(x_denoised, x, reduction="none")
        losses = reduce(losses, "b ... -> b", "mean")
        losses = losses * self.loss_weight(sigmas)
        loss = losses.mean()
        return loss


class VKDiffusion(Diffusion):
    alias = "vk"

    def __init__(self, net: nn.Module, *, sigma_distribution: Distribution):
        super().__init__()
        self.net = net
        self.sigma_distribution = sigma_distribution

    def get_scale_weights(self, sigmas: Tensor) -> tuple[Tensor, ...]:
        sigma_data = 1.0
        sigmas = rearrange(sigmas, "b -> b 1 1")
        c_skip = (sigma_data**2) / (sigmas**2 + sigma_data**2)
        c_out = -sigmas * sigma_data * (sigma_data**2 + sigmas**2) ** -0.5
        c_in = (sigmas**2 + sigma_data**2) ** -0.5
        return c_skip, c_out, c_in

    def sigma_to_t(self, sigmas: Tensor) -> Tensor:
        return sigmas.atan() / math.pi * 2

    def t_to_sigma(self, t: Tensor) -> Tensor:
        return (t * math.pi / 2).tan()

    def denoise_fn(
        self,
        x_noisy: Tensor,
        sigmas: Tensor | None = None,
        sigma: float | None = None,
        **kwargs,
    ) -> Tensor:
        batch_size, device = x_noisy.shape[0], x_noisy.device
        sigmas = to_batch(x=sigma, xs=sigmas, batch_size=batch_size, device=device)

        # Predict network output and add skip connection
        c_skip, c_out, c_in = self.get_scale_weights(sigmas)
        x_pred = self.net(c_in * x_noisy, self.sigma_to_t(sigmas), **kwargs)
        x_denoised = c_skip * x_noisy + c_out * x_pred
        return x_denoised

    def forward(self, x: Tensor, noise: Tensor = None, **kwargs) -> Tensor:
        batch_size, device = x.shape[0], x.device

        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        sigmas_padded = rearrange(sigmas, "b -> b 1 1")

        # Add noise to input
        noise = default(noise, lambda: torch.randn_like(x))
        x_noisy = x + sigmas_padded * noise

        # Compute model output
        c_skip, c_out, c_in = self.get_scale_weights(sigmas)
        x_pred = self.net(c_in * x_noisy, self.sigma_to_t(sigmas), **kwargs)

        # Compute v-objective target
        v_target = (x - c_skip * x_noisy) / (c_out + 1e-7)

        # Compute loss
        loss = F.mse_loss(x_pred, v_target)
        return loss


class Sampler(nn.Module):
    diffusion_types: ClassVar[list[type[Diffusion]]] = []

    def forward(self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int) -> Tensor:
        raise NotImplementedError()

    def inpaint(
        self,
        source: Tensor,
        mask: Tensor,
        fn: Callable,
        sigmas: Tensor,
        num_steps: int,
        num_resamples: int,
    ) -> Tensor:
        raise NotImplementedError("Inpainting not available with current sampler")


class VSampler(Sampler):
    diffusion_types: ClassVar[list[type[Diffusion]]] = [VDiffusion]

    def get_alpha_beta(self, sigma: float) -> tuple[float, float]:
        angle = sigma * math.pi / 2
        alpha = math.cos(angle)
        beta = math.sin(angle)
        return alpha, beta

    def forward(self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int) -> Tensor:
        x = sigmas[0] * noise
        alpha, beta = self.get_alpha_beta(sigmas[0].item())

        for i in range(num_steps - 1):
            is_last = i == num_steps - 1

            x_denoised = fn(x, sigma=sigmas[i])
            x_pred = x * alpha - x_denoised * beta
            x_eps = x * beta + x_denoised * alpha

            if not is_last:
                alpha, beta = self.get_alpha_beta(sigmas[i + 1].item())
                x = x_pred * alpha + x_eps * beta

        return x_pred


class ADPM2Sampler(Sampler):
    """https://www.desmos.com/calculator/jbxjlqd9mb"""

    diffusion_types: ClassVar[list[type[Diffusion]]] = [KDiffusion, VKDiffusion]

    def __init__(self, rho: float = 1.0):
        super().__init__()
        self.rho = rho

    def get_sigmas(self, sigma: float, sigma_next: float) -> tuple[float, float, float]:
        r = self.rho
        sigma_up = math.sqrt(sigma_next**2 * (sigma**2 - sigma_next**2) / sigma**2)
        sigma_down = math.sqrt(sigma_next**2 - sigma_up**2)
        sigma_mid = ((sigma ** (1 / r) + sigma_down ** (1 / r)) / 2) ** r
        return sigma_up, sigma_down, sigma_mid

    def step(self, x: Tensor, fn: Callable, sigma: float, sigma_next: float) -> Tensor:
        # Sigma steps
        sigma_up, sigma_down, sigma_mid = self.get_sigmas(sigma, sigma_next)
        # Derivative at sigma (∂x/∂sigma)
        d = (x - fn(x, sigma=sigma)) / sigma
        # Denoise to midpoint
        x_mid = x + d * (sigma_mid - sigma)
        # Derivative at sigma_mid (∂x_mid/∂sigma_mid)
        d_mid = (x_mid - fn(x_mid, sigma=sigma_mid)) / sigma_mid
        # Denoise to next
        x = x + d_mid * (sigma_down - sigma)
        # Add randomness
        x_next = x + torch.randn_like(x) * sigma_up
        return x_next

    def forward(self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int) -> Tensor:
        x = sigmas[0] * noise
        # Denoise to sample
        for i in range(num_steps - 1):
            x = self.step(x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1])  # type: ignore
        return x

    def inpaint(
        self,
        source: Tensor,
        mask: Tensor,
        fn: Callable,
        sigmas: Tensor,
        num_steps: int,
        num_resamples: int,
    ) -> Tensor:
        x = sigmas[0] * torch.randn_like(source)

        for i in range(num_steps - 1):
            # Noise source to current noise level
            source_noisy = source + sigmas[i] * torch.randn_like(source)
            for r in range(num_resamples):
                # Merge noisy source and current then denoise
                x = source_noisy * mask + x * ~mask
                x = self.step(x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1])  # type: ignore
                # Renoise if not last resample step
                if r < num_resamples - 1:
                    sigma = math.sqrt(sigmas[i] ** 2 - sigmas[i + 1] ** 2)
                    x = x + sigma * torch.randn_like(x)

        return source * mask + x * ~mask


class Schedule(nn.Module):
    """Interface used by different sampling schedules"""

    def forward(self, num_steps: int, device: torch.device) -> Tensor:
        raise NotImplementedError()


class LinearSchedule(Schedule):
    def forward(self, num_steps: int, device: Any) -> Tensor:
        sigmas = torch.linspace(1, 0, num_steps + 1)[:-1]
        return sigmas


class KarrasSchedule(Schedule):
    """https://arxiv.org/abs/2206.00364 equation 5"""

    def __init__(self, sigma_min: float = 0.0001, sigma_max: float = 3.0, rho: float = 9.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def forward(self, num_steps: int, device: Any) -> Tensor:
        rho_inv = 1.0 / self.rho
        steps = torch.arange(num_steps, device=device, dtype=torch.float32)
        sigmas = (
            self.sigma_max**rho_inv + (steps / (num_steps - 1)) * (self.sigma_min**rho_inv - self.sigma_max**rho_inv)
        ) ** self.rho
        sigmas = F.pad(sigmas, pad=(0, 1), value=0.0)
        return sigmas


class DiffusionSampler(nn.Module):
    def __init__(
        self,
        diffusion: Diffusion,
        *,
        sampler: Sampler,
        sigma_schedule: Schedule,
        num_steps: int | None = None,
        clamp: bool = True,
    ):
        super().__init__()
        self.denoise_fn = diffusion.denoise_fn
        self.sampler = sampler
        self.sigma_schedule = sigma_schedule
        self.num_steps = num_steps
        self.clamp = clamp

        # Check sampler is compatible with diffusion type
        sampler_class = sampler.__class__.__name__
        diffusion_class = diffusion.__class__.__name__
        message = f"{sampler_class} incompatible with {diffusion_class}"
        assert diffusion.alias in [t.alias for t in sampler.diffusion_types], message

    def forward(self, noise: Tensor, num_steps: int | None = None, **kwargs) -> Tensor:
        device = noise.device
        num_steps = default(num_steps, self.num_steps)  # type: ignore
        assert exists(num_steps), "Parameter `num_steps` must be provided"
        # Compute sigmas using schedule
        sigmas = self.sigma_schedule(num_steps, device)
        # Append additional kwargs to denoise function (used e.g. for conditional unet)
        fn = lambda *a, **ka: self.denoise_fn(*a, **{**ka, **kwargs})
        # Sample using sampler
        x = self.sampler(noise, fn=fn, sigmas=sigmas, num_steps=num_steps)
        x = x.clamp(-1.0, 1.0) if self.clamp else x
        return x


class Model1d(nn.Module):
    def __init__(self, diffusion: Diffusion):
        super().__init__()
        self.diffusion = diffusion

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.diffusion(x, **kwargs)

    def sample(self, *args, **kwargs) -> Tensor:
        return self.diffusion.sample(*args, **kwargs)


def get_default_model_kwargs():
    return {
        "channels": 128,
        "patch_size": 16,
        "multipliers": [1, 2, 4, 4, 4, 4, 4],
        "factors": [4, 4, 4, 2, 2, 2],
        "num_blocks": [2, 2, 2, 2, 2, 2],
        "attentions": [0, 0, 0, 1, 1, 1, 1],
        "attention_heads": 8,
        "attention_features": 64,
        "attention_multiplier": 2,
        "attention_use_rel_pos": False,
        "diffusion_type": "v",
        "diffusion_sigma_distribution": UniformDistribution(),
    }


def get_default_sampling_kwargs():
    return {"sigma_schedule": LinearSchedule(), "sampler": VSampler(), "clamp": True}


class AudioDiffusionConditional(Model1d):
    def __init__(
        self,
        diffusion: Diffusion,
        *,
        embedding_features: int,
        embedding_max_length: int,
        embedding_mask_proba: float = 0.1,
        **kwargs,
    ):
        self.embedding_mask_proba = embedding_mask_proba
        default_kwargs = dict(
            **get_default_model_kwargs(),
            context_embedding_features=embedding_features,
            context_embedding_max_length=embedding_max_length,
        )
        super().__init__(diffusion)

    def forward(self, *args, **kwargs):
        default_kwargs = {"embedding_mask_proba": self.embedding_mask_proba}
        return super().forward(*args, **{**default_kwargs, **kwargs})

    def sample(self, *args, **kwargs):
        default_kwargs = dict(
            **get_default_sampling_kwargs(),
            embedding_scale=5.0,
        )
        return super().sample(*args, **{**default_kwargs, **kwargs})
