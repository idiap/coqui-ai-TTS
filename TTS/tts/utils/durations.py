"""Utilities for working with token durations derived from attention maps."""

from __future__ import annotations

from math import ceil
from typing import Iterable

import torch


def _ensure_3d_attention(attn_mask: torch.Tensor) -> torch.Tensor:
    """Collapse a batched attention map to ``[B, T_text, T_mel]``.

    The dataset collate function pads the attention masks to a 4-D tensor of
    shape ``[B, 1, T_text, T_mel]``. Some models may also return multi-head
    attention tensors. In both cases we pick the strongest head per frame by
    taking the maximum over the head dimension.
    """

    if attn_mask.dim() == 3:
        return attn_mask

    if attn_mask.dim() == 4:
        # Collapse the head dimension. If there is only a single head this is
        # equivalent to ``squeeze`` but keeps the code path uniform.
        attn_mask = attn_mask.max(dim=1).values
        return attn_mask

    raise ValueError(
        "Expected attention mask with 3 or 4 dimensions, "
        f"got shape {tuple(attn_mask.shape)}"
    )


def _repeat_indices(indices: torch.Tensor, total: int) -> torch.Tensor:
    """Repeat ``indices`` until reaching ``total`` elements and crop the result."""

    if indices.numel() == 0 or total <= 0:
        return indices.new_empty(0)

    repeats = ceil(total / indices.numel())
    return indices.repeat(repeats)[:total]


def durations_from_alignment(
    attn_mask: torch.Tensor | None,
    text_lengths: torch.Tensor | Iterable[int],
    mel_lengths: torch.Tensor | Iterable[int],
) -> torch.Tensor | None:
    """Compute token durations from a padded attention alignment map.

    Parameters
    ----------
    attn_mask:
        Attention tensor with shape ``[B, T_text, T_mel]`` or ``[B, 1, T_text, T_mel]``.
        When ``None`` no durations are computed and ``None`` is returned.
    text_lengths:
        Sequence of text lengths for the batch.
    mel_lengths:
        Sequence of mel-frame lengths for the batch.

    Returns
    -------
    torch.Tensor | None
        Tensor with shape ``[B, max(text_lengths)]`` that stores the duration of
        each token. The tensor uses the same ``dtype`` and ``device`` as the
        attention mask.
    """

    if attn_mask is None:
        return None

    attn_mask = _ensure_3d_attention(attn_mask)
    batch, max_text_len, _ = attn_mask.shape
    device = attn_mask.device
    dtype = attn_mask.dtype

    # Convert inputs to tensors for indexing while supporting lists/tuples.
    if not torch.is_tensor(text_lengths):
        text_lengths = torch.as_tensor(text_lengths, device=device)
    if not torch.is_tensor(mel_lengths):
        mel_lengths = torch.as_tensor(mel_lengths, device=device)

    durations = torch.zeros(batch, max_text_len, device=device, dtype=dtype)

    for idx in range(batch):
        text_len = int(text_lengths[idx])
        mel_len = int(mel_lengths[idx])

        if text_len <= 0 or mel_len <= 0:
            continue

        attn = attn_mask[idx, :text_len, :mel_len]

        # Count how many frames attend to each token.
        frame_to_token = attn.max(dim=0).indices
        dur = torch.bincount(frame_to_token, minlength=text_len)
        dur = dur.to(dtype)

        # Guarantee that every token has at least one frame by assigning a
        # single frame to zero-duration tokens and compensating elsewhere.
        zero_tokens = dur == 0
        if zero_tokens.any():
            dur = dur + zero_tokens.to(dtype)

        diff = int(dur.sum().item() - mel_len)
        if diff != 0:
            if diff > 0:
                candidates = torch.nonzero(dur > 1, as_tuple=False).view(-1)
                if candidates.numel() == 0:
                    candidates = torch.arange(text_len, device=device)
                adjust_idx = _repeat_indices(
                    candidates[torch.argsort(dur[candidates], descending=True)],
                    diff,
                )
                if adjust_idx.numel() > 0:
                    adjustment = torch.zeros_like(dur)
                    adjustment.scatter_add_(
                        0,
                        adjust_idx,
                        torch.ones(adjust_idx.numel(), device=device, dtype=dtype),
                    )
                    dur = dur - adjustment
            else:
                diff = -diff
                candidates = torch.arange(text_len, device=device)
                adjust_idx = _repeat_indices(
                    candidates[torch.argsort(dur[candidates], descending=True)],
                    diff,
                )
                if adjust_idx.numel() > 0:
                    adjustment = torch.zeros_like(dur)
                    adjustment.scatter_add_(
                        0,
                        adjust_idx,
                        torch.ones(adjust_idx.numel(), device=device, dtype=dtype),
                    )
                    dur = dur + adjustment

        durations[idx, :text_len] = dur

    return durations

