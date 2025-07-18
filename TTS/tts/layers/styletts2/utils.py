"""Utilities for StyleTTS 2."""

import torch


def _length_to_mask(lengths: torch.Tensor) -> torch.Tensor:
    """Convert lengths to mask."""
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    return mask >= lengths.unsqueeze(1)


def rand_bool(shape, proba, device=None):
    if proba == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif proba == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.bernoulli(torch.full(shape, proba, device=device)).to(torch.bool)


def group_dict_by_prefix(prefix: str, d: dict) -> tuple[dict, dict]:
    return_dicts: tuple[dict, dict] = ({}, {})
    for key, value in d.items():
        no_prefix = int(not key.startswith(prefix))
        return_dicts[no_prefix][key] = value
    return return_dicts


def groupby(prefix: str, d: dict, keep_prefix: bool = False) -> tuple[dict, dict]:
    kwargs_with_prefix, kwargs = group_dict_by_prefix(prefix, d)
    if keep_prefix:
        return kwargs_with_prefix, kwargs
    kwargs_no_prefix = {k[len(prefix) :]: v for k, v in kwargs_with_prefix.items()}
    return kwargs_no_prefix, kwargs
