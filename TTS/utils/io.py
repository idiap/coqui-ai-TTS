"""Helpers for opening paths through fsspec.

``open_fsspec`` is a thin wrapper around :func:`fsspec.open` that the codebase
uses in place of calling ``fsspec.open`` directly. For S3-compatible backends
it threads a distribution identifier into the s3fs client configuration; every
other backend and local path is opened unchanged.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any

import fsspec

_S3_PROTOCOLS = ("s3://", "s3a://")


def _distribution_tag() -> str:
    try:
        version = importlib.metadata.version("coqui-tts")
    except importlib.metadata.PackageNotFoundError:
        version = "dev"
    return f"coqui-tts/{version}"


def _is_s3_path(path: Any) -> bool:
    return isinstance(path, str) and path.startswith(_S3_PROTOCOLS)


def _append_s3_tag(kwargs: dict[str, Any]) -> None:
    """Add the distribution tag to the s3fs client config in-place, keeping any
    value the caller already configured."""
    config_kwargs = dict(kwargs.get("config_kwargs") or {})
    existing = config_kwargs.get("user_agent_extra")
    tag = _distribution_tag()
    config_kwargs["user_agent_extra"] = f"{existing} {tag}" if existing else tag
    kwargs["config_kwargs"] = config_kwargs


def open_fsspec(path: Any, *args: Any, **kwargs: Any) -> Any:
    """Open ``path`` with :func:`fsspec.open`.

    Args:
        path: Any path or URL supported by fsspec.
        *args: Positional arguments forwarded to :func:`fsspec.open`.
        **kwargs: Storage and connection options forwarded to
            :func:`fsspec.open`.

    Returns:
        An :class:`fsspec.core.OpenFile` instance.
    """
    if _is_s3_path(path):
        _append_s3_tag(kwargs)
    return fsspec.open(path, *args, **kwargs)
