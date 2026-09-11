"""Utilities to check availability of packages.

Largely adapted from:
https://github.com/huggingface/transformers/blob/main/src/transformers/utils/import_utils.py
"""

import importlib.metadata
import importlib.util
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, overload

from packaging import version

logger = logging.getLogger(__name__)

PACKAGE_DISTRIBUTION_MAPPING = importlib.metadata.packages_distributions()

PYTORCH_IMPORT_ERROR = """
Coqui TTS requires the PyTorch and Torchaudio libraries but they were not found in your
environment. Check out the instructions on the installation page:
https://pytorch.org/get-started/locally/
and follow the ones that match your environment.
"""

TORCHCODEC_IMPORT_ERROR = """
From Pytorch 2.9, the torchcodec library is required for audio IO, but it was not
found in your environment. You can install it with Coqui's `codec` extra:
```
pip install coqui-tts[codec]
```
"""


@dataclass(frozen=True)
class PhonemizeDeps:
    """Dependencies for a phonemizer."""

    packages: list[str]
    extra: str

    def import_error(self, name: str) -> str:
        packages_str = ", ".join(self.packages)
        return f"""
Phonemizer `{name}` requires: {packages_str}. Install with Coqui's `{self.extra}` extra:
```
pip install coqui-tts[{self.extra}]
```
"""


PHONEMIZER_DEPS: dict[str, PhonemizeDeps] = {
    "bn_phonemizer": PhonemizeDeps(["bangla", "bnnumerizer", "bnunicodenormalizer"], "bn"),
    "gruut": PhonemizeDeps(
        [
            "gruut",
        ],
        "languages",
    ),
    "ja_jp_phonemizer": PhonemizeDeps(
        [
            "fugashi",
        ],
        "ja",
    ),
    "ko_kr_phonemizer": PhonemizeDeps(
        [
            "mecab-ko",
        ],
        "ko",
    ),
    "zh_cn_phonemizer": PhonemizeDeps(["pypinyin", "spacy_pkuseg"], "zh"),
}


@overload
def _is_package_available(pkg_name: str, *, return_version: Literal[False] = False) -> bool: ...
@overload
def _is_package_available(pkg_name: str, *, return_version: Literal[True]) -> tuple[bool, str]: ...
def _is_package_available(pkg_name: str, *, return_version: bool = False):
    """Check if `pkg_name` exist, and optionally try to get its version."""
    spec = importlib.util.find_spec(pkg_name)
    package_exists = spec is not None
    package_version = "N/A"
    if package_exists and return_version:
        try:
            # importlib.metadata works with the distribution package, which may be different from the import
            # name (e.g. `PIL` is the import name, but `pillow` is the distribution name)
            distributions = PACKAGE_DISTRIBUTION_MAPPING[pkg_name]
            # Per PEP 503, underscores and hyphens are equivalent in package names.
            # Prefer the distribution that matches the (normalized) package name.
            normalized_pkg_name = pkg_name.replace("_", "-")
            if normalized_pkg_name in distributions:
                distribution_name = normalized_pkg_name
            elif pkg_name in distributions:
                distribution_name = pkg_name
            else:
                distribution_name = distributions[0]
            package_version = importlib.metadata.version(distribution_name)
        except (importlib.metadata.PackageNotFoundError, KeyError):
            # If we cannot find the metadata (because of editable install for example), try to import directly.
            # Note that this branch will almost never be run, so we do not import packages for nothing here
            package = importlib.import_module(pkg_name)
            package_version = getattr(package, "__version__", "N/A")
        logger.debug("Detected %s version: %s", pkg_name, package_version)
    if return_version:
        return package_exists, package_version
    return package_exists


@lru_cache
def is_torch_available() -> bool:
    """Return whether PyTorch is available in the environment."""
    return _is_package_available("torch")


@lru_cache
def get_torch_version() -> str:
    """Return the PyTorch version."""
    _, torch_version = _is_package_available("torch", return_version=True)
    return torch_version


@lru_cache
def is_torch_greater_or_equal(library_version: str) -> bool:
    """Return True if the current PyTorch version is greater than or equal to the given version."""
    if not is_torch_available():
        return False
    return version.parse(get_torch_version()) >= version.parse(library_version)


@lru_cache
def is_torchaudio_available() -> bool:
    """Return whether Torchaudio is available in the environment."""
    return _is_package_available("torchaudio")


@lru_cache
def is_torchcodec_available() -> bool:
    """Return whether Torchcodec is available in the environment."""
    return _is_package_available("torchcodec")


@lru_cache
def is_phonemizer_available(phonemizer: str) -> bool:
    """Return whether packages for the given phonemizer are available in the environment."""
    if phonemizer not in PHONEMIZER_DEPS:
        msg = f"Not a valid phonemizer name: {phonemizer}"
        raise ValueError(msg)
    return all(_is_package_available(p) for p in PHONEMIZER_DEPS[phonemizer].packages)
