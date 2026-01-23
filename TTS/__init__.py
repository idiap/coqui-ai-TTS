import importlib.metadata

from transformers.utils.import_utils import (
    is_torch_available,
    is_torch_greater_or_equal,
    is_torchaudio_available,
    is_torchcodec_available,
)

from TTS.utils.import_utils import PYTORCH_IMPORT_ERROR, TORCHCODEC_IMPORT_ERROR

__version__ = importlib.metadata.version("coqui-tts")

if "coqpit" in importlib.metadata.packages_distributions().get("coqpit", []):
    msg = (
        "coqui-tts switched to a forked version of Coqpit, but you still have the original "
        "package installed. Run the following to avoid conflicts:\n"
        "  pip uninstall coqpit\n"
        "  pip install coqpit-config"
    )
    raise ImportError(msg)

if not is_torch_available() or not is_torchaudio_available:
    raise ImportError(PYTORCH_IMPORT_ERROR)

if is_torch_greater_or_equal("2.4"):
    import _codecs
    from collections import defaultdict

    import numpy as np
    import torch
    from packaging import version

    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig
    from TTS.utils.radam import RAdam

    torch.serialization.add_safe_globals([dict, defaultdict, RAdam])

    # XTTS
    torch.serialization.add_safe_globals([BaseDatasetConfig, XttsConfig, XttsAudioConfig, XttsArgs])

if is_torch_greater_or_equal("2.9"):
    if not is_torchcodec_available():
        raise ImportError(TORCHCODEC_IMPORT_ERROR)
