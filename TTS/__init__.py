import importlib.metadata

from TTS.utils.generic_utils import is_pytorch_at_least_2_4

__version__ = importlib.metadata.version("coqui-tts")

if "coqpit" in importlib.metadata.packages_distributions().get("coqpit", []):
    msg = (
        "coqui-tts switched to a forked version of Coqpit, but you still have the original "
        "package installed. Run the following to avoid conflicts:\n"
        "  pip uninstall coqpit\n"
        "  pip install coqpit-config"
    )
    raise ImportError(msg)


if is_pytorch_at_least_2_4():
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
