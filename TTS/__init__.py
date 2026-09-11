import importlib.metadata

from TTS.utils.import_utils import (
    PYTORCH_IMPORT_ERROR,
    TORCHCODEC_IMPORT_ERROR,
    is_torch_available,
    is_torch_greater_or_equal,
    is_torchaudio_available,
    is_torchcodec_available,
)

__version__ = importlib.metadata.version("coqui-tts")

if "coqpit" in importlib.metadata.packages_distributions().get("coqpit", []):
    msg = (
        "coqui-tts switched to a forked version of Coqpit, but you still have the original "
        "package installed. Run the following to avoid conflicts:\n"
        "  pip uninstall coqpit\n"
        "  pip install coqpit-config"
    )
    raise ImportError(msg)

if not is_torch_available() or not is_torchaudio_available():
    raise ImportError(PYTORCH_IMPORT_ERROR)

if is_torch_greater_or_equal("2.9"):
    if not is_torchcodec_available():
        raise ImportError(TORCHCODEC_IMPORT_ERROR)
