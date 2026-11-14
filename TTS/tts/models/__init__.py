import logging
from typing import TYPE_CHECKING

from TTS.utils.generic_utils import find_module

if TYPE_CHECKING:
    from TTS.tts.configs.shared_configs import BaseTTSConfig
    from TTS.tts.models.base_tts import BaseTTS

logger = logging.getLogger(__name__)


def get_model_class(config: "BaseTTSConfig") -> type["BaseTTS"]:
    """Return the TTS model class for the given config."""
    if "base_model" in config and config["base_model"] is not None:
        return find_module("TTS.tts.models", config.base_model.lower())
    return find_module("TTS.tts.models", config.model.lower())


def setup_model(config: "BaseTTSConfig") -> "BaseTTS":
    logger.info("Using model: %s", config.model)
    return get_model_class(config)(config)
