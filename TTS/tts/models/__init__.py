import logging

from TTS.utils.generic_utils import find_module

logger = logging.getLogger(__name__)


def setup_model(config: "BaseTTSConfig") -> "BaseTTS":
    logger.info("Using model: %s", config.model)
    # fetch the right model implementation.
    if "base_model" in config and config["base_model"] is not None:
        MyModel = find_module("TTS.tts.models", config.base_model.lower())
    else:
        MyModel = find_module("TTS.tts.models", config.model.lower())
    return MyModel(config)
