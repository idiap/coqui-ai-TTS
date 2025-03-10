import logging

from TTS.utils.generic_utils import find_module

logger = logging.getLogger(__name__)


def setup_model(config: "Coqpit", samples: list[list] | list[dict] = None) -> "BaseTTS":
    logger.info("Using model: %s", config.model)
    # fetch the right model implementation.
    if "base_model" in config and config["base_model"] is not None:
        MyModel = find_module("TTS.tts.models", config.base_model.lower())
    else:
        MyModel = find_module("TTS.tts.models", config.model.lower())
    model = MyModel.init_from_config(config=config, samples=samples)
    return model
