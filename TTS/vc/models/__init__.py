import importlib
import logging
import re

from TTS.vc.configs.shared_configs import BaseVCConfig
from TTS.vc.models.base_vc import BaseVC

logger = logging.getLogger(__name__)


def get_model_class(config: BaseVCConfig) -> type[BaseVC]:
    """Return the voice conversion model class for the given config."""
    if config["model"].lower() == "freevc":
        return importlib.import_module("TTS.vc.models.freevc").FreeVC
    if config["model"].lower() == "knnvc":
        return importlib.import_module("TTS.vc.models.knnvc").KNNVC
    msg = f"Model {config.model} does not exist!"
    raise ValueError(msg)


def setup_model(config: BaseVCConfig) -> BaseVC:
    logger.info("Using model: %s", config.model)
    return get_model_class(config)(config)
