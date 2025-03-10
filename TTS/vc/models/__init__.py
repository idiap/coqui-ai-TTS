import importlib
import logging
import re

from TTS.vc.configs.shared_configs import BaseVCConfig
from TTS.vc.models.base_vc import BaseVC

logger = logging.getLogger(__name__)


def setup_model(config: BaseVCConfig) -> BaseVC:
    logger.info("Using model: %s", config.model)
    # fetch the right model implementation.
    if config["model"].lower() == "freevc":
        MyModel = importlib.import_module("TTS.vc.models.freevc").FreeVC
    elif config["model"].lower() == "knnvc":
        MyModel = importlib.import_module("TTS.vc.models.knnvc").KNNVC
    else:
        msg = f"Model {config.model} does not exist!"
        raise ValueError(msg)
    return MyModel.init_from_config(config)
