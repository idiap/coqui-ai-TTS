import logging
from typing import Any, cast

import torch
from coqpit import Coqpit
from trainer import TrainerConfig

from TTS.config.shared_configs import ModelArgs
from TTS.model import BaseTrainerModel
from TTS.vc.configs.shared_configs import BaseVCConfig

# pylint: skip-file

logger = logging.getLogger(__name__)


class BaseVC(BaseTrainerModel):
    """Base VC class. Every new voice conversion model must inherit this.

    It defines common VC-specific functions on top of the :py:class:`~TTS.model.BaseTrainerModel`.
    """

    MODEL_TYPE = "vc"
    config: BaseVCConfig

    def __init__(self, config: Coqpit) -> None:
        super().__init__()
        self.config = cast(BaseVCConfig, config)
        self._set_model_args()

    def _set_model_args(self) -> None:
        """Set up model args based on the config type (``ModelConfig`` or ``ModelArgs``).

        ``ModelArgs`` has all the fields required to initialize the model architecture.

        ``ModelConfig`` has all the fields required for training, inference and containes ``ModelArgs``.

        If the config is for training with a name like ``*Config``, then the model args are embeded in the
        ``config.model_args``

        If the config is for the model with a name like ``*Args``, then we assign them directly.
        """
        if isinstance(self.config, BaseVCConfig):
            self.args = self.config.model_args
        elif isinstance(self.config, ModelArgs):
            self.args = self.config
        else:
            raise ValueError("config must be either a *Config or *Args")

    def get_data_loader(
        self, config: TrainerConfig, *, is_eval: bool = False, samples: list[Any] | None = None, verbose: bool = True
    ) -> torch.utils.data.DataLoader[Any]:
        raise NotImplementedError
