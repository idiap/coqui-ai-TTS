from typing import cast

from TTS.model import BaseTrainerModel
from TTS.utils.audio.processor import AudioProcessor
from TTS.vocoder.configs.shared_configs import BaseVocoderConfig

# pylint: skip-file


class BaseVocoder(BaseTrainerModel):
    """Base `vocoder` class. Every new `vocoder` model must inherit this.

    It defines `vocoder` specific functions on top of `Model`.

    Notes on input/output tensor shapes:
        Any input or output tensor of the model must be shaped as

        - 3D tensors `batch x time x channels`
        - 2D tensors `batch x channels`
        - 1D tensors `batch x 1`
    """

    MODEL_TYPE = "vocoder"
    config: BaseVocoderConfig

    def __init__(self, config):
        super().__init__()
        self.config = cast(BaseVocoderConfig, config)
        self.ap = AudioProcessor(self.config.audio)
        self._set_model_args()

    def _set_model_args(self) -> None:
        """Setup model args from the config."""
        if isinstance(self.config, BaseVocoderConfig):
            if "model_args" in self.config:
                self.args = self.config.model_args
            # This is for backward compatibility
            if "model_params" in self.config:
                self.args = self.config.model_params
        else:
            raise ValueError("config must be an instance of BaseVocoderConfig")
