from dataclasses import dataclass, field

from coqpit import Coqpit

from TTS.config.shared_configs import BaseAudioConfig
from TTS.vc.configs.shared_configs import BaseVCConfig


@dataclass
class KNNVCAudioConfig(BaseAudioConfig):
    """Audio configuration.

    Args:
        sample_rate (int):
            The sampling rate of the input waveform.
    """

    sample_rate: int = field(default=16000)


@dataclass
class KNNVCArgs(Coqpit):
    """Model arguments.

    Args:
        ssl_dim (int):
            The dimension of the self-supervised learning embedding.
    """

    ssl_dim: int = field(default=1024)


@dataclass
class KNNVCConfig(BaseVCConfig):
    """Parameters.

    Args:
        model (str):
            Model name. Do not change unless you know what you are doing.

        model_args (KNNVCArgs):
            Model architecture arguments. Defaults to `KNNVCArgs()`.

        audio (KNNVCAudioConfig):
            Audio processing configuration. Defaults to `KNNVCAudioConfig()`.

        wavlm_layer (int):
            WavLM layer to use for feature extraction.

        topk (int):
            k in the kNN -- the number of nearest neighbors to average over
    """

    model: str = "knnvc"
    model_args: KNNVCArgs = field(default_factory=KNNVCArgs)
    audio: KNNVCAudioConfig = field(default_factory=KNNVCAudioConfig)

    wavlm_layer: int = 6
    topk: int = 4
