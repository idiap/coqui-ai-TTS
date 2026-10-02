from TTS.vocoder.configs.fullband_melgan_config import FullbandMelganConfig
from TTS.vocoder.configs.hifigan_config import HifiganConfig
from TTS.vocoder.configs.melgan_config import MelganConfig
from TTS.vocoder.configs.multiband_melgan_config import MultibandMelganConfig
from TTS.vocoder.configs.parallel_wavegan_config import ParallelWaveganConfig
from TTS.vocoder.configs.shared_configs import BaseGANVocoderConfig, BaseVocoderConfig
from TTS.vocoder.configs.univnet_config import UnivnetConfig
from TTS.vocoder.configs.wavegrad_config import WavegradArgs, WavegradConfig
from TTS.vocoder.configs.wavernn_config import WavernnArgs, WavernnConfig

__all__ = [
    "BaseGANVocoderConfig",
    "BaseVocoderConfig",
    "FullbandMelganConfig",
    "HifiganConfig",
    "MelganConfig",
    "MultibandMelganConfig",
    "ParallelWaveganConfig",
    "UnivnetConfig",
    "WavegradArgs",
    "WavegradConfig",
    "WavernnArgs",
    "WavernnConfig",
]
