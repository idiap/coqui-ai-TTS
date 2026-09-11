"""Configuration for the optional OmniVoice integration."""

from dataclasses import dataclass, field

from coqpit import Coqpit

from TTS.tts.configs.shared_configs import BaseTTSConfig


@dataclass
class OmnivoiceAudioConfig(Coqpit):
    sample_rate: int = 24000
    output_sample_rate: int = 24000


@dataclass
class OmnivoiceConfig(BaseTTSConfig):
    """Coqui configuration adapter for a Hugging Face OmniVoice checkpoint."""

    model: str = "omnivoice"
    _supports_cloning: bool = True
    # Replaced with OmniVoice's complete language list after loading the checkpoint.
    languages: list[str] = field(default_factory=lambda: ["en"])
    audio: OmnivoiceAudioConfig = field(default_factory=OmnivoiceAudioConfig)
