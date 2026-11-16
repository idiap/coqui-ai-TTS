"""Kokoro configuration."""

from dataclasses import dataclass

from TTS.tts.configs.styletts2_config import StyleTTS2Config


@dataclass
class KokoroConfig(StyleTTS2Config):
    """Configuration for Kokoro models.

    Kokoro uses the StyleTTS 2 architecture without the style encoders and
    style diffusion: speakers are represented by precomputed style vectors. See
    :class:`~TTS.tts.configs.styletts2_config.StyleTTS2Config` for the
    available arguments.
    """

    model: str = "kokoro"
