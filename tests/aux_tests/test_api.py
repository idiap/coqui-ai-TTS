import pytest

from TTS.api import TTS


def test_vocoder_model_name_rejected():
    """Passing a vocoder model name as model_name should raise a clear error."""
    with pytest.raises(ValueError, match="cannot be used as a standalone model"):
        TTS(model_name="vocoder_models/de/thorsten/wavegrad")
