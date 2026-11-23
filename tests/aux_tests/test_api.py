import pytest
import torch
from torch import nn

from TTS.api import TTS


def test_vocoder_model_name_rejected():
    """Passing a vocoder model name as model_name should raise a clear error."""
    with pytest.raises(ValueError, match="cannot be used as a standalone model"):
        TTS(model_name="vocoder_models/de/thorsten/wavegrad")


def test_new_models_follow_current_device():
    """Models loaded later (e.g. by tts_with_vc) are moved to the device of already loaded ones."""
    tts = TTS()
    assert tts._target_device(gpu=False) is None
    tts.synthesizer = nn.Linear(1, 1).to("meta")
    assert tts._target_device(gpu=False) == torch.device("meta")


def test_gpu_argument_deprecated():
    tts = TTS()
    with pytest.warns(DeprecationWarning, match="`gpu` is deprecated"):
        assert tts._target_device(gpu=True) == "cuda"
