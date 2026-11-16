import pytest
import torch
from torch import nn

from TTS.api import TTS
from TTS.utils.manage import ModelManager


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


MODEL_NAME = "tts_models/en/ljspeech/vits"


def test_voice_dir_next_to_model_files(tmp_path):
    """Voices of models loaded by name are cached in the folder of the model files."""
    tts = TTS()
    tts.manager = ModelManager(output_prefix=tmp_path, progress_bar=False)

    assert tts._get_voice_dir(MODEL_NAME) == tts.manager.output_prefix / "tts_models--en--ljspeech--vits" / "voices"


def test_voice_dir_moved_from_old_location(tmp_path):
    """Voices cached in the location used in v0.28 are moved to the new one."""
    tts = TTS()
    tts.manager = ModelManager(output_prefix=tmp_path, progress_bar=False)
    old_voice_dir = tts.manager.output_prefix / MODEL_NAME / "voices"
    old_voice_dir.mkdir(parents=True)
    (old_voice_dir / "speaker.pth").write_bytes(b"voice")

    voice_dir = tts._get_voice_dir(MODEL_NAME)

    assert (voice_dir / "speaker.pth").read_bytes() == b"voice"
    assert not (tts.manager.output_prefix / "tts_models").exists()
