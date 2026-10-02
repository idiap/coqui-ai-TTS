import os
from pathlib import Path

import pytest

from tests import get_tests_data_path
from TTS.api import TTS

GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

SOURCE_WAV = Path(get_tests_data_path()) / "ljspeech/wavs/LJ001-0001.wav"
SPEAKER_WAV = Path(get_tests_data_path()) / "ljspeech/wavs/LJ001-0002.wav"
SPEAKER = "LJ"

TEXT = "hello world"

TTS_MODELS = [
    "tts_models/multilingual/multi-dataset/xtts_v2",
    "tts_models/multilingual/multi-dataset/your_tts",
    "tts_models/multilingual/multi-dataset/bark",
    "tts_models/en/multi-dataset/tortoise-v2",
]

VC_MODELS = [
    "voice_conversion_models/multilingual/multi-dataset/knnvc",
    "voice_conversion_models/multilingual/vctk/freevc24",
    "voice_conversion_models/multilingual/multi-dataset/openvoice_v2",
    "tts_models/multilingual/multi-dataset/your_tts",
]

MARKED_VC_MODELS = [
    pytest.param(
        model, marks=pytest.mark.skipif(GITHUB_ACTIONS and "openvoice" not in model, reason="Model too big for CI")
    )
    for model in VC_MODELS
]


@pytest.mark.skipif(GITHUB_ACTIONS, reason="Model too big for CI")
@pytest.mark.parametrize("model_name", TTS_MODELS)
def test_clone_tts(model_name, tmp_path):
    api = TTS(model_name)
    language = "en" if "xtts" in model_name or "your_tts" in model_name else None
    with pytest.raises(FileNotFoundError, match=f"{SPEAKER}.pth"):
        api.tts_to_file(
            TEXT,
            language=language,
            speaker=SPEAKER,
            voice_dir=tmp_path / "voices",
        )
    api.tts_to_file(
        TEXT,
        language=language,
        speaker_wav=SPEAKER_WAV,
        speaker=SPEAKER,
        file_path=tmp_path / "out1.wav",
        voice_dir=tmp_path / "voices",
    )
    assert (tmp_path / "voices" / f"{SPEAKER}.pth").is_file()
    api.tts_to_file(
        TEXT,
        language=language,
        speaker=SPEAKER,
        file_path=tmp_path / "out2.wav",
        voice_dir=tmp_path / "voices",
    )
    del api


@pytest.mark.parametrize("model_name", MARKED_VC_MODELS)
def test_clone_vc(model_name, tmp_path):
    api = TTS(model_name)
    with pytest.raises(FileNotFoundError, match=f"{SPEAKER}.pth"):
        api.voice_conversion_to_file(
            SOURCE_WAV,
            speaker=SPEAKER,
            voice_dir=tmp_path / "voices",
        )
    api.voice_conversion_to_file(
        SOURCE_WAV,
        SPEAKER_WAV,
        speaker=SPEAKER,
        file_path=tmp_path / "out1.wav",
        voice_dir=tmp_path / "voices",
    )
    assert (tmp_path / "voices" / f"{SPEAKER}.pth").is_file()
    api.voice_conversion_to_file(
        SOURCE_WAV,
        speaker=SPEAKER,
        file_path=tmp_path / "out2.wav",
        voice_dir=tmp_path / "voices",
    )
    del api
