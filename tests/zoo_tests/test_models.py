#!/usr/bin/env python3`
import os
import shutil

import huggingface_hub.constants
import pytest

from tests import get_tests_data_path, run_main
from TTS.api import TTS
from TTS.bin.synthesize import main
from TTS.utils.manage import ModelManager

MODELS_WITH_SEP_TESTS = [
    "tts_models/multilingual/multi-dataset/bark",
    "tts_models/en/multi-dataset/tortoise-v2",
    "tts_models/multilingual/multi-dataset/xtts_v1.1",
    "tts_models/multilingual/multi-dataset/xtts_v2",
]

# These contain np.core.multiarray.scalar which cannot be added safe globals for
# weights-only loading because it's renamed to np._core.multiarray.scalar
BROKEN_MODELS = [
    "tts_models/en/blizzard2013/capacitron-t2-c50",
    "tts_models/en/blizzard2013/capacitron-t2-c150_v2",
]


@pytest.fixture(autouse=True)
def run_around_tests(monkeypatch, tmp_path):
    """Download models to a temp folder and delete it afterwards."""
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setenv("TTS_HOME", str(tmp_path))
    yield
    shutil.rmtree(tmp_path)


# To split tests into different CI jobs
num_partitions = int(os.getenv("NUM_PARTITIONS", "1"))
partition = int(os.getenv("TEST_PARTITION", "0"))
model_names = [name for name in TTS.list_models() if name not in MODELS_WITH_SEP_TESTS and name not in BROKEN_MODELS]
model_names.extend(["tts_models/deu/fairseq/vits", "tts_models/sqi/fairseq/vits"])
model_names = [name for i, name in enumerate(model_names) if i % num_partitions == partition]


@pytest.mark.parametrize("model_name", model_names)
def test_models(tmp_path, model_name):
    print(f"\n > Run - {model_name}")
    file_path = tmp_path / "output.wav"
    if "tts_models" in model_name:
        api = TTS(model_name, progress_bar=False)
        speaker = None
        language = api.languages[0] if api.languages else None
        speaker = api.speakers[0] if api.speakers else None
        api.tts_to_file("This is an example.", speaker=speaker, language=language, file_path=file_path)
    elif "voice_conversion_models" in model_name:
        api = TTS(model_name, progress_bar=False)
        speaker_wav = os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0001.wav")
        reference_wav1 = os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0028.wav")
        reference_wav2 = os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0032.wav")
        api.voice_conversion_to_file(speaker_wav, [reference_wav1, reference_wav2], file_path=file_path)
    else:
        # only download the model
        ModelManager(output_prefix=tmp_path, progress_bar=False).download_model(model_name)
    print(f" | > OK: {model_name}")


def test_voice_conversion(tmp_path):
    print(" > Run voice conversion inference using YourTTS model.")
    args = [
        "--model_name",
        "tts_models/multilingual/multi-dataset/your_tts",
        "--out_path",
        str(tmp_path / "output.wav"),
        "--speaker_wav",
        os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0001.wav"),
        "--reference_wav",
        os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0032.wav"),
        "--language_idx",
        "en",
        "--no-progress_bar",
    ]
    run_main(main, args)
