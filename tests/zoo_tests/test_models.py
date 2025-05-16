#!/usr/bin/env python3`
import os
import shutil

import pytest

from tests import get_tests_data_path, run_main
from TTS.api import TTS
from TTS.bin.synthesize import main
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
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
def run_around_tests(tmp_path):
    """Download models to a temp folder and delete it afterwards."""
    os.environ["TTS_HOME"] = str(tmp_path)
    yield
    shutil.rmtree(tmp_path)


@pytest.fixture
def manager(tmp_path):
    """Set up model manager."""
    return ModelManager(output_prefix=tmp_path, progress_bar=False)


# To split tests into different CI jobs
num_partitions = int(os.getenv("NUM_PARTITIONS", "1"))
partition = int(os.getenv("TEST_PARTITION", "0"))
model_names = [name for name in TTS.list_models() if name not in MODELS_WITH_SEP_TESTS and name not in BROKEN_MODELS]
model_names.extend(["tts_models/deu/fairseq/vits", "tts_models/sqi/fairseq/vits"])
model_names = [name for i, name in enumerate(model_names) if i % num_partitions == partition]


@pytest.mark.parametrize("model_name", model_names)
def test_models(tmp_path, model_name, manager):
    print(f"\n > Run - {model_name}")
    output_path = str(tmp_path / "output.wav")
    model_path, _, _ = manager.download_model(model_name)
    args = ["--model_name", model_name, "--out_path", output_path, "--no-progress_bar"]
    if "tts_models" in model_name:
        local_download_dir = model_path.parent
        # download and run the model
        speaker_files = list(local_download_dir.glob("speaker*"))
        language_files = list(local_download_dir.glob("language*"))
        speaker_arg = []
        language_arg = []
        if len(speaker_files) > 0:
            # multi-speaker model
            if "speaker_ids" in speaker_files[0].stem:
                speaker_manager = SpeakerManager(speaker_id_file_path=speaker_files[0])
            elif "speakers" in speaker_files[0].stem:
                speaker_manager = SpeakerManager(d_vectors_file_path=speaker_files[0])
            speakers = list(speaker_manager.name_to_id.keys())
            if len(speakers) > 1:
                speaker_arg = ["--speaker_idx", speakers[0]]
        if len(language_files) > 0 and "language_ids" in language_files[0].stem:
            # multi-lingual model
            language_manager = LanguageManager(language_ids_file_path=language_files[0])
            languages = language_manager.language_names
            if len(languages) > 1:
                language_arg = ["--language_idx", languages[0]]
        run_main(main, [*args, "--text", "This is an example.", *speaker_arg, *language_arg])
    elif "voice_conversion_models" in model_name:
        speaker_wav = os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0001.wav")
        reference_wav1 = os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0028.wav")
        reference_wav2 = os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0032.wav")
        run_main(main, [*args, "--source_wav", speaker_wav, "--target_wav", reference_wav1, reference_wav2])
    else:
        # only download the model
        manager.download_model(model_name)
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
