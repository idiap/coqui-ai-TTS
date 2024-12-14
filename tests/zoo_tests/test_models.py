#!/usr/bin/env python3`
import os
import shutil

import pytest
import torch
from trainer.io import get_user_data_dir

from tests import get_tests_data_path, run_main
from TTS.api import TTS
from TTS.bin.synthesize import main
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.manage import ModelManager

GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

MODELS_WITH_SEP_TESTS = [
    "tts_models/multilingual/multi-dataset/bark",
    "tts_models/en/multi-dataset/tortoise-v2",
    "tts_models/multilingual/multi-dataset/xtts_v1.1",
    "tts_models/multilingual/multi-dataset/xtts_v2",
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
model_names = [name for name in TTS.list_models() if name not in MODELS_WITH_SEP_TESTS]
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
        reference_wav = os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0032.wav")
        run_main(main, [*args, "--source_wav", speaker_wav, "--target_wav", reference_wav])
    else:
        # only download the model
        manager.download_model(model_name)
    print(f" | > OK: {model_name}")


@pytest.mark.skipif(GITHUB_ACTIONS, reason="Model too big for CI")
def test_xtts(tmp_path, manager):
    """XTTS is too big to run on github actions. We need to test it locally"""
    model_name = "tts_models/multilingual/multi-dataset/xtts_v1.1"
    model_path, _, _ = manager.download_model(model_name)
    (model_path / "tos_agreed.txt").touch()
    args = [
        "--model_name",
        model_name,
        "--text",
        "C'est un exemple.",
        "--language_idx",
        "fr",
        "--out_path",
        str(tmp_path / "output.wav"),
        "--no-progress_bar",
        "--speaker_wav",
        os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0001.wav"),
        "--use_cuda" if torch.cuda.is_available() else "",
    ]
    run_main(main, args)


@pytest.mark.skipif(GITHUB_ACTIONS, reason="Model too big for CI")
def test_xtts_streaming():
    """Testing the new inference_stream method"""
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    speaker_wav = [os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0001.wav")]
    speaker_wav_2 = os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0002.wav")
    speaker_wav.append(speaker_wav_2)
    model_path = os.path.join(get_user_data_dir("tts"), "tts_models--multilingual--multi-dataset--xtts_v1.1")
    config = XttsConfig()
    config.load_json(os.path.join(model_path, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_path)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print("Computing speaker latents...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=speaker_wav)

    print("Inference...")
    chunks = model.inference_stream(
        "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
        "en",
        gpt_cond_latent,
        speaker_embedding,
    )
    wav_chuncks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            assert chunk.shape[-1] > 5000
        wav_chuncks.append(chunk)
    assert len(wav_chuncks) > 1


@pytest.mark.skipif(GITHUB_ACTIONS, reason="Model too big for CI")
def test_xtts_v2(tmp_path):
    """XTTS is too big to run on github actions. We need to test it locally"""
    args = [
        "--model_name",
        "tts_models/multilingual/multi-dataset/xtts_v2",
        "--text",
        "C'est un exemple.",
        "--language_idx",
        "fr",
        "--out_path",
        str(tmp_path / "output.wav"),
        "--no-progress_bar",
        "--speaker_wav",
        os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0001.wav"),
        os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0002.wav"),
        "--use_cuda" if torch.cuda.is_available() else "",
    ]
    run_main(main, args)


@pytest.mark.skipif(GITHUB_ACTIONS, reason="Model too big for CI")
def test_xtts_v2_streaming():
    """Testing the new inference_stream method"""
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    speaker_wav = [os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0001.wav")]
    model_path = os.path.join(get_user_data_dir("tts"), "tts_models--multilingual--multi-dataset--xtts_v2")
    config = XttsConfig()
    config.load_json(os.path.join(model_path, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_path)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print("Computing speaker latents...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=speaker_wav)

    print("Inference...")
    chunks = model.inference_stream(
        "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
        "en",
        gpt_cond_latent,
        speaker_embedding,
    )
    wav_chuncks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            assert chunk.shape[-1] > 5000
        wav_chuncks.append(chunk)
    assert len(wav_chuncks) > 1
    normal_len = sum([len(chunk) for chunk in wav_chuncks])

    chunks = model.inference_stream(
        "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
        "en",
        gpt_cond_latent,
        speaker_embedding,
        speed=1.5,
    )
    wav_chuncks = []
    for i, chunk in enumerate(chunks):
        wav_chuncks.append(chunk)
    fast_len = sum([len(chunk) for chunk in wav_chuncks])

    chunks = model.inference_stream(
        "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
        "en",
        gpt_cond_latent,
        speaker_embedding,
        speed=0.66,
    )
    wav_chuncks = []
    for i, chunk in enumerate(chunks):
        wav_chuncks.append(chunk)
    slow_len = sum([len(chunk) for chunk in wav_chuncks])

    assert slow_len > normal_len
    assert normal_len > fast_len


@pytest.mark.skipif(GITHUB_ACTIONS, reason="Model too big for CI")
def test_tortoise(tmp_path):
    args = [
        "--model_name",
        "tts_models/en/multi-dataset/tortoise-v2",
        "--text",
        "This is an example.",
        "--out_path",
        str(tmp_path / "output.wav"),
        "--no-progress_bar",
        "--use_cuda" if torch.cuda.is_available() else "",
    ]
    run_main(main, args)


@pytest.mark.skipif(GITHUB_ACTIONS, reason="Model too big for CI")
def test_bark(tmp_path):
    """Bark is too big to run on github actions. We need to test it locally"""
    args = [
        "--model_name",
        "tts_models/multilingual/multi-dataset/bark",
        "tts_models/en/multi-dataset/tortoise-v2",
        "--text",
        "This is an example.",
        "--out_path",
        str(tmp_path / "output.wav"),
        "--no-progress_bar",
        "--use_cuda" if torch.cuda.is_available() else "",
    ]
    run_main(main, args)
    output_path = tmp_path / "output.wav"


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
