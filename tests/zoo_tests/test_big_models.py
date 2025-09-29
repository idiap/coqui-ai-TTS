"""These tests should be run locally because the models are too big for CI."""

import os

import pytest
import torch

from tests import get_tests_data_path, run_main
from TTS.bin.synthesize import main
from TTS.utils.manage import ModelManager

GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.fixture(scope="session", autouse=True)
def set_env():
    os.environ["COQUI_TOS_AGREED"] = "1"


@pytest.fixture
def manager():
    """Set up model manager."""
    return ModelManager(progress_bar=False)


@pytest.mark.skipif(GITHUB_ACTIONS, reason="Model too big for CI")
def test_xtts(tmp_path):
    """XTTS is too big to run on github actions. We need to test it locally"""
    args = [
        "--model_name",
        "tts_models/multilingual/multi-dataset/xtts_v1.1",
        "--text",
        "C'est un exemple.",
        "--language_idx",
        "fr",
        "--out_path",
        str(tmp_path / "output.wav"),
        "--no-progress_bar",
        "--speaker_wav",
        os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0001.wav"),
    ]
    if torch.cuda.is_available():
        args.append("--use_cuda")
    run_main(main, args)


@pytest.mark.skipif(GITHUB_ACTIONS, reason="Model too big for CI")
def test_xtts_streaming(manager):
    """Testing the new inference_stream method"""
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    speaker_wav = [os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0001.wav")]
    speaker_wav_2 = os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0002.wav")
    speaker_wav.append(speaker_wav_2)
    model_path, _, _ = manager.download_model("tts_models/multilingual/multi-dataset/xtts_v1.1")
    config = XttsConfig()
    config.load_json(model_path / "config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=str(model_path))
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
    wav_chunks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            assert chunk.shape[-1] > 5000
        wav_chunks.append(chunk)
    assert len(wav_chunks) > 1


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
    ]
    if torch.cuda.is_available():
        args.append("--use_cuda")
    run_main(main, args)


@pytest.mark.skipif(GITHUB_ACTIONS, reason="Model too big for CI")
def test_xtts_v2_streaming(manager):
    """Testing the new inference_stream method"""
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    speaker_wav = [os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0001.wav")]
    model_path, _, _ = manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
    config = XttsConfig()
    config.load_json(model_path / "config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=str(model_path))
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
    wav_chunks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            assert chunk.shape[-1] > 5000
        wav_chunks.append(chunk)
    assert len(wav_chunks) > 1
    normal_len = sum([len(chunk) for chunk in wav_chunks])

    chunks = model.inference_stream(
        "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
        "en",
        gpt_cond_latent,
        speaker_embedding,
        speed=1.5,
    )
    wav_chunks = []
    for i, chunk in enumerate(chunks):
        wav_chunks.append(chunk)
    fast_len = sum([len(chunk) for chunk in wav_chunks])

    chunks = model.inference_stream(
        "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
        "en",
        gpt_cond_latent,
        speaker_embedding,
        speed=0.66,
    )
    wav_chunks = []
    for i, chunk in enumerate(chunks):
        wav_chunks.append(chunk)
    slow_len = sum([len(chunk) for chunk in wav_chunks])

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
    ]
    if torch.cuda.is_available():
        args.append("--use_cuda")
    run_main(main, args)


@pytest.mark.skipif(GITHUB_ACTIONS, reason="Model too big for CI")
def test_bark(tmp_path):
    """Bark is too big to run on github actions. We need to test it locally"""
    args = [
        "--model_name",
        "tts_models/multilingual/multi-dataset/bark",
        "--text",
        "This is an example.",
        "--out_path",
        str(tmp_path / "output.wav"),
        "--no-progress_bar",
    ]
    if torch.cuda.is_available():
        args.append("--use_cuda")
    run_main(main, args)
