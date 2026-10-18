"""These tests should be run locally because the models are too big for CI."""

import os

import pytest
import torch

from tests import get_tests_data_path, run_main
from TTS.bin.synthesize import main
from TTS.utils.manage import ModelManager

GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("COQUI_TOS_AGREED", "1")


@pytest.fixture
def manager():
    """Set up model manager."""
    return ModelManager(progress_bar=False)


@pytest.mark.skipif(GITHUB_ACTIONS, reason="Model too big for CI")
def test_xtts(tmp_path, check_output):
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
    check_output(tmp_path / "output.wav")


@pytest.mark.skipif(GITHUB_ACTIONS, reason="Model too big for CI")
def test_xtts_streaming(manager, device: torch.device, check_output):
    """Testing the new inference_stream method"""
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    speaker_wav = [os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0001.wav")]
    speaker_wav_2 = os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0002.wav")
    speaker_wav.append(speaker_wav_2)
    model_path, config_path, _ = manager.download_model("tts_models/multilingual/multi-dataset/xtts_v1.1")
    config = XttsConfig()
    config.load_json(config_path)
    model = Xtts(config)
    model.load_checkpoint(config, checkpoint_dir=str(model_path.parent))
    model.to(device)

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
    check_output(torch.cat(wav_chunks), config.audio.output_sample_rate)


@pytest.mark.skipif(GITHUB_ACTIONS, reason="Model too big for CI")
def test_xtts_v2(tmp_path, check_output):
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
    check_output(tmp_path / "output.wav")


@pytest.mark.skipif(GITHUB_ACTIONS, reason="Model too big for CI")
def test_xtts_v2_streaming(manager, device: torch.device, check_output):
    """Testing the new inference_stream method"""
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    speaker_wav = [os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0001.wav")]
    model_path, config_path, _ = manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
    config = XttsConfig()
    config.load_json(config_path)
    model = Xtts(config)
    model.load_checkpoint(config, checkpoint_dir=str(model_path.parent))
    model.to(device)

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
    check_output(torch.cat(wav_chunks), config.audio.output_sample_rate)
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
    check_output(torch.cat(wav_chunks), config.audio.output_sample_rate, name="fast")
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
    check_output(torch.cat(wav_chunks), config.audio.output_sample_rate, name="slow")
    slow_len = sum([len(chunk) for chunk in wav_chunks])

    assert slow_len > normal_len
    assert normal_len > fast_len


@pytest.mark.skipif(GITHUB_ACTIONS, reason="Model too big for CI")
def test_tortoise(tmp_path, check_output, monkeypatch):
    from TTS.tts.models import tortoise

    # Tortoise seeds itself with the current time by default
    deterministic_state = tortoise.deterministic_state
    monkeypatch.setattr(tortoise, "deterministic_state", lambda seed=None: deterministic_state(seed or 1))

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
    check_output(tmp_path / "output.wav")


@pytest.mark.skipif(GITHUB_ACTIONS, reason="Model too big for CI")
def test_bark(tmp_path, check_output):
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
    check_output(tmp_path / "output.wav")
