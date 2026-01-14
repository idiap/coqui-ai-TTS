from pathlib import Path

import pytest
import torch
from trainer.io import save_checkpoint

from TTS.config import load_config
from TTS.config.shared_configs import BaseAudioConfig
from TTS.encoder.configs.speaker_encoder_config import SpeakerEncoderConfig
from TTS.encoder.utils.generic_utils import setup_encoder_model


@pytest.fixture(scope="session", autouse=True)
def set_seed() -> None:
    """Set random seed for reproducibility across all tests."""
    torch.manual_seed(1)


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Provide torch device (CUDA if available, otherwise CPU)."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def encoder_config_path(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    """Create and save speaker encoder config.

    Can be parameterized with model_params dict to customize config:
        @pytest.mark.parametrize('encoder_config_path', [{'use_torch_spec': True}], indirect=True)
    """
    custom_model_params = getattr(request, "param", {})

    audio = BaseAudioConfig(
        num_mels=40,
        fft_size=400,
        sample_rate=16000,
        win_length=400,
        hop_length=160,
        preemphasis=0.98,
        mel_fmax=8000,
        trim_db=60,
    )
    model_params = {
        "model_name": "lstm",
        "input_dim": 40,
        "proj_dim": 256,
        "lstm_dim": 768,
        "num_lstm_layers": 3,
        "use_lstm_with_projection": True,
    }
    model_params.update(custom_model_params)

    config = SpeakerEncoderConfig(
        num_loader_workers=8,
        num_classes_in_batch=64,
        num_utter_per_class=10,
        audio=audio,
        model_params=model_params,
    )

    out_path = tmp_path / "encoder_config.json"
    config.save_json(out_path)
    return out_path


@pytest.fixture
def encoder_model_path(encoder_config_path: Path, tmp_path: Path):
    """Create and save speaker encoder model"""
    step = 0
    config = load_config(encoder_config_path)
    config.audio.resample = True
    model = setup_encoder_model(config)

    encoder_model_path = tmp_path / f"checkpoint_{step}.pth"
    save_checkpoint(config, model, tmp_path, current_step=step, epoch=0)

    return encoder_model_path
