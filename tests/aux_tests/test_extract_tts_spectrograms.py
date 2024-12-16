from pathlib import Path

import pytest
import torch

from tests import get_tests_input_path, run_cli
from TTS.config import load_config
from TTS.tts.models import setup_model

torch.manual_seed(1)


@pytest.mark.parametrize("model", ["glow_tts", "tacotron", "tacotron2"])
def test_extract_tts_spectrograms(tmp_path, model):
    config_path = Path(get_tests_input_path()) / f"test_{model}_config.json"
    checkpoint_path = tmp_path / f"{model}.pth"
    output_path = tmp_path / "output_extract_tts_spectrograms"

    c = load_config(str(config_path))
    model = setup_model(c)
    torch.save({"model": model.state_dict()}, checkpoint_path)
    run_cli(
        f'CUDA_VISIBLE_DEVICES="" python TTS/bin/extract_tts_spectrograms.py --config_path "{config_path}" --checkpoint_path "{checkpoint_path}" --output_path "{output_path}"'
    )
