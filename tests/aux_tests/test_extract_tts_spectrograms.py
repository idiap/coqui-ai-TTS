from pathlib import Path

import pytest
import torch

from tests import get_tests_input_path, run_main
from TTS.bin.extract_tts_spectrograms import main
from TTS.config import load_config
from TTS.tts.models import setup_model

torch.manual_seed(1)


@pytest.mark.parametrize("model", ["glow_tts", "tacotron", "tacotron2"])
def test_extract_tts_spectrograms(tmp_path, model):
    config_path = str(Path(get_tests_input_path()) / f"test_{model}_config.json")
    checkpoint_path = str(tmp_path / f"{model}.pth")
    output_path = str(tmp_path / "output_extract_tts_spectrograms")

    config = load_config(config_path)
    model = setup_model(config)
    torch.save({"model": model.state_dict()}, checkpoint_path)
    run_main(main, ["--config_path", config_path, "--checkpoint_path", checkpoint_path, "--output_path", output_path])
