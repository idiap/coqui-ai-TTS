from pathlib import Path

import pytest
import torch

from tests import get_tests_data_path, get_tests_input_path, run_main
from TTS.bin.compute_attention_masks import main
from TTS.config import load_config
from TTS.tts.models import setup_model

torch.manual_seed(1)


@pytest.mark.parametrize("model", ["tacotron", "tacotron2"])
def test_compute_attention_masks(tmp_path, model):
    config_path = str(Path(get_tests_input_path()) / f"test_{model}_config.json")
    checkpoint_path = str(tmp_path / f"{model}.pth")
    output_path = str(tmp_path / "output_compute_attention_masks")
    data_path = str(Path(get_tests_data_path()) / "ljspeech")
    metafile = str(Path(get_tests_data_path()) / "ljspeech" / "metadata.csv")

    config = load_config(config_path)
    model = setup_model(config)
    torch.save({"model": model.state_dict()}, checkpoint_path)
    run_main(
        main,
        [
            "--config_path",
            config_path,
            "--model_path",
            checkpoint_path,
            "--output_path",
            output_path,
            "--data_path",
            data_path,
            # "--dataset_metafile",
            # metafile,
        ],
    )
