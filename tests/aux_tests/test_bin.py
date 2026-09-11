from pathlib import Path

import pytest
import torch

from tests import get_tests_data_path, get_tests_input_path, run_main
from TTS.bin.collect_env_info import main as collect_env_info
from TTS.bin.compute_attention_masks import main as compute_attention_masks
from TTS.bin.compute_embeddings import main as compute_embeddings
from TTS.config import load_config
from TTS.tts.models import setup_model


def test_collect_env_info():
    collect_env_info()


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
        compute_attention_masks,
        [
            "--config_path",
            config_path,
            "--model_path",
            checkpoint_path,
            "--output_path",
            output_path,
            "--data_path",
            data_path,
        ],
    )


def test_compute_embeddings(tmp_path):
    data_path = Path(get_tests_data_path())
    run_main(
        compute_embeddings,
        [
            "--output_path",
            str(tmp_path / "speakers.pth"),
            "--formatter_name",
            "ljspeech",
            "--dataset_path",
            str(data_path / "ljspeech"),
            "--dataset_name",
            "ljspeech",
            "--meta_file_train",
            "metadata.csv",
            "--no_eval",
        ],
    )
