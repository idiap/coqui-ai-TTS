import json
from pathlib import Path

import torch
from trainer.io import save_checkpoint

from tests import get_tests_input_path
from TTS.config import load_config
from TTS.tts.models import setup_model
from TTS.utils.manage import ModelManager

TTS_CONFIG = Path(get_tests_input_path()) / "dummy_model_config.json"
MODEL_NAME = "tts_models/en/test/model"


class ExternalFormatModel:
    """Stand-in for a model whose checkpoints are not in Coqui format."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir

    @classmethod
    def from_original(cls, model_dir: Path) -> "ExternalFormatModel":
        return cls(model_dir)


def _manager(tmp_path: Path, model_item: dict) -> ModelManager:
    models_file = tmp_path / "models.json"
    models_file.write_text(json.dumps({"tts_models": {"en": {"test": {"model": model_item}}}}))
    return ModelManager(models_file=models_file, output_prefix=tmp_path / "models", progress_bar=False)


def test_load_model_coqui_format(tmp_path):
    """Models in Coqui format are loaded with from_pretrained() of the class given in their config."""
    manager = _manager(tmp_path, {"github_rls_url": "https://example.com/unused.zip", "default_vocoder": None})
    # Pretend the model was already downloaded
    model_dir = manager.output_prefix / "tts_models--en--test--model"
    model_dir.mkdir(parents=True)
    config = load_config(str(TTS_CONFIG))
    config.save_json(str(model_dir / "config.json"))
    model = setup_model(config)
    save_checkpoint(config, model, str(model_dir), current_step=1, epoch=1)

    loaded = manager.load_model(MODEL_NAME)

    assert type(loaded) is type(model)
    assert not loaded.training
    for (name, expected), actual in zip(model.state_dict().items(), loaded.state_dict().values(), strict=True):
        assert torch.equal(expected, actual), name


def test_load_model_custom_loader(tmp_path):
    """Models in other formats are loaded with the model_class and loader given in .models.json."""
    manager = _manager(
        tmp_path,
        {
            "model_class": f"{__name__}.ExternalFormatModel",
            "loader": "from_original",
            "default_vocoder": None,
        },
    )

    loaded = manager.load_model(MODEL_NAME)

    assert type(loaded).__name__ == "ExternalFormatModel"
    assert loaded.model_dir == manager.output_prefix / "tts_models--en--test--model"


def test_download_model_coqui_format(tmp_path):
    manager = _manager(tmp_path, {"github_rls_url": "https://example.com/unused.zip", "default_vocoder": None})
    model_dir = manager.output_prefix / "tts_models--en--test--model"
    model_dir.mkdir(parents=True)
    config = load_config(str(TTS_CONFIG))
    config.save_json(str(model_dir / "config.json"))
    save_checkpoint(config, setup_model(config), str(model_dir), current_step=1, epoch=1)

    model_path, config_path, _ = manager.download_model(MODEL_NAME)

    assert model_path == model_dir / "checkpoint_1.pth"
    assert config_path == model_dir / "config.json"


def test_download_model_custom_loader(tmp_path):
    """Models in other formats return the directory with all files and no config."""
    manager = _manager(
        tmp_path, {"model_class": f"{__name__}.ExternalFormatModel", "loader": "from_original", "default_vocoder": None}
    )

    model_path, config_path, _ = manager.download_model(MODEL_NAME)

    assert model_path == manager.output_prefix / "tts_models--en--test--model"
    assert config_path is None
