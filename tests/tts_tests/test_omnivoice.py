import json
import sys
from types import ModuleType

import numpy as np
import torch

from TTS.config import load_config
from TTS.tts.configs.omnivoice_config import OmnivoiceConfig
from TTS.tts.models.omnivoice import Omnivoice


class _FakeOmniVoiceModel(torch.nn.Module):
    sampling_rate = 24000

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        assert path == "/model"
        assert kwargs["dtype"] == torch.float32
        return cls()

    @staticmethod
    def supported_language_ids():
        # The adapter must not restrict OmniVoice to its example languages.
        return {"vi", "en", "aae", "yue"}

    @staticmethod
    def generate(**kwargs):
        assert kwargs["text"] == "Hello"
        assert kwargs["language"] == "en"
        return [np.zeros(240, dtype=np.float32)]


def test_load_transformers_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model_type": "omnivoice", "architectures": ["OmniVoice"]}))
    config = load_config(config_path)
    assert isinstance(config, OmnivoiceConfig)


def test_load_and_synthesize_without_optional_dependency(monkeypatch):
    module = ModuleType("omnivoice")
    module.OmniVoice = _FakeOmniVoiceModel
    monkeypatch.setitem(sys.modules, "omnivoice", module)

    model = Omnivoice(OmnivoiceConfig())
    model.load_checkpoint(model.config, "/model", eval=True)
    result = model.synthesize("Hello", language="en")

    assert result["wav"].shape == (240,)
    assert model.language_manager.language_names == ["aae", "en", "vi", "yue"]
    assert model.config.languages == ["aae", "en", "vi", "yue"]
