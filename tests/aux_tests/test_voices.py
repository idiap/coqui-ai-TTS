import logging

import numpy as np
import pytest
import torch

from TTS.tts.models.bark import Bark
from TTS.utils.voices import CloningMixin


class DummyCloner(CloningMixin):
    def _clone_voice(self, speaker_wav, **kwargs):
        return {"emb": torch.ones(2)}, {"name": "dummy"}


@pytest.fixture
def cloner():
    return DummyCloner()


@pytest.fixture
def bark():
    # Voice file handling doesn't depend on model state, so skip loading weights
    return Bark.__new__(Bark)


@pytest.mark.parametrize("speaker_id", ["my.voice", "Craig Gutsy", "Zoë", "../escape"])
def test_clone_and_load_voice(cloner, tmp_path, speaker_id):
    cloner.clone_voice("x.wav", speaker_id, tmp_path)
    voice = cloner.clone_voice(None, speaker_id, tmp_path)
    assert torch.equal(voice["emb"], torch.ones(2))


@pytest.mark.parametrize("load_id", ["my.voice", "my_voice", "my voice"])
def test_load_voice_equivalent_ids(cloner, tmp_path, load_id):
    cloner.clone_voice("x.wav", "my.voice", tmp_path)
    voice = cloner.load_voice_file(load_id, tmp_path)
    assert torch.equal(voice["emb"], torch.ones(2))


def test_clone_voice_overwrites_same_slug(cloner, tmp_path, caplog):
    cloner.clone_voice("x.wav", "my.voice", tmp_path)
    with caplog.at_level(logging.INFO, logger="TTS.utils.voices"):
        cloner.clone_voice("x.wav", "my voice", tmp_path)
    assert "already exists" in caplog.text
    assert list(cloner.get_voices(tmp_path)) == ["my_voice"]


def test_clone_voice_file_and_metadata(cloner, tmp_path):
    voice_dir = tmp_path / "voices"
    cloner.clone_voice("x.wav", "my.voice", voice_dir)
    voice_fn = voice_dir / "my_voice.pth"
    assert voice_fn.is_file()
    voice = torch.load(voice_fn, weights_only=True)
    assert voice["metadata"]["speaker_id"] == "my_voice"
    assert voice["metadata"]["source_files"] == ["x.wav"]

    cloner.clone_voice("x.wav", "../escape", voice_dir)
    assert (voice_dir / "escape.pth").is_file()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["voices"]


def test_load_voice_file_not_found(cloner, tmp_path):
    with pytest.raises(FileNotFoundError, match=r"`my_voice\.pth` for speaker `my\.voice`"):
        cloner.load_voice_file("my.voice", tmp_path)


def test_bark_load_npz_voice(bark, tmp_path):
    prompts = {
        "semantic_prompt": np.arange(3),
        "coarse_prompt": np.ones((2, 3)),
        "fine_prompt": np.zeros((2, 3)),
    }
    np.savez(tmp_path / "my_voice.npz", **prompts)
    voice = bark.load_voice_file("my.voice", tmp_path)
    assert set(voice) == set(prompts)
    for key, value in prompts.items():
        assert torch.equal(voice[key], torch.tensor(value))


def test_bark_load_pth_voice(bark, tmp_path):
    torch.save({"semantic_prompt": torch.arange(3)}, tmp_path / "my_voice.pth")
    voice = bark.load_voice_file("my.voice", tmp_path)
    assert torch.equal(voice["semantic_prompt"], torch.arange(3))


def test_bark_load_voice_file_not_found(bark, tmp_path):
    with pytest.raises(FileNotFoundError, match=r"`my_voice\.pth` or \.npz for speaker `my\.voice`"):
        bark.load_voice_file("my.voice", tmp_path)
