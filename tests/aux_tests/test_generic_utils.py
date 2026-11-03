import pytest
import torch

from TTS.utils.generic_utils import resolve_device, select_device, slugify


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        # ASCII-safe names
        ("Craig Gutsy", "Craig_Gutsy"),
        ("Dr.Jane", "Dr_Jane"),
        ("Mr-Smith_01", "Mr-Smith_01"),
        # Names with accents and spaces
        ("Zoë", "Zoe"),
        ("Renée O'Connor", "Renee_O_Connor"),
        ("Jürgen Müller", "Jurgen_Muller"),
        # Unsafe punctuation and repeated underscores
        ("Hello!!!", "Hello"),
        ("foo///bar", "foo_bar"),
        (" a  b ", "a_b"),
        # Emojis and symbols
        ("Speaker 🔥 #1", "Speaker_1"),
        # Edge cases
        ("", ""),
    ],
)
def test_slugify(input_text, expected_output):
    assert slugify(input_text) == expected_output


def test_resolve_device():
    assert resolve_device("cpu", use_cuda=False) == "cpu"
    assert resolve_device(None, use_cuda=False) is None
    with pytest.warns(DeprecationWarning, match="`use_cuda` is deprecated.*`device='cuda'`"):
        assert resolve_device("cpu", use_cuda=True) == "cuda"
    with pytest.warns(DeprecationWarning, match="`gpu` is deprecated.*`.to\\('cuda'\\)`"):
        assert resolve_device(None, use_cuda=True, arg_name="gpu", replacement="`.to('cuda')`") == "cuda"


def test_select_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device(None) == torch.device("cpu")
    assert select_device("meta") == torch.device("meta")
    with pytest.warns(DeprecationWarning, match="`--use_cuda` is deprecated"):
        assert select_device("meta", use_cuda=True, arg_name="--use_cuda") == torch.device("cuda")
    with pytest.warns(DeprecationWarning):
        assert select_device(None, use_cuda=False) == torch.device("cpu")
