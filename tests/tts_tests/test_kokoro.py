import pytest
import torch

from TTS.tts.models.kokoro import Kokoro, KokoroCharacters
from TTS.tts.utils.text.phonemizers.espeak_wrapper import _is_tool

# Minimal version of the original config.json (with a partial vocabulary)
ORIG_CONFIG = {
    "istftnet": {
        "upsample_kernel_sizes": [20, 12],
        "upsample_rates": [10, 6],
        "gen_istft_hop_size": 5,
        "gen_istft_n_fft": 20,
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "resblock_kernel_sizes": [3, 7, 11],
        "upsample_initial_channel": 512,
    },
    "dim_in": 64,
    "dropout": 0.2,
    "hidden_dim": 512,
    "max_conv_dim": 512,
    "max_dur": 50,
    "multispeaker": True,
    "n_layer": 3,
    "n_mels": 80,
    "n_token": 178,
    "style_dim": 128,
    "text_encoder_kernel_size": 5,
    "plbert": {
        "hidden_size": 64,
        "num_attention_heads": 2,
        "intermediate_size": 128,
        "max_position_embeddings": 512,
        "num_hidden_layers": 2,
        "dropout": 0.1,
    },
    "vocab": {",": 3, ".": 4, "!": 5, " ": 16, "I": 25, "O": 31, "Y": 41, "b": 44, "f": 48, "h": 50}
    | {"l": 54, "v": 64, "z": 68, "ə": 83, "ˈ": 156},
}


@pytest.fixture(scope="module")
def model() -> Kokoro:
    model = Kokoro(Kokoro._convert_original_config(ORIG_CONFIG)).eval()
    model.voices = {"af_test": torch.randn(510, 1, 256) * 0.1, "bm_test": torch.randn(510, 1, 256) * 0.1}
    return model


def test_characters(model):
    characters = model.tokenizer.characters
    assert isinstance(characters, KokoroCharacters)
    assert characters.num_chars == 178
    assert characters.pad_id == characters.bos_id == characters.eos_id == 0
    for char, idx in ORIG_CONFIG["vocab"].items():
        assert characters.char_to_id(char) == idx
        assert characters.id_to_char(idx) == char
    # Round trip through the config
    restored, _ = KokoroCharacters.init_from_config(model.config)
    assert restored.vocab == characters.vocab
    assert characters.to_config().vocab_dict == model.config.characters.vocab_dict


def test_inference(model):
    tokens = torch.tensor([[0, 50, 83, 54, 156, 31, 3, 16, 48, 156, 25, 64, 5, 0]])
    style = model.voices["af_test"][11]
    torch.manual_seed(0)
    outputs = model.inference(tokens, aux_input={"style": style})
    num_frames = int(outputs["durations"].sum())
    assert outputs["alignments"].shape == (tokens.shape[1], num_frames)
    # 2 frames per alignment frame, then 300 samples per frame
    assert outputs["model_outputs"].shape == (1, 1, num_frames * 2 * 300)

    torch.manual_seed(0)
    fast = model.inference(tokens, aux_input={"style": style}, speed=2.0)
    assert fast["durations"].sum() < outputs["durations"].sum()


def test_get_voice(model):
    mixed = model.get_voice("af_test, bm_test")
    torch.testing.assert_close(mixed, (model.voices["af_test"] + model.voices["bm_test"]) / 2)
    with pytest.raises(ValueError, match="not found"):
        model.get_voice("af_missing")


def test_synthesize_errors(model):
    with pytest.raises(ValueError, match="requires a voice"):
        model.synthesize("Hello")
    with pytest.raises(ValueError, match="voice cloning"):
        model.synthesize("Hello", speaker_wav="test.wav")


@pytest.mark.skipif(not _is_tool("espeak-ng"), reason="espeak-ng not installed")
def test_synthesize(model):
    outputs = model.synthesize("Hello, five boys!", speaker="af_test")
    assert outputs["wav"].ndim == 1
    assert len(outputs["wav"]) > 0
    # Pad tokens are added at start and end
    assert outputs["text_inputs"][0, 0] == outputs["text_inputs"][0, -1] == 0
