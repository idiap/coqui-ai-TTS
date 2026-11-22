import pytest

from TTS.utils.generic_utils import slugify


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


@pytest.mark.parametrize(
    "input_text",
    [
        "你好",  # Chinese
        "안녕",  # Korean
        "नमस्ते",  # Hindi
        "مرحبا",  # Arabic
    ],
)
def test_slugify_non_latin_script_is_not_empty(input_text):
    # A script with no ASCII decomposition must not collapse to "": that would
    # make every such name share one filename and silently overwrite each
    # other's cached voice.
    assert slugify(input_text) != ""


def test_slugify_non_latin_script_names_stay_distinct():
    assert slugify("你好") != slugify("안녕")


def test_slugify_non_latin_script_is_stable():
    assert slugify("नमस्ते") == slugify("नमस्ते")
