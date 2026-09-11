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
