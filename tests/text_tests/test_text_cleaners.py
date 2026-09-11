import pytest

from TTS.tts.utils.text.characters import Graphemes
from TTS.tts.utils.text.cleaners import (
    english_cleaners,
    italian_cleaners,
    multilingual_cleaners,
    multilingual_phoneme_cleaners,
    normalize_unicode,
    phoneme_cleaners,
    replace_symbols,
    romanize,
)
from TTS.tts.utils.text.tokenizer import TTSTokenizer


def test_time() -> None:
    assert english_cleaners("It's 11:00") == "it's eleven a m"
    assert english_cleaners("It's 9:01") == "it's nine oh one a m"
    assert english_cleaners("It's 16:00") == "it's four p m"
    assert english_cleaners("It's 00:00 am") == "it's twelve a m"


def test_currency() -> None:
    assert phoneme_cleaners("It's $10.50") == "It's ten dollars fifty cents"
    assert phoneme_cleaners("£1.1") == "one pound sterling one penny"
    assert phoneme_cleaners("¥1") == "one yen"


def test_expand_numbers() -> None:
    assert phoneme_cleaners("-1") == "minus one"
    assert phoneme_cleaners("1") == "one"
    assert phoneme_cleaners("1" + "0" * 35) == "one hundred decillion"
    assert phoneme_cleaners("1" + "0" * 36) == "one" + " zero" * 36


def test_multilingual_phoneme_cleaners() -> None:
    assert multilingual_phoneme_cleaners("(Hello)") == "Hello"
    assert multilingual_phoneme_cleaners("1:") == "1,"


def test_normalize_unicode() -> None:
    test_cases = [
        ("Häagen-Dazs", "Häagen-Dazs"),
        ("你好!", "你好!"),
        ("𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼", "𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼"),
        ("é", "é"),
        ("e\u0301", "é"),
        ("a\u0300", "à"),
        ("a\u0327", "a̧"),
        ("na\u0303", "nã"),
        ("o\u0302u", "ôu"),
        ("n\u0303", "ñ"),
        ("\u4e2d\u56fd", "中国"),
        ("niño", "niño"),
        ("a\u0308", "ä"),
        ("\u3053\u3093\u306b\u3061\u306f", "こんにちは"),
        ("\u03b1\u03b2", "αβ"),
    ]
    for arg, expect in test_cases:
        assert normalize_unicode(arg) == expect


@pytest.mark.parametrize(
    ("text", "language", "expected"),
    [
        ("Игорь Стравинский", None, "Igor Stravinsky"),
        ("Игорь Стравинский", "ukr", "Yhor Stravynsky"),
        ("안녕하세요.", None, "annyeonghaseyo."),
    ],
)
def test_romanize(text, language, expected) -> None:
    assert romanize(text, language) == expected


def test_italian_cleaners_numbers_and_abbreviations() -> None:
    text = "Alle 09:05 il Sig. Bianchi ha pagato €1.234,50 per 50%"
    expected = "alle 9 e cinque il signor bianchi ha pagato 1234 virgola 50 euro per 50 percento"
    assert italian_cleaners(text) == expected


def test_italian_cleaners_temperature_and_time() -> None:
    text = "Temperatura: 3.5\N{DEGREE SIGN}C alle 14.00"
    expected = "temperatura, 3 virgola 5 gradi celsius alle 14"
    assert italian_cleaners(text) == expected


@pytest.mark.parametrize(
    ("text", "lang", "expected"),
    [
        # Semicolon is replaced with comma for most languages
        ("Hello; world", "en", "Hello, world"),
        ("Bonjour; monde", "fr", "Bonjour, monde"),
        # Greek semicolon (;) functions as question mark, so it's preserved
        ("Πού είναι;", "el", "Πού είναι;"),
        # Hyphen handling varies by language
        ("well-known", "en", "well known"),
        ("well-known", "ca", "wellknown"),
        # Ampersand expansion
        ("Tom & Jerry", "en", "Tom  and  Jerry"),
        ("Tom & Jerry", "fr", "Tom  et  Jerry"),
        ("Tom & Jerry", "it", "Tom  e  Jerry"),
        ("Tom & Jerry", "pt", "Tom  e  Jerry"),
        ("Tom & Jerry", "ca", "Tom  i  Jerry"),
    ],
)
def test_replace_symbols(text: str, lang: str, expected: str) -> None:
    assert replace_symbols(text, lang) == expected


@pytest.fixture(scope="module")
def tokenizer_with_cleaner():
    """TTSTokenizer with multilingual_phoneme_cleaners for integrated testing."""
    return TTSTokenizer(
        use_phonemes=False,
        text_cleaner=multilingual_cleaners,
        characters=Graphemes(),
    )


@pytest.mark.parametrize(
    ("text", "lang", "expected"),
    [
        # Greek: semicolon preserved (it functions as question mark)
        ("Hello; world", "el", "hello; world"),
        # English: semicolon converted to comma
        ("Hello; world", "en", "hello, world"),
    ],
)
def test_tokenizer_language_dependent_cleaning(
    tokenizer_with_cleaner: TTSTokenizer, text: str, lang: str, expected: str
) -> None:
    """Test that language-dependent cleaning works through the tokenizer pipeline."""
    token_ids = tokenizer_with_cleaner.text_to_ids(text, language=lang)
    result = tokenizer_with_cleaner.ids_to_text(token_ids)
    assert result == expected
