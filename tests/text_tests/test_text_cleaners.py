#!/usr/bin/env python3

from TTS.tts.utils.text.cleaners import (
    english_cleaners,
    multilingual_phoneme_cleaners,
    normalize_unicode,
    phoneme_cleaners,
)


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
        ("\u4E2D\u56FD", "中国"),
        ("niño", "niño"),
        ("a\u0308", "ä"),
        ("\u3053\u3093\u306b\u3061\u306f", "こんにちは"),
        ("\u03B1\u03B2", "αβ"),
    ]
    for arg, expect in test_cases:
        assert normalize_unicode(arg) == expect
