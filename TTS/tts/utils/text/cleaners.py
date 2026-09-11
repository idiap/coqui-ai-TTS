"""Set of default text cleaners.

Original cleaners from (MIT license):
https://github.com/keithito/tacotron/blob/master/text/cleaners.py
"""

import logging
import re
from typing import Protocol, cast
from unicodedata import normalize

from anyascii import anyascii

from TTS.tts.utils.text.chinese_mandarin.numbers import replace_numbers_to_characters_in_text

from .english.abbreviations import abbreviations_en
from .english.number_norm import normalize_numbers as en_normalize_numbers
from .english.time_norm import expand_time_english
from .french.abbreviations import abbreviations_fr
from .italian.abbreviations import abbreviations_it
from .italian.number_norm import normalize_numbers as it_normalize_numbers
from .italian.time_norm import expand_time_italian

logger = logging.getLogger(__name__)


class TextCleaner(Protocol):
    """Protocol that any text cleaner needs to implement."""

    def __call__(self, text: str, lang: str | None) -> str: ...


# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

_uroman = None


def expand_abbreviations(text: str, lang: str | None = "en") -> str:
    if lang == "en":
        _abbreviations = abbreviations_en
    elif lang == "fr":
        _abbreviations = abbreviations_fr
    elif lang == "it":
        _abbreviations = abbreviations_it
    else:
        logger.info("Language %s not supported in expand_abbreviations()", lang)
        return text
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text: str) -> str:
    return text.lower()


def collapse_whitespace(text: str) -> str:
    return re.sub(_whitespace_re, " ", text).strip()


def convert_to_ascii(text: str) -> str:
    return anyascii(text)


def romanize(text: str, language: str | None = None) -> str:
    """Romanize any unicode input with uroman."""
    global _uroman
    if _uroman is None:
        try:
            import uroman
        except ImportError as e:
            msg = "Package not installed: uroman (available in the `languages` extra)"
            raise ImportError(msg) from e
        _uroman = uroman.Uroman()
    return cast(str, _uroman.romanize_string(text, lcode=language))


def remove_aux_symbols(text: str) -> str:
    text = re.sub(r"[\<\>\(\)\[\]\"]+", "", text)
    return text


def replace_symbols(text: str, lang: str | None = "en") -> str:
    """Replace symbols based on the language tag.

    Args:
      text:
       Input text.
      lang:
        Language identifier. ex: "en", "fr", "pt", "ca".

    Returns:
      The modified text
      example:
        input args:
            text: "si l'avi cau, diguem-ho"
            lang: "ca"
        Output:
            text: "si lavi cau, diguemho"
    """
    if lang != "el":  # Semicolons in Greek function as question marks
        text = text.replace(";", ",")
    text = text.replace("-", " ") if lang != "ca" else text.replace("-", "")
    text = text.replace(":", ",")
    if lang == "en":
        text = text.replace("&", " and ")
    elif lang == "fr":
        text = text.replace("&", " et ")
    elif lang in ("it", "pt"):
        text = text.replace("&", " e ")
    elif lang == "ca":
        text = text.replace("&", " i ")
        text = text.replace("'", "")
    return text


def basic_cleaners(text: str, lang: str | None = None) -> str:
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = normalize_unicode(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text: str, lang: str | None = None) -> str:
    """Pipeline for non-English text that transliterates to ASCII."""
    text = normalize_unicode(text)
    # text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def uroman_cleaners(text: str, lang: str | None = None) -> str:
    """Pipeline for romanizing non-Latin text with uroman used by some Fairseq models."""
    text = normalize_unicode(text)
    text = romanize(text, lang)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def basic_german_cleaners(text: str, lang: str | None = "de") -> str:
    """Pipeline for German text."""
    text = normalize_unicode(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


# TODO: elaborate it
def basic_turkish_cleaners(text: str, lang: str | None = "tr") -> str:
    """Pipeline for Turkish text."""
    text = normalize_unicode(text)
    text = text.replace("I", "ı")
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text: str, lang: str | None = "en") -> str:
    """Pipeline for English text, including number and abbreviation expansion."""
    text = normalize_unicode(text)
    # text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_time_english(text)
    text = en_normalize_numbers(text)
    text = expand_abbreviations(text, lang)
    text = replace_symbols(text, lang)
    text = remove_aux_symbols(text)
    text = collapse_whitespace(text)
    return text


def phoneme_cleaners(text: str, lang: str | None = "en") -> str:
    """Pipeline for phonemes mode, including number and abbreviation expansion.

    NB: This cleaner converts numbers into English words, for other languages
    use multilingual_phoneme_cleaners().
    """
    text = normalize_unicode(text)
    text = en_normalize_numbers(text)
    text = expand_abbreviations(text, lang)
    text = replace_symbols(text, lang)
    text = remove_aux_symbols(text)
    text = collapse_whitespace(text)
    return text


def multilingual_phoneme_cleaners(text: str, lang: str | None = None) -> str:
    """Pipeline for phonemes mode, including number and abbreviation expansion."""
    text = normalize_unicode(text)
    text = replace_symbols(text, lang)
    text = remove_aux_symbols(text)
    text = collapse_whitespace(text)
    return text


def italian_cleaners(text: str, lang: str | None = "it") -> str:
    """Pipeline for Italian text: time + light number + abbreviations + symbol cleanup."""
    text = normalize_unicode(text)
    text = lowercase(text)
    text = expand_time_italian(text)
    text = it_normalize_numbers(text)
    text = expand_abbreviations(text, lang)
    text = replace_symbols(text, lang)
    text = remove_aux_symbols(text)
    text = collapse_whitespace(text)
    return text


def french_cleaners(text: str, lang: str | None = "fr") -> str:
    """Pipeline for French text.

    There is no need to expand numbers, phonemizer already does that.
    """
    text = normalize_unicode(text)
    text = expand_abbreviations(text, lang)
    text = lowercase(text)
    text = replace_symbols(text, lang)
    text = remove_aux_symbols(text)
    text = collapse_whitespace(text)
    return text


def portuguese_cleaners(text: str, lang: str | None = "pt") -> str:
    """Basic pipeline for Portuguese text.

    There is no need to expand abbreviation and numbers, phonemizer already does that.
    """
    text = normalize_unicode(text)
    text = lowercase(text)
    text = replace_symbols(text, lang)
    text = remove_aux_symbols(text)
    text = collapse_whitespace(text)
    return text


def chinese_mandarin_cleaners(text: str, lang: str | None = "zh") -> str:
    """Basic pipeline for chinese."""
    text = normalize_unicode(text)
    text = replace_numbers_to_characters_in_text(text)
    return text


def multilingual_cleaners(text: str, lang: str | None = None) -> str:
    """Pipeline for multilingual text."""
    text = normalize_unicode(text)
    text = lowercase(text)
    text = replace_symbols(text, lang)
    text = remove_aux_symbols(text)
    text = collapse_whitespace(text)
    return text


def no_cleaners(text: str, lang: str | None = None) -> str:
    """Only remove newline characters."""
    text = text.replace("\n", "")
    return text


def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters."""
    text = normalize("NFC", text)
    return text
