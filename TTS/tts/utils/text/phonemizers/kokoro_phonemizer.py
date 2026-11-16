"""Phonemizer producing the phoneme set of Kokoro models.

Kokoro was trained on phonemes from misaki (https://github.com/hexgrad/misaki).
For non-English languages, misaki uses espeak with a few replacements to map
its output to the Kokoro phoneme set. These replacements are reproduced here,
so misaki is not required. For English, misaki mainly uses its own lexicons and
only falls back to espeak for unknown words. If misaki is installed, it is used
for English, otherwise espeak with misaki's fallback replacements.

The mappings are adapted from misaki (Apache-2.0 license):
https://github.com/hexgrad/misaki/blob/main/misaki/espeak.py
"""

import logging
import re
from functools import cache
from typing import Any

from TTS.tts.utils.text.phonemizers.base import BasePhonemizer
from TTS.tts.utils.text.phonemizers.espeak_wrapper import ESpeak

logger = logging.getLogger(__name__)

# Language names used by Kokoro and the corresponding espeak voices
_LANGUAGES = {
    "en-us": "en-us",
    "en-gb": "en-gb",
    "es": "es",
    "fr-fr": "fr-fr",
    "hi": "hi",
    "it": "it",
    "pt-br": "pt-br",
}

_PUNCTUATIONS = ';:,.!?¡¿—…"«»“”()'

# Applied to espeak output for English (misaki's EspeakFallback.E2M)
_EN_E2M = sorted(
    {
        "ʔˌn̩": "ʔn",
        "ʔn̩": "ʔn",
        "a^ɪ": "I",
        "a^ʊ": "W",
        "d^ʒ": "ʤ",
        "e^ɪ": "A",
        "e": "A",
        "t^ʃ": "ʧ",
        "ɔ^ɪ": "Y",
        "ə^l": "ᵊl",
        "ʲo": "jo",
        "ʲə": "jə",
        "ʲ": "",
        "ɚ": "əɹ",
        "r": "ɹ",
        "x": "k",
        "ç": "k",
        "ɐ": "ə",
        "ɬ": "l",
        "̃": "",
    }.items(),
    key=lambda kv: -len(kv[0]),
)

# Applied to espeak output for other languages (misaki's EspeakG2P.e2m)
_E2M = sorted(
    {
        "a^ɪ": "I",
        "a^ʊ": "W",
        "d^z": "ʣ",
        "d^ʒ": "ʤ",
        "e^ɪ": "A",
        "o^ʊ": "O",
        "ə^ʊ": "Q",
        "s^s": "S",
        "t^s": "ʦ",
        "t^ʃ": "ʧ",
        "ɔ^ɪ": "Y",
    }.items()
)


def _map_english(phonemes: str, *, british: bool) -> str:
    """Map espeak output to Kokoro phonemes for English (misaki's EspeakFallback)."""
    if british:
        # NB: misaki applies this after _EN_E2M, where it never matches because
        # e -> A. That only affects its rare fallback, but here espeak is used
        # for all words, so apply it first to match misaki's lexicons.
        phonemes = phonemes.replace("e^ə", "ɛː")
    for old, new in _EN_E2M:
        phonemes = phonemes.replace(old, new)
    phonemes = re.sub(r"(\S)̩", r"ᵊ\1", phonemes).replace("̩", "")
    if british:
        phonemes = phonemes.replace("iə", "ɪə").replace("ə^ʊ", "Q")
    else:
        phonemes = phonemes.replace("o^ʊ", "O").replace("ɜːɹ", "ɜɹ").replace("ɜː", "ɜɹ")
        phonemes = phonemes.replace("ɪə", "iə").replace("ː", "")
    phonemes = phonemes.replace("o", "ɔ")  # for espeak < 1.52
    phonemes = phonemes.replace("ɾ", "T").replace("ʔ", "t")
    # espeak's linking r after ɚ ("weather is" -> ðɚɹ ɪz) would otherwise result in əɹɹ
    return phonemes.replace("^", "").replace("ɹɹ", "ɹ")


def _map_other(phonemes: str) -> str:
    """Map espeak output to Kokoro phonemes for non-English languages (misaki's EspeakG2P)."""
    for old, new in _E2M:
        phonemes = phonemes.replace(old, new)
    return phonemes.replace("^", "").replace("-", "")


@cache
def _load_misaki_g2p(language: str) -> Any | None:
    """Return misaki's English G2P for the given language or None if misaki is not available."""
    try:
        from misaki import en, espeak
    except ImportError:
        return None
    british = language == "en-gb"
    try:
        fallback = espeak.EspeakFallback(british=british)
    except Exception as e:  # noqa: BLE001
        logger.warning("misaki's espeak fallback not available, unknown words will be skipped: %s", e)
        fallback = None
    logger.info("Using misaki for English G2P")
    return en.G2P(trf=False, british=british, fallback=fallback, unk="")


class KokoroPhonemizer(BasePhonemizer):
    """Phonemizer for Kokoro models, based on espeak and optionally misaki for English.

    Unlike most phonemizers, the language can be changed per call with the
    ``language`` argument of :meth:`phonemize`.

    Args:
        language: Default language, one of :meth:`supported_languages`.
        punctuations: Characters to be treated as punctuation.
        keep_puncs: Keep the punctuations after phonemization.
        use_misaki: Use misaki for English if it is installed.
    """

    def __init__(
        self,
        language: str = "en-us",
        punctuations: str = _PUNCTUATIONS,
        *,
        keep_puncs: bool = True,
        use_misaki: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(language, punctuations=punctuations, keep_puncs=keep_puncs)
        self.use_misaki = use_misaki
        self._espeak: dict[str, ESpeak] = {}

    @staticmethod
    def name() -> str:
        return "kokoro_phonemizer"

    @classmethod
    def is_available(cls) -> bool:
        return ESpeak.is_available()

    def version(self) -> str:
        return "1.0"

    @staticmethod
    def supported_languages() -> dict[str, str]:
        return dict(_LANGUAGES)

    def phonemize(self, text: str, separator: str = "", language: str | None = None) -> str:
        """Convert the text to Kokoro phonemes.

        Args:
            text: Text to be phonemized.
            separator: Not supported, phonemes are not separated.
            language: Language of the text. Defaults to the language set at init.
        """
        language = self.language if language is None else language
        if not self.is_supported_language(language):
            msg = f"Language `{language}` is not supported by {self.name()}, choose one of: {list(_LANGUAGES)}"
            raise ValueError(msg)
        if language.startswith("en") and self.use_misaki and (g2p := _load_misaki_g2p(language)) is not None:
            phonemes, _ = g2p(text)
            return phonemes.strip()
        preprocessed, punctuations = self._phonemize_preprocess(text)
        phonemized = [self._phonemize_espeak(t, language) for t in preprocessed]
        phonemes = self._phonemize_postprocess(phonemized, punctuations)
        # Map punctuation not in the Kokoro vocabulary
        return phonemes.replace("«", "“").replace("»", "”").replace("¡", "").replace("¿", "").strip()

    def _phonemize(self, text: str, separator: str = "") -> str:
        return self._phonemize_espeak(text, self.language)

    def _phonemize_espeak(self, text: str, language: str) -> str:
        if language not in self._espeak:
            self._espeak[language] = ESpeak(_LANGUAGES[language], keep_puncs=False)
        phonemes = self._espeak[language].phonemize_espeak(text, separator="", tie="^")
        if language.startswith("en"):
            return _map_english(phonemes, british=language == "en-gb")
        return _map_other(phonemes)
