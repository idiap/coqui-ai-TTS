"""Initialize phonemizers based on installed packages."""

from typing import Any

from TTS.tts.utils.text.phonemizers.base import BasePhonemizer
from TTS.tts.utils.text.phonemizers.belarusian_phonemizer import BEL_Phonemizer
from TTS.tts.utils.text.phonemizers.espeak_wrapper import ESpeak
from TTS.utils.import_utils import PHONEMIZER_DEPS, is_phonemizer_available

# Dict mapping phonemizer name to its class
PHONEMIZERS: dict[str, type[BasePhonemizer]] = {}

# Dict setting default phonemizers for each language
DEF_LANG_TO_PHONEMIZER: dict[str, str] = {}


def _register_bangla() -> None:
    """Register the Bangla phonemizer if available."""
    if is_phonemizer_available("bn_phonemizer"):
        from TTS.tts.utils.text.phonemizers.bangla_phonemizer import BN_Phonemizer  # noqa: PLC0415

        DEF_LANG_TO_PHONEMIZER["bn"] = BN_Phonemizer.name()
        PHONEMIZERS[BN_Phonemizer.name()] = BN_Phonemizer


def _register_chinese() -> None:
    """Register the Chinese phonemizer if available."""
    if is_phonemizer_available("zh_cn_phonemizer"):
        from TTS.tts.utils.text.phonemizers.zh_cn_phonemizer import ZH_CN_Phonemizer  # noqa: PLC0415

        DEF_LANG_TO_PHONEMIZER["zh-cn"] = ZH_CN_Phonemizer.name()
        PHONEMIZERS[ZH_CN_Phonemizer.name()] = ZH_CN_Phonemizer


def _register_espeak() -> None:
    """Register ESpeak and its supported languages."""
    DEF_LANG_TO_PHONEMIZER.update({lang: ESpeak.name() for lang in ESpeak.supported_languages()})
    PHONEMIZERS[ESpeak.name()] = ESpeak


def _register_gruut() -> None:
    """Register Gruut and its supported languages if available."""
    if is_phonemizer_available("gruut"):
        from TTS.tts.utils.text.phonemizers.gruut_wrapper import Gruut  # noqa: PLC0415

        DEF_LANG_TO_PHONEMIZER.update({lang: Gruut.name() for lang in Gruut.supported_languages()})
        PHONEMIZERS[Gruut.name()] = Gruut


def _register_japanese() -> None:
    """Register the Japanese phonemizer if available."""
    if is_phonemizer_available("ja_jp_phonemizer"):
        from TTS.tts.utils.text.phonemizers.ja_jp_phonemizer import JA_JP_Phonemizer  # noqa: PLC0415

        DEF_LANG_TO_PHONEMIZER["ja-jp"] = JA_JP_Phonemizer.name()
        PHONEMIZERS[JA_JP_Phonemizer.name()] = JA_JP_Phonemizer


def _register_korean() -> None:
    """Register the Korean phonemizer if available."""
    if is_phonemizer_available("ko_kr_phonemizer"):
        from TTS.tts.utils.text.phonemizers.ko_kr_phonemizer import KO_KR_Phonemizer  # noqa: PLC0415

        DEF_LANG_TO_PHONEMIZER["ko-kr"] = KO_KR_Phonemizer.name()
        PHONEMIZERS[KO_KR_Phonemizer.name()] = KO_KR_Phonemizer


def _register_phonemizers() -> None:
    if len(DEF_LANG_TO_PHONEMIZER) == 0:
        _register_gruut()
        _register_espeak()  # override existing Gruut languages

        # Force default for some languages
        if "en-us" in DEF_LANG_TO_PHONEMIZER:
            DEF_LANG_TO_PHONEMIZER["en"] = DEF_LANG_TO_PHONEMIZER["en-us"]
        DEF_LANG_TO_PHONEMIZER["be"] = BEL_Phonemizer.name()
        PHONEMIZERS[BEL_Phonemizer.name()] = BEL_Phonemizer

        _register_bangla()
        _register_japanese()
        _register_korean()
        _register_chinese()


def get_default_phonemizer(language: str | None) -> str:
    """Return the name of the default phonemizer for the given language."""
    _register_phonemizers()

    if language not in DEF_LANG_TO_PHONEMIZER:
        msg = f"No phonemizer found for language {language}. You may need to install a third party library for it."
        raise ValueError(msg)
    return DEF_LANG_TO_PHONEMIZER[language]


def get_phonemizer_by_name(name: str, **kwargs: Any) -> BasePhonemizer:
    """Initialize a phonemizer by name.

    Args:
        name (str):
            Name of the phonemizer that should match `phonemizer.name()`.

        kwargs (dict):
            Extra keyword arguments that should be passed to the phonemizer.

    """
    _register_phonemizers()
    if name in PHONEMIZER_DEPS and not is_phonemizer_available(name):
        raise ImportError(PHONEMIZER_DEPS[name].import_error(name))
    if name not in PHONEMIZERS:
        msg = f"Phonemizer {name} not found"
        raise ValueError(msg)
    return PHONEMIZERS[name](**kwargs)
