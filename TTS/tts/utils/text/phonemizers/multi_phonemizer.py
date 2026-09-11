import logging

from TTS.tts.utils.text.phonemizers import get_default_phonemizer, get_phonemizer_by_name

logger = logging.getLogger(__name__)


class MultiPhonemizer:
    """🐸TTS multi-phonemizer that operates phonemizers for multiple languages.

    Args:
        custom_lang_to_phonemizer (Dict):
            Custom phonemizer mapping if you want to change the defaults. In the format of
            `{"lang_code", "phonemizer_name"}`. When the phonemizer name is
            None, a default phonemizer is used. Defaults to `{}`.

    TODO: find a way to pass custom kwargs to the phonemizers
    """

    def __init__(self, lang_to_phonemizer_name: dict[str, str | None] | None = None) -> None:
        self.lang_to_phonemizer = {}
        if lang_to_phonemizer_name is None:
            lang_to_phonemizer_name = {}
        for lang, name in lang_to_phonemizer_name.items():
            if not name:
                name = get_default_phonemizer(lang)
            self.lang_to_phonemizer[lang] = get_phonemizer_by_name(name, language=lang)

    @staticmethod
    def name() -> str:
        return "multi-phonemizer"

    def phonemize(self, text: str, separator: str = "|", language: str | None = None) -> str:
        if language is None:
            raise ValueError("Language must be set for multi-phonemizer to phonemize.")
        return self.lang_to_phonemizer[language].phonemize(text, separator)

    def supported_languages(self) -> list[str]:
        return list(self.lang_to_phonemizer.keys())

    def print_logs(self, level: int = 0) -> None:
        indent = "\t" * level
        logger.info("%s| phoneme language: %s", indent, self.supported_languages())
        logger.info("%s| phoneme backend: %s", indent, self.name())


# if __name__ == "__main__":
#     texts = {
#         "tr": "Merhaba, bu Türkçe bit örnek!",
#         "en-us": "Hello, this is English example!",
#         "de": "Hallo, das ist ein Deutches Beipiel!",
#         "zh-cn": "这是中国的例子",
#     }
#     phonemes = {}
#     ph = MultiPhonemizer({"tr": "espeak", "en-us": "", "de": "gruut", "zh-cn": ""})
#     for lang, text in texts.items():
#         phoneme = ph.phonemize(text, lang)
#         phonemes[lang] = phoneme
#     print(phonemes)
