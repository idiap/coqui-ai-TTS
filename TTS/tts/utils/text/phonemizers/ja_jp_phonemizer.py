from typing import Any

from TTS.tts.utils.text.japanese.phonemizer import japanese_text_to_phonemes
from TTS.tts.utils.text.phonemizers.base import BasePhonemizer

_DEF_JA_PUNCS = "、.,[]()?!〽~『』「」【】"

_TRANS_TABLE = {"、": ","}


def trans(text: str) -> str:
    for i, j in _TRANS_TABLE.items():
        text = text.replace(i, j)
    return text


class JA_JP_Phonemizer(BasePhonemizer):
    """🐸TTS Ja-Jp phonemizer using functions in `TTS.tts.utils.text.japanese.phonemizer`

    TODO: someone with JA knowledge should check this implementation

    Example:

        >>> from TTS.tts.utils.text.phonemizers import JA_JP_Phonemizer
        >>> phonemizer = JA_JP_Phonemizer()
        >>> phonemizer.phonemize("どちらに行きますか？", separator="|")
        'd|o|c|h|i|r|a|n|i|i|k|i|m|a|s|u|k|a|?'

    """

    language = "ja-jp"

    def __init__(self, punctuations: str = _DEF_JA_PUNCS, *, keep_puncs: bool = True, **kwargs: Any) -> None:
        super().__init__(self.language, punctuations=punctuations, keep_puncs=keep_puncs)

    @staticmethod
    def name() -> str:
        return "ja_jp_phonemizer"

    def _phonemize(self, text: str, separator: str = "|") -> str:
        ph = japanese_text_to_phonemes(text)
        if separator is not None or separator != "":
            return separator.join(ph)
        return ph

    def phonemize(self, text: str, separator: str = "|", language: str | None = None) -> str:
        """Custom phonemize for JP_JA

        Skip pre-post processing steps used by the other phonemizers.
        """
        return self._phonemize(text, separator)

    @staticmethod
    def supported_languages() -> dict[str, str]:
        return {"ja-jp": "Japanese (Japan)"}

    def version(self) -> str:
        return "0.0.1"

    @classmethod
    def is_available(cls) -> bool:
        return True


# if __name__ == "__main__":
#     text = "これは、電話をかけるための私の日本語の例のテキストです。"
#     e = JA_JP_Phonemizer()
#     print(e.supported_languages())
#     print(e.version())
#     print(e.language)
#     print(e.name())
#     print(e.is_available())
#     print("`" + e.phonemize(text) + "`")
