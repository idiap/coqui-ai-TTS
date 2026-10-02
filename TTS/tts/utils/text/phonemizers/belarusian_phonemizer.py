from typing import Any

from TTS.tts.utils.text.belarusian.phonemizer import belarusian_text_to_phonemes
from TTS.tts.utils.text.phonemizers.base import BasePhonemizer

_DEF_BE_PUNCS = ",!."  # TODO


class BEL_Phonemizer(BasePhonemizer):
    """🐸TTS be phonemizer using functions in `TTS.tts.utils.text.belarusian.phonemizer`

    Args:
        punctuations (str):
            Set of characters to be treated as punctuation. Defaults to `_DEF_BE_PUNCS`.

        keep_puncs (bool):
            If True, keep the punctuations after phonemization. Defaults to False.
    """

    language = "be"

    def __init__(self, punctuations: str = _DEF_BE_PUNCS, *, keep_puncs: bool = True, **kwargs: Any) -> None:
        super().__init__(self.language, punctuations=punctuations, keep_puncs=keep_puncs)

    @staticmethod
    def name() -> str:
        return "be_phonemizer"

    @staticmethod
    def phonemize_be(text: str) -> str:
        return belarusian_text_to_phonemes(text)

    def _phonemize(self, text: str, separator: str) -> str:
        return self.phonemize_be(text)

    @staticmethod
    def supported_languages() -> dict[str, str]:
        return {"be": "Belarusian"}

    def version(self) -> str:
        return "0.0.1"

    @classmethod
    def is_available(cls) -> bool:
        return True


if __name__ == "__main__":
    txt = "тэст"
    e = BEL_Phonemizer()
    print(e.supported_languages())
    print(e.version())
    print(e.language)
    print(e.name())
    print(e.is_available())
    print("`" + e.phonemize(txt) + "`")
