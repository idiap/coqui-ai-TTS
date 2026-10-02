from typing import Any

from TTS.tts.utils.text.bangla.phonemizer import bangla_text_to_phonemes
from TTS.tts.utils.text.phonemizers.base import BasePhonemizer

_DEF_ZH_PUNCS = "、.,[]()?!〽~『』「」【】"


class BN_Phonemizer(BasePhonemizer):
    """🐸TTS bn phonemizer using functions in `TTS.tts.utils.text.bangla.phonemizer`

    Args:
        punctuations (str):
            Set of characters to be treated as punctuation. Defaults to `_DEF_ZH_PUNCS`.

        keep_puncs (bool):
            If True, keep the punctuations after phonemization. Defaults to False.

    Example ::

        "这是，样本中文。" -> `d|ʒ|ø|4| |ʂ|ʏ|4| |，| |i|ɑ|ŋ|4|b|œ|n|3| |d|ʒ|o|ŋ|1|w|œ|n|2| |。`

    TODO: someone with Bangla knowledge should check this implementation
    """

    language = "bn"

    def __init__(self, punctuations: str = _DEF_ZH_PUNCS, *, keep_puncs: bool = False, **kwargs: Any) -> None:
        super().__init__(self.language, punctuations=punctuations, keep_puncs=keep_puncs)

    @staticmethod
    def name() -> str:
        return "bn_phonemizer"

    @staticmethod
    def phonemize_bn(text: str) -> str:
        return bangla_text_to_phonemes(text)

    def _phonemize(self, text: str, separator: str) -> str:
        return self.phonemize_bn(text)

    @staticmethod
    def supported_languages() -> dict:
        return {"bn": "Bangla"}

    def version(self) -> str:
        return "0.0.1"

    def is_available(self) -> bool:
        return True


if __name__ == "__main__":
    txt = "রাসূলুল্লাহ সাল্লাল্লাহু আলাইহি ওয়া সাল্লাম শিক্ষা দিয়েছেন যে, কেউ যদি কোন খারাপ কিছুর সম্মুখীন হয়, তখনও যেন বলে."
    e = BN_Phonemizer()
    print(e.supported_languages())
    print(e.version())
    print(e.language)
    print(e.name())
    print(e.is_available())
    print("`" + e.phonemize(txt) + "`")
