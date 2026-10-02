from typing import Any

from TTS.tts.utils.text.chinese_mandarin.phonemizer import chinese_text_to_phonemes
from TTS.tts.utils.text.phonemizers.base import BasePhonemizer

_DEF_ZH_PUNCS = "、.,[]()?!〽~『』「」【】"


class ZH_CN_Phonemizer(BasePhonemizer):
    """🐸TTS Zh-Cn phonemizer using functions in `TTS.tts.utils.text.chinese_mandarin.phonemizer`

    Args:
        punctuations (str):
            Set of characters to be treated as punctuation. Defaults to `_DEF_ZH_PUNCS`.

        keep_puncs (bool):
            If True, keep the punctuations after phonemization. Defaults to False.

    Example ::

        "这是，样本中文。" -> `d|ʒ|ø|4| |ʂ|ʏ|4| |，| |i|ɑ|ŋ|4|b|œ|n|3| |d|ʒ|o|ŋ|1|w|œ|n|2| |。`

    TODO: someone with Mandarin knowledge should check this implementation
    """

    language = "zh-cn"

    def __init__(self, punctuations: str = _DEF_ZH_PUNCS, *, keep_puncs: bool = False, **kwargs: Any) -> None:
        super().__init__(self.language, punctuations=punctuations, keep_puncs=keep_puncs)

    @staticmethod
    def name() -> str:
        return "zh_cn_phonemizer"

    @staticmethod
    def phonemize_zh_cn(text: str, separator: str = "|") -> str:
        return chinese_text_to_phonemes(text, separator)

    def _phonemize(self, text: str, separator: str) -> str:
        return self.phonemize_zh_cn(text, separator)

    @staticmethod
    def supported_languages() -> dict[str, str]:
        return {"zh-cn": "Chinese (China)"}

    def version(self) -> str:
        return "0.0.1"

    @classmethod
    def is_available(cls) -> bool:
        return True


# if __name__ == "__main__":
#     text = "这是，样本中文。"
#     e = ZH_CN_Phonemizer()
#     print(e.supported_languages())
#     print(e.version())
#     print(e.language)
#     print(e.name())
#     print(e.is_available())
#     print("`" + e.phonemize(text) + "`")
