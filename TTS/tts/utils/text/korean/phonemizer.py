import ko_speech_tools as kst

from TTS.tts.utils.text.korean.korean import normalize

g2p = None


def korean_text_to_phonemes(text, character: str = "hangeul") -> str:
    """

    The input and output values look the same, but they are different in Unicode.

    example :

        input = '하늘' (Unicode : \ud558\ub298), (하 + 늘)
        output = '하늘' (Unicode :\u1112\u1161\u1102\u1173\u11af), (ᄒ + ᅡ + ᄂ + ᅳ + ᆯ)

    """
    global g2p  # pylint: disable=global-statement
    if g2p is None:
        try:
            g2p = kst.G2p()
        except ImportError as e:
            raise ImportError("Korean requires: mecab-ko (available in the `ko` extra)") from e

    text = normalize(text)
    text = g2p(text)

    if character == "english":
        # Not used in practice, just allows easier debugging
        from anyascii import anyascii

        return anyascii(text)
    text = list(kst.jamo.hangul_to_jamo(text))  # '하늘' --> ['ᄒ', 'ᅡ', 'ᄂ', 'ᅳ', 'ᆯ']
    return "".join(text)
