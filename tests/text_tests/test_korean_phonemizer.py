import sys

import pytest

from TTS.tts.utils.text.korean.phonemizer import korean_text_to_phonemes

# TODO: enable when possible
pytestmark = pytest.mark.skipif(sys.version_info >= (3, 14), reason="Requires Python 3.13 or lower")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "포상은 열심히 한 아이에게만 주어지기 때문에 포상인 것입니다.",
            "포상으 녈심히 하 나이에게만 주어지기 때무네 포상인 거심니다.",
        ),
        ("오늘은 8월 31일 입니다.", "오느른 파뤌 삼시비리 림니다."),
        ("친구 100명 만들기가 목표입니다.", "친구 뱅명 만들기가 목표임니다."),
        ("A부터 Z까지 입니다.", "에이부터 제트까지 임니다."),
        ("이게 제 마음이에요.", "이게 제 마으미에요."),
    ],
)
def test_korean_text_to_phonemes(text, expected):
    assert korean_text_to_phonemes(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("이제야 이쪽을 보는구나.", "IJeYa IJjoGeul BoNeunGuNa."),
        ("크고 맛있는 cake를 부탁해요.", "KeuGo MaSinNeun KeIKeuLeul BuTaKaeYo."),
        ("전부 거짓말이야.", "JeonBu GeoJinMaLiYa."),
        ("좋은 노래를 찾았어요.", "JoEun NoLaeLeul ChaJaSseoYo."),
    ],
)
def test_korean_text_to_phonemes_english(text, expected):
    assert korean_text_to_phonemes(text, character="english") == expected
