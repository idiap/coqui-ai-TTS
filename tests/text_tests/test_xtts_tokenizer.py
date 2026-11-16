import pytest

from TTS.tts.layers.xtts.tokenizer import multilingual_cleaners


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("It is 11:08.", "it is eleven oh eight."),
        ("Meet at 14:30.", "meet at two thirty p m."),
        ("It is 12:00.", "it is twelve o'clock."),
        ("Alarm at 8:00 AM.", "alarm at eight a m."),
        ("At 7:05pm.", "at seven oh five p m."),
        ("At 9:30 a.m. tomorrow", "at nine thirty a m tomorrow"),
        ("At 9:30 a.m., then", "at nine thirty a m, then"),
        ("It starts at 9:30 a.m.", "it starts at nine thirty a m."),
        ("It starts at 9:30 a.m", "it starts at nine thirty a m"),
        ("At 8:00am. Then", "at eight a m. then"),
        ("It is 00:00.", "it is twelve a m."),
        ("It is 0:05.", "it is twelve oh five a m."),
        ("It is 13:00.", "it is one p m."),
        ("It is 23:45.", "it is eleven forty-five p m."),
        ("It is 24:00.", "it is twelve a m."),
        ("It is 24:30.", "it is twenty-four:thirty."),
        ("It is 13:00 pm.", "it is one p m."),
        ("Low of -5 degrees.", "low of minus five degrees."),
        ("From -2.5 to 3", "from minus two point five to three"),
        ("covid-19", "covid-nineteen"),
    ],
)
def test_english_times_and_negative_numbers(text: str, expected: str) -> None:
    assert multilingual_cleaners(text, "en") == expected
