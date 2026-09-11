from pathlib import Path

import pysbd
import pytest
from trainer.io import save_checkpoint

from tests import get_tests_input_path
from TTS.config import load_config
from TTS.tts.models import setup_model
from TTS.utils.synthesizer import Synthesizer

CURRENT_STEP = 10
TTS_CONFIG = str(Path(get_tests_input_path()) / "dummy_model_config.json")


def _create_random_model(output_path):
    config = load_config(TTS_CONFIG)
    model = setup_model(config)
    save_checkpoint(config, model, str(output_path), current_step=CURRENT_STEP, epoch=1)


@pytest.fixture(scope="module")
def mock_synthesizer():
    """Mock synthesizer with segmenters for testing sentence splitting."""

    class _MockLanguageManager:
        language_names = ["en", "el"]

    class _MockTTSModel:
        language_manager = _MockLanguageManager()

    class _MockSynthesizer:
        def __init__(self):
            self.segmenter = {
                "en": pysbd.Segmenter(language="en", clean=True),
                "el": pysbd.Segmenter(language="el", clean=True),
            }
            self.tts_model = _MockTTSModel()

    return _MockSynthesizer()


@pytest.mark.parametrize(
    ("text", "expected", "language"),
    [
        ("Hello. Two sentences", ["Hello.", "Two sentences"], "en"),
        (
            "He went to meet the adviser from Scott, Waltman & Co. next morning.",
            ["He went to meet the adviser from Scott, Waltman & Co. next morning."],
            "en",
        ),
        (
            "Let's run it past Sarah and co. They'll want to see this.",
            ["Let's run it past Sarah and co.", "They'll want to see this."],
            "en",
        ),
        ("Where is Bobby Jr.'s rabbit?", ["Where is Bobby Jr.'s rabbit?"], "en"),
        ("Please inform the U.K. authorities right away.", ["Please inform the U.K. authorities right away."], "en"),
        ("Were David and co. at the event?", ["Were David and co. at the event?"], "en"),
        (
            "paging dr. green, please come to theatre four immediately.",
            ["paging dr. green, please come to theatre four immediately."],
            "en",
        ),
        (
            "The email format is Firstname.Lastname@example.com. I think you reversed them.",
            ["The email format is Firstname.Lastname@example.com.", "I think you reversed them."],
            "en",
        ),
        (
            "The demo site is: https://top100.example.com/subsection/latestnews.html. Please send us your feedback.",
            [
                "The demo site is: https://top100.example.com/subsection/latestnews.html.",
                "Please send us your feedback.",
            ],
            "en",
        ),
        # With the final lowercase "she" we see it's all one sentence
        (
            "Scowling at him, 'You are not done yet!' she yelled.",
            ["Scowling at him, 'You are not done yet!' she yelled."],
            "en",
        ),
        ("Hey!! So good to see you.", ["Hey!!", "So good to see you."], "en"),
        (
            "He went to Yahoo! but I don't know the division.",
            ["He went to Yahoo! but I don't know the division."],
            "en",
        ),
        (
            "If you can't remember a quote, \"at least make up a memorable one that's plausible...\"",
            ["If you can't remember a quote, \"at least make up a memorable one that's plausible...\""],
            "en",
        ),
        ("The address is not google.com.", ["The address is not google.com."], "en"),
        ("1.) The first item 2.) The second item", ["1.) The first item", "2.) The second item"], "en"),
        ("1) The first item 2) The second item", ["1) The first item", "2) The second item"], "en"),
        (
            "a. The first item b. The second item c. The third list item",
            ["a. The first item", "b. The second item", "c. The third list item"],
            "en",
        ),
        # Greek
        (
            "Με συγχωρείτε· πού είναι οι τουαλέτες; Τις Κυριακές δε δούλευε κανένας. το κόστος του σπιτιού ήταν £260.950,00.",
            [
                "Με συγχωρείτε· πού είναι οι τουαλέτες;",
                "Τις Κυριακές δε δούλευε κανένας.",
                "το κόστος του σπιτιού ήταν £260.950,00.",
            ],
            "el",
        ),
    ],
)
def test_split_into_sentences(mock_synthesizer, text, expected, language):
    """Check synthesizer sentences split as expected."""
    assert Synthesizer.split_into_sentences(mock_synthesizer, text, language) == expected


def test_synthesizer_timestamps(tmp_path):
    """Check if timestamps are generated."""
    _create_random_model(tmp_path)
    tts_checkpoint = str(tmp_path / f"checkpoint_{CURRENT_STEP}.pth")
    synthesizer = Synthesizer(tts_checkpoint=tts_checkpoint, tts_config_path=TTS_CONFIG)
    print(synthesizer.tts_model.language_manager.language_names)
    wav = synthesizer.tts("Hello world.", return_dict=False)
    assert isinstance(wav, list)

    output = synthesizer.tts("Hello world. This is a test.", return_dict=True)
    assert isinstance(output, dict)
    assert "segments" in output
    assert "wav" in output

    segments = output["segments"]
    assert len(segments) == 2
    assert segments[0]["end"] > segments[0]["start"]

    output_no_split = synthesizer.tts("Hello world. This is a test.", split_sentences=False, return_dict=True)
    assert len(output_no_split["segments"]) == 1
