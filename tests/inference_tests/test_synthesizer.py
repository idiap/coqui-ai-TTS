from pathlib import Path

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


def test_split_into_sentences():
    """Check synthesizer sentences split as expected"""

    class _MockSynthesizer:
        """Mock synthesizer with segmenter for testing sentence splitting."""

        def __init__(self, seg):
            self.seg = seg

    mock = _MockSynthesizer(Synthesizer._get_segmenter("en"))
    sis = Synthesizer.split_into_sentences
    assert sis(mock, "Hello. Two sentences") == ["Hello.", "Two sentences"]
    assert sis(mock, "He went to meet the adviser from Scott, Waltman & Co. next morning.") == [
        "He went to meet the adviser from Scott, Waltman & Co. next morning."
    ]
    assert sis(mock, "Let's run it past Sarah and co. They'll want to see this.") == [
        "Let's run it past Sarah and co.",
        "They'll want to see this.",
    ]
    assert sis(mock, "Where is Bobby Jr.'s rabbit?") == ["Where is Bobby Jr.'s rabbit?"]
    assert sis(mock, "Please inform the U.K. authorities right away.") == [
        "Please inform the U.K. authorities right away."
    ]
    assert sis(mock, "Were David and co. at the event?") == ["Were David and co. at the event?"]
    assert sis(mock, "paging dr. green, please come to theatre four immediately.") == [
        "paging dr. green, please come to theatre four immediately."
    ]
    assert sis(mock, "The email format is Firstname.Lastname@example.com. I think you reversed them.") == [
        "The email format is Firstname.Lastname@example.com.",
        "I think you reversed them.",
    ]
    assert sis(
        mock,
        "The demo site is: https://top100.example.com/subsection/latestnews.html. Please send us your feedback.",
    ) == [
        "The demo site is: https://top100.example.com/subsection/latestnews.html.",
        "Please send us your feedback.",
    ]
    assert sis(mock, "Scowling at him, 'You are not done yet!' she yelled.") == [
        "Scowling at him, 'You are not done yet!' she yelled."
    ]  # with the  final lowercase "she" we see it's all one sentence
    assert sis(mock, "Hey!! So good to see you.") == ["Hey!!", "So good to see you."]
    assert sis(mock, "He went to Yahoo! but I don't know the division.") == [
        "He went to Yahoo! but I don't know the division."
    ]
    assert sis(mock, "If you can't remember a quote, \"at least make up a memorable one that's plausible...\"") == [
        "If you can't remember a quote, \"at least make up a memorable one that's plausible...\""
    ]
    assert sis(mock, "The address is not google.com.") == ["The address is not google.com."]
    assert sis(mock, "1.) The first item 2.) The second item") == ["1.) The first item", "2.) The second item"]
    assert sis(mock, "1) The first item 2) The second item") == ["1) The first item", "2) The second item"]
    assert sis(mock, "a. The first item b. The second item c. The third list item") == [
        "a. The first item",
        "b. The second item",
        "c. The third list item",
    ]


def test_synthesizer_timestamps(tmp_path):
    """Check if timestamps are generated."""
    _create_random_model(tmp_path)
    tts_checkpoint = str(tmp_path / f"checkpoint_{CURRENT_STEP}.pth")
    synthesizer = Synthesizer(tts_checkpoint=tts_checkpoint, tts_config_path=TTS_CONFIG)

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
