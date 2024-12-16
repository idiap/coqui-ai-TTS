from tests import run_main
from TTS.bin.synthesize import main


def test_synthesize(tmp_path):
    """Test synthesize.py with diffent arguments."""
    output_path = str(tmp_path / "output.wav")

    run_main(main, ["--list_models"])

    # single speaker model
    args = ["--text", "This is an example.", "--out_path", output_path]
    run_main(main, args)

    args = [*args, "--model_name", "tts_models/en/ljspeech/glow-tts"]
    run_main(main, args)
    run_main(main, [*args, "--vocoder_name", "vocoder_models/en/ljspeech/multiband-melgan"])
