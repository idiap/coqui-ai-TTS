from tests import run_cli


def test_synthesize(tmp_path):
    """Test synthesize.py with diffent arguments."""
    output_path = tmp_path / "output.wav"
    run_cli("tts --list_models")

    # single speaker model
    run_cli(f'tts --text "This is an example." --out_path "{output_path}"')
    run_cli(
        "tts --model_name tts_models/en/ljspeech/glow-tts " f'--text "This is an example." --out_path "{output_path}"'
    )
    run_cli(
        "tts --model_name tts_models/en/ljspeech/glow-tts  "
        "--vocoder_name vocoder_models/en/ljspeech/multiband-melgan "
        f'--text "This is an example." --out_path "{output_path}"'
    )
