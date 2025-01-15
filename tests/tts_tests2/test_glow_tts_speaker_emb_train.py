import json
import shutil

from trainer.io import get_last_checkpoint

from tests import get_device_id, run_cli
from TTS.tts.configs.glow_tts_config import GlowTTSConfig


def test_train(tmp_path):
    config_path = tmp_path / "test_model_config.json"
    output_path = tmp_path / "train_outputs"

    config = GlowTTSConfig(
        batch_size=2,
        eval_batch_size=8,
        num_loader_workers=0,
        num_eval_loader_workers=0,
        text_cleaner="english_cleaners",
        use_phonemes=True,
        phoneme_language="en-us",
        phoneme_cache_path=tmp_path / "phoneme_cache",
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1,
        print_step=1,
        print_eval=True,
        test_sentences=[
            "Be a voice, not an echo.",
        ],
        data_dep_init_steps=1.0,
        use_speaker_embedding=True,
    )
    config.audio.do_trim_silence = True
    config.audio.trim_db = 60
    config.save_json(config_path)

    # train the model for one epoch
    command_train = (
        f"CUDA_VISIBLE_DEVICES='{get_device_id()}' python TTS/bin/train_tts.py --config_path {config_path} "
        f"--coqpit.output_path {output_path} "
        "--coqpit.datasets.0.formatter ljspeech_test "
        "--coqpit.datasets.0.meta_file_train metadata.csv "
        "--coqpit.datasets.0.meta_file_val metadata.csv "
        "--coqpit.datasets.0.path tests/data/ljspeech "
        "--coqpit.datasets.0.meta_file_attn_mask tests/data/ljspeech/metadata_attn_mask.txt "
        "--coqpit.test_delay_epochs 0"
    )
    run_cli(command_train)

    # Find latest folder
    continue_path = max(output_path.iterdir(), key=lambda p: p.stat().st_mtime)

    # Inference using TTS API
    continue_config_path = continue_path / "config.json"
    continue_restore_path, _ = get_last_checkpoint(continue_path)
    out_wav_path = tmp_path / "output.wav"
    speaker_id = "ljspeech-1"
    continue_speakers_path = continue_path / "speakers.json"

    # Check integrity of the config
    with continue_config_path.open() as f:
        config_loaded = json.load(f)
    assert config_loaded["characters"] is not None
    assert config_loaded["output_path"] in str(continue_path)
    assert config_loaded["test_delay_epochs"] == 0

    # Load the model and run inference
    inference_command = f"CUDA_VISIBLE_DEVICES='{get_device_id()}' tts --text 'This is an example.' --speaker_idx {speaker_id} --speakers_file_path {continue_speakers_path} --config_path {continue_config_path} --model_path {continue_restore_path} --out_path {out_wav_path}"
    run_cli(inference_command)

    # restore the model and continue training for one more epoch
    command_train = (
        f"CUDA_VISIBLE_DEVICES='{get_device_id()}' python TTS/bin/train_tts.py --continue_path {continue_path} "
    )
    run_cli(command_train)
    shutil.rmtree(continue_path)
