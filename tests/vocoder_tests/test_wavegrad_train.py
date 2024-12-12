import os

import pytest

from tests import get_device_id, run_cli
from TTS.vocoder.configs import WavegradConfig

GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


# TODO: Reactivate after improving CI run times
@pytest.mark.skipif(GITHUB_ACTIONS, reason="Takes ~2h on CI (15min/step vs 8sec/step locally)")
def test_train(tmp_path):
    config_path = tmp_path / "test_vocoder_config.json"
    output_path = tmp_path / "train_outputs"

    config = WavegradConfig(
        batch_size=8,
        eval_batch_size=8,
        num_loader_workers=0,
        num_eval_loader_workers=0,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1,
        seq_len=8192,
        eval_split_size=1,
        print_step=1,
        print_eval=True,
        data_path="tests/data/ljspeech",
        output_path=output_path,
        test_noise_schedule={"min_val": 1e-6, "max_val": 1e-2, "num_steps": 2},
    )
    config.audio.do_trim_silence = True
    config.audio.trim_db = 60
    config.save_json(config_path)

    # train the model for one epoch
    command_train = (
        f"CUDA_VISIBLE_DEVICES='{get_device_id()}' python TTS/bin/train_vocoder.py --config_path {config_path} "
    )
    run_cli(command_train)

    # Find latest folder
    continue_path = max(output_path.iterdir(), key=lambda p: p.stat().st_mtime)

    # restore the model and continue training for one more epoch
    command_train = (
        f"CUDA_VISIBLE_DEVICES='{get_device_id()}' python TTS/bin/train_vocoder.py --continue_path {continue_path} "
    )
    run_cli(command_train)
