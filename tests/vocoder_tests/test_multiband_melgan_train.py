from tests import get_device_id, run_cli
from TTS.vocoder.configs import MultibandMelganConfig


def test_train(tmp_path):
    config_path = tmp_path / "test_vocoder_config.json"
    output_path = tmp_path / "train_outputs"

    config = MultibandMelganConfig(
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
        steps_to_start_discriminator=1,
        data_path="tests/data/ljspeech",
        discriminator_model_params={"base_channels": 16, "max_channels": 64, "downsample_factors": [4, 4, 4]},
        output_path=output_path,
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
