import shutil

from tests import run_main
from TTS.bin.train_encoder import main
from TTS.config.shared_configs import BaseAudioConfig
from TTS.encoder.configs.speaker_encoder_config import SpeakerEncoderConfig


def test_train(tmp_path):
    config_path = tmp_path / "test_speaker_encoder_config.json"
    output_path = tmp_path / "train_outputs"

    def run_test_train():
        command = [
            "--config_path",
            str(config_path),
            "--coqpit.output_path",
            str(output_path),
            "--coqpit.datasets.0.formatter",
            "ljspeech_test",
            "--coqpit.datasets.0.meta_file_train",
            "metadata.csv",
            "--coqpit.datasets.0.meta_file_val",
            "metadata.csv",
            "--coqpit.datasets.0.path",
            "tests/data/ljspeech",
        ]
        run_main(main, command)

    config = SpeakerEncoderConfig(
        batch_size=4,
        num_classes_in_batch=4,
        num_utter_per_class=2,
        eval_num_classes_in_batch=4,
        eval_num_utter_per_class=2,
        num_loader_workers=1,
        epochs=1,
        print_step=1,
        save_step=2,
        print_eval=True,
        run_eval=True,
        audio=BaseAudioConfig(num_mels=80),
    )
    config.audio.do_trim_silence = True
    config.audio.trim_db = 60
    config.loss = "ge2e"
    config.save_json(config_path)

    print(config)
    # train the model for one epoch
    run_test_train()

    # Find latest folder
    continue_path = max(output_path.iterdir(), key=lambda p: p.stat().st_mtime)

    # restore the model and continue training for one more epoch
    run_main(main, ["--continue_path", str(continue_path)])
    shutil.rmtree(continue_path)

    # test resnet speaker encoder
    config.model_params["model_name"] = "resnet"
    config.save_json(config_path)

    # train the model for one epoch
    run_test_train()

    # Find latest folder
    continue_path = max(output_path.iterdir(), key=lambda p: p.stat().st_mtime)

    # restore the model and continue training for one more epoch
    run_main(main, ["--continue_path", str(continue_path)])
    shutil.rmtree(continue_path)

    # test model with ge2e loss function
    # config.loss = "ge2e"
    # config.save_json(config_path)
    # run_test_train()

    # test model with angleproto loss function
    # config.loss = "angleproto"
    # config.save_json(config_path)
    # run_test_train()

    # test model with softmaxproto loss function
    config.loss = "softmaxproto"
    config.save_json(config_path)
    run_test_train()
