import json
import shutil

from trainer.io import get_last_checkpoint

from tests import run_main
from TTS.bin.synthesize import main as synthesize
from TTS.bin.train_tts import main as train_tts
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig


def test_train(tmp_path):
    config_path = tmp_path / "test_model_config.json"
    output_path = tmp_path / "train_outputs"

    dataset_config_en = BaseDatasetConfig(
        formatter="ljspeech_test",
        meta_file_train="metadata.csv",
        meta_file_val="metadata.csv",
        path="tests/data/ljspeech",
        language="en",
    )

    dataset_config_pt = BaseDatasetConfig(
        formatter="ljspeech_test",
        meta_file_train="metadata.csv",
        meta_file_val="metadata.csv",
        path="tests/data/ljspeech",
        language="pt-br",
    )

    config = VitsConfig(
        batch_size=2,
        eval_batch_size=2,
        num_loader_workers=0,
        num_eval_loader_workers=0,
        text_cleaner="multilingual_cleaners",
        use_phonemes=False,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1,
        print_step=1,
        print_eval=True,
        test_sentences=[
            ["Be a voice, not an echo.", "ljspeech-0", None, "en"],
            ["Be a voice, not an echo.", "ljspeech-1", None, "pt-br"],
        ],
        datasets=[dataset_config_en, dataset_config_en, dataset_config_en, dataset_config_pt],
    )
    # set audio config
    config.audio.do_trim_silence = True
    config.audio.trim_db = 60

    # active multilingual mode
    config.model_args.use_language_embedding = True
    config.use_language_embedding = True

    # deactivate multispeaker mode
    config.model_args.use_speaker_embedding = False
    config.use_speaker_embedding = False

    # active multispeaker d-vec mode
    config.model_args.use_d_vector_file = True
    config.use_d_vector_file = True
    config.model_args.d_vector_file = ["tests/data/ljspeech/speakers.json"]
    config.d_vector_file = ["tests/data/ljspeech/speakers.json"]
    config.model_args.d_vector_dim = 256
    config.d_vector_dim = 256

    # duration predictor
    config.model_args.use_sdp = True
    config.use_sdp = True

    # activate language and speaker samplers
    config.use_language_weighted_sampler = True
    config.language_weighted_sampler_alpha = 10
    config.use_speaker_weighted_sampler = True
    config.speaker_weighted_sampler_alpha = 5

    config.save_json(config_path)

    # train the model for one epoch
    command_train = [
        "--config_path",
        str(config_path),
        "--coqpit.output_path",
        str(output_path),
        "--coqpit.test_delay_epochs",
        "0",
    ]
    run_main(train_tts, command_train)

    # Find latest folder
    continue_path = max(output_path.iterdir(), key=lambda p: p.stat().st_mtime)

    # Inference using TTS API
    continue_config_path = continue_path / "config.json"
    continue_restore_path, _ = get_last_checkpoint(continue_path)
    out_wav_path = tmp_path / "output.wav"
    speaker_id = "ljspeech-1"
    language_id = "en"
    continue_speakers_path = config.d_vector_file
    continue_languages_path = continue_path / "language_ids.json"

    # Check integrity of the config
    with continue_config_path.open() as f:
        config_loaded = json.load(f)
    assert config_loaded["characters"] is not None
    assert config_loaded["output_path"] in str(continue_path)
    assert config_loaded["test_delay_epochs"] == 0

    # Load the model and run inference
    inference_command = [
        "--text",
        "This is an example for the tests.",
        "--speaker_idx",
        speaker_id,
        "--language_idx",
        language_id,
        "--speakers_file_path",
        str(continue_speakers_path),
        "--language_ids_file_path",
        str(continue_languages_path),
        "--config_path",
        str(continue_config_path),
        "--model_path",
        str(continue_restore_path),
        "--out_path",
        str(out_wav_path),
    ]
    run_main(synthesize, inference_command)

    # restore the model and continue training for one more epoch
    run_main(train_tts, ["--continue_path", str(continue_path)])
    shutil.rmtree(tmp_path)
