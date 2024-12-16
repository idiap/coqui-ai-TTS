import glob
import os

import pytest

from tests import run_main
from TTS.bin.train_vocoder import main
from TTS.vocoder.configs import (
    FullbandMelganConfig,
    HifiganConfig,
    MelganConfig,
    MultibandMelganConfig,
    ParallelWaveganConfig,
    WavegradConfig,
    WavernnConfig,
)
from TTS.vocoder.models.wavernn import WavernnArgs

GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

BASE_CONFIG = {
    "batch_size": 8,
    "eval_batch_size": 8,
    "num_loader_workers": 0,
    "num_eval_loader_workers": 0,
    "run_eval": True,
    "test_delay_epochs": -1,
    "epochs": 1,
    "seq_len": 8192,
    "eval_split_size": 1,
    "print_step": 1,
    "print_eval": True,
    "data_path": "tests/data/ljspeech",
}

DISCRIMINATOR_MODEL_PARAMS = {
    "base_channels": 16,
    "max_channels": 64,
    "downsample_factors": [4, 4, 4],
}


def create_config(config_class, **overrides):
    params = {**BASE_CONFIG, **overrides}
    return config_class(**params)


def run_train(tmp_path, config):
    config_path = str(tmp_path / "test_vocoder_config.json")
    output_path = tmp_path / "train_outputs"
    config.output_path = output_path
    config.audio.do_trim_silence = True
    config.audio.trim_db = 60
    config.save_json(config_path)

    # Train the model for one epoch
    run_main(main, ["--config_path", config_path])

    # Find the latest folder
    continue_path = str(max(glob.glob(os.path.join(output_path, "*/")), key=os.path.getmtime))

    # Restore the model and continue training for one more epoch
    run_main(main, ["--continue_path", continue_path])


def test_train_hifigan(tmp_path):
    config = create_config(HifiganConfig, seq_len=1024)
    run_train(tmp_path, config)


def test_train_melgan(tmp_path):
    config = create_config(
        MelganConfig,
        batch_size=4,
        eval_batch_size=4,
        seq_len=2048,
        discriminator_model_params=DISCRIMINATOR_MODEL_PARAMS,
    )
    run_train(tmp_path, config)


def test_train_multiband_melgan(tmp_path):
    config = create_config(
        MultibandMelganConfig, steps_to_start_discriminator=1, discriminator_model_params=DISCRIMINATOR_MODEL_PARAMS
    )
    run_train(tmp_path, config)


def test_train_fullband_melgan(tmp_path):
    config = create_config(FullbandMelganConfig, discriminator_model_params=DISCRIMINATOR_MODEL_PARAMS)
    run_train(tmp_path, config)


def test_train_parallel_wavegan(tmp_path):
    config = create_config(ParallelWaveganConfig, batch_size=4, eval_batch_size=4, seq_len=2048)
    run_train(tmp_path, config)


# TODO: Reactivate after improving CI run times
@pytest.mark.skipif(GITHUB_ACTIONS, reason="Takes ~2h on CI (15min/step vs 8sec/step locally)")
def test_train_wavegrad(tmp_path):
    config = create_config(WavegradConfig, test_noise_schedule={"min_val": 1e-6, "max_val": 1e-2, "num_steps": 2})
    run_train(tmp_path, config)


def test_train_wavernn(tmp_path):
    config = create_config(
        WavernnConfig,
        model_args=WavernnArgs(),
        seq_len=256,  # For shorter test time
    )
    run_train(tmp_path, config)
