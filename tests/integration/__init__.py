import json
import shutil
from pathlib import Path
from typing import Any, TypeVar, Union

import torch
from trainer.io import get_last_checkpoint

from tests import run_main
from TTS.bin.synthesize import main as synthesize
from TTS.bin.train_tts import main as train_tts
from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.vc.configs.shared_configs import BaseVCConfig

TEST_TTS_CONFIG = {
    "batch_size": 8,
    "eval_batch_size": 8,
    "num_loader_workers": 0,
    "num_eval_loader_workers": 0,
    "text_cleaner": "english_cleaners",
    "use_phonemes": True,
    "phoneme_language": "en-us",
    "run_eval": True,
    "test_delay_epochs": -1,
    "epochs": 1,
    "print_step": 1,
    "print_eval": True,
    "test_sentences": ["Be a voice, not an echo."],
}

TEST_VC_CONFIG = {
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

Config = TypeVar("Config", BaseTTSConfig, BaseVCConfig)


def create_config(config_class: type[Config], **overrides: Any) -> Config:
    base_config = TEST_TTS_CONFIG if issubclass(config_class, BaseTTSConfig) else TEST_VC_CONFIG
    params = {**base_config, **overrides}
    return config_class(**params)


def run_tts_train(tmp_path: Path, config: BaseTTSConfig):
    config_path = tmp_path / "test_model_config.json"
    output_path = tmp_path / "train_outputs"

    # For NeuralHMM and Overflow
    parameter_path = tmp_path / "lj_parameters.pt"
    torch.save({"mean": -5.5138, "std": 2.0636, "init_transition_prob": 0.3212}, parameter_path)
    config.mel_statistics_parameter_path = parameter_path

    config.audio.do_trim_silence = True
    config.audio.trim_db = 60
    config.save_json(config_path)

    # train the model for one epoch
    is_multi_speaker = config.use_speaker_embedding or config.use_d_vector_file
    formatter = "ljspeech_test" if is_multi_speaker else "ljspeech"
    command_train = [
        "--config_path",
        str(config_path),
        "--coqpit.output_path",
        str(output_path),
        "--coqpit.phoneme_cache_path",
        str(output_path / "phoneme_cache"),
        "--coqpit.datasets.0.formatter",
        formatter,
        "--coqpit.datasets.0.meta_file_train",
        "metadata.csv",
        "--coqpit.datasets.0.meta_file_val",
        "metadata.csv",
        "--coqpit.datasets.0.path",
        "tests/data/ljspeech",
        "--coqpit.test_delay_epochs",
        "0",
        "--coqpit.datasets.0.meta_file_attn_mask",
        "tests/data/ljspeech/metadata_attn_mask.txt",
    ]
    run_main(train_tts, command_train)

    # Find latest folder
    continue_path = max(output_path.iterdir(), key=lambda p: p.stat().st_mtime)

    # Inference using TTS API
    continue_config_path = continue_path / "config.json"
    continue_restore_path, _ = get_last_checkpoint(continue_path)
    out_wav_path = tmp_path / "output.wav"

    # Check integrity of the config
    with continue_config_path.open() as f:
        config_loaded = json.load(f)
    assert config_loaded["characters"] is not None
    assert config_loaded["output_path"] in str(continue_path)
    assert config_loaded["test_delay_epochs"] == 0

    inference_command = [
        "--text",
        "This is an example for the tests.",
        "--config_path",
        str(continue_config_path),
        "--model_path",
        str(continue_restore_path),
        "--out_path",
        str(out_wav_path),
    ]
    if config.use_speaker_embedding:
        continue_speakers_path = continue_path / "speakers.json"
    elif config.use_d_vector_file:
        continue_speakers_path = config.d_vector_file
    if is_multi_speaker:
        inference_command.extend(["--speaker_idx", "ljspeech-1", "--speakers_file_path", str(continue_speakers_path)])
    run_main(synthesize, inference_command)

    # restore the model and continue training for one more epoch
    run_main(train_tts, ["--continue_path", str(continue_path)])
    shutil.rmtree(tmp_path)
