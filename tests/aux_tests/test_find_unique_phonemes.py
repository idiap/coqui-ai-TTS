import torch

from tests import run_main
from TTS.bin.find_unique_phonemes import main
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig

torch.manual_seed(1)

dataset_config_en = BaseDatasetConfig(
    formatter="ljspeech",
    meta_file_train="metadata.csv",
    meta_file_val="metadata.csv",
    path="tests/data/ljspeech",
    language="en",
)

"""
dataset_config_pt = BaseDatasetConfig(
    formatter="ljspeech",
    meta_file_train="metadata.csv",
    meta_file_val="metadata.csv",
    path="tests/data/ljspeech",
    language="pt-br",
)
"""


def test_find_phonemes(tmp_path):
    # prepare the config
    config_path = str(tmp_path / "test_model_config.json")
    config = VitsConfig(
        batch_size=2,
        eval_batch_size=2,
        num_loader_workers=0,
        num_eval_loader_workers=0,
        text_cleaner="english_cleaners",
        use_phonemes=True,
        phoneme_language="en-us",
        phoneme_cache_path="tests/data/ljspeech/phoneme_cache/",
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1,
        print_step=1,
        print_eval=True,
        datasets=[dataset_config_en],
    )
    config.save_json(config_path)

    # run test
    run_main(main, ["--config_path", config_path])
