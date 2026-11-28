from dataclasses import replace

import pytest
import torch

from tests import run_main
from TTS.bin.find_unique_phonemes import main
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.text.tokenizer import TTSTokenizer

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


def assert_phonemes_match_tokenizer(config, output):
    phones_line = next(line for line in output.splitlines() if line.startswith(" > Unique phonemes: "))
    phones = set(phones_line.removeprefix(" > Unique phonemes: "))

    tokenizer, _ = TTSTokenizer.init_from_config(config)
    train_items, eval_items = load_tts_samples(config, eval_split=True)
    expected = set()
    for item in train_items + eval_items:
        expected.update(tokenizer.text_to_phonemes(item["text"], item["language"]))
    assert phones == expected
    assert f" > Number of unique phonemes: {len(expected)}" in output


@pytest.mark.parametrize(
    ("phonemizer", "languages"),
    [(None, ["en"]), ("gruut", [""]), ("multi_phonemizer", ["en", "de"])],
    ids=["default", "gruut", "multi_phonemizer"],
)
def test_find_phonemes(tmp_path, capsys, monkeypatch, phonemizer, languages):
    # Avoid starting one worker per CPU core.
    monkeypatch.setattr("TTS.bin.find_unique_phonemes.multiprocessing.cpu_count", lambda: 2)
    # prepare the config
    config_path = str(tmp_path / "test_model_config.json")
    config = VitsConfig(
        batch_size=2,
        eval_batch_size=2,
        num_loader_workers=0,
        num_eval_loader_workers=0,
        text_cleaner="english_cleaners",
        use_phonemes=True,
        phonemizer=phonemizer,
        phoneme_language=None if phonemizer == "multi_phonemizer" else "en-us",
        phoneme_cache_path="tests/data/ljspeech/phoneme_cache/",
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1,
        print_step=1,
        print_eval=True,
        datasets=[replace(dataset_config_en, language=language) for language in languages],
    )
    config.save_json(config_path)

    # run test
    run_main(main, ["--config_path", config_path])
    assert_phonemes_match_tokenizer(config, capsys.readouterr().out)


def test_find_phonemes_invalid_language(tmp_path):
    config_path = str(tmp_path / "test_model_config.json")
    config = VitsConfig(use_phonemes=True, phonemizer="espeak", phoneme_language=None, datasets=[dataset_config_en])
    config.save_json(config_path)

    with pytest.raises(RuntimeError, match='language "None" is not supported by the espeak backend'):
        main(["--config_path", config_path])


def test_find_phonemes_requires_use_phonemes(tmp_path):
    config_path = str(tmp_path / "test_model_config.json")
    VitsConfig(use_phonemes=False, datasets=[dataset_config_en]).save_json(config_path)

    with pytest.raises(ValueError, match="use_phonemes"):
        main(["--config_path", config_path])
