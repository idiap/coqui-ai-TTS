"""Find all the unique characters in a dataset."""

import argparse
import logging
import multiprocessing
import sys
from argparse import RawTextHelpFormatter

from tqdm.contrib.concurrent import process_map

from TTS.config import load_config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.text.phonemizers import Gruut
from TTS.utils.generic_utils import ConsoleFormatter, setup_logger


def compute_phonemes(item: dict) -> set[str]:
    text = item["text"]
    ph = phonemizer.phonemize(text).replace("|", "")
    return set(ph)


def parse_args(arg_list: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""Find all the unique characters or phonemes in a dataset.\n\n"""
        """
    Example runs:

    python TTS/bin/find_unique_phonemes.py --config_path config.json
    """,
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("--config_path", type=str, help="Path to dataset config file.", required=True)
    return parser.parse_args(arg_list)


def main(arg_list: list[str] | None = None) -> None:
    setup_logger("TTS", level=logging.INFO, stream=sys.stdout, formatter=ConsoleFormatter())
    global phonemizer
    args = parse_args(arg_list)
    config = load_config(args.config_path)

    # load all datasets
    train_items, eval_items = load_tts_samples(
        config.datasets,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    items = train_items + eval_items
    print("Num items:", len(items))

    language_list = [item["language"] for item in items]
    is_lang_def = all(language_list)

    if not config.phoneme_language or not is_lang_def:
        msg = "Phoneme language must be defined in config."
        raise ValueError(msg)

    if language_list.count(language_list[0]) != len(language_list):
        msg = (
            "Currently, just one phoneme language per config file is supported !! "
            "Please split the dataset config into different configs and run it individually for each language !!"
        )
        raise ValueError(msg)

    phonemizer = Gruut(language=language_list[0], keep_puncs=True)

    phonemes = process_map(compute_phonemes, items, max_workers=multiprocessing.cpu_count(), chunksize=15)
    phones = []
    for ph in phonemes:
        phones.extend(ph)

    phones = set(phones)
    lower_phones = filter(lambda c: c.islower(), phones)
    phones_force_lower = [c.lower() for c in phones]
    phones_force_lower = set(phones_force_lower)

    print(f" > Number of unique phonemes: {len(phones)}")
    print(f" > Unique phonemes: {''.join(sorted(phones))}")
    print(f" > Unique lower phonemes: {''.join(sorted(lower_phones))}")
    print(f" > Unique all forced to lower phonemes: {''.join(sorted(phones_force_lower))}")
    sys.exit(0)


if __name__ == "__main__":
    main()
