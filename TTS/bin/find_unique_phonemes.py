"""Find all the unique characters in a dataset."""

import argparse
import logging
import multiprocessing
import sys
from argparse import RawTextHelpFormatter
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from TTS.config import load_config
from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.languages import normalize_language
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.generic_utils import ConsoleFormatter, setup_logger

tokenizer: TTSTokenizer


def init_worker(config: BaseTTSConfig) -> None:
    """Build the model's tokenizer once per worker."""
    global tokenizer
    tokenizer, _ = TTSTokenizer.init_from_config(config)


def compute_phonemes(item: dict) -> set[str]:
    """Clean and phonemize as in text_to_ids, without discarding unknown symbols."""
    language = item["language"]
    text = item["text"]
    if tokenizer.text_cleaner is not None:
        text = tokenizer.text_cleaner(text, lang=normalize_language(language))
    ph = tokenizer.phonemizer.phonemize(text, separator="", language=language)
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
    args = parse_args(arg_list)
    config = load_config(args.config_path)

    if not config.use_phonemes:
        msg = "Set use_phonemes=True in the config to find its phonemes."
        raise ValueError(msg)
    # Validate before spawning workers so configuration errors propagate directly.
    _, config = TTSTokenizer.init_from_config(config)

    # load all datasets
    train_items, eval_items = load_tts_samples(
        config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    items = train_items + eval_items
    print("Num items:", len(items))

    with ProcessPoolExecutor(
        max_workers=multiprocessing.cpu_count(),
        initializer=init_worker,
        initargs=(config,),
    ) as executor:
        phonemes = list(tqdm(executor.map(compute_phonemes, items, chunksize=15), total=len(items)))
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
