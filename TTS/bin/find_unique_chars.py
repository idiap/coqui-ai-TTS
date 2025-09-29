"""Find all the unique characters in a dataset"""

import argparse
import logging
import sys
from argparse import RawTextHelpFormatter

from TTS.config import load_config
from TTS.tts.datasets import find_unique_chars, load_tts_samples
from TTS.utils.generic_utils import ConsoleFormatter, setup_logger


def main():
    setup_logger("TTS", level=logging.INFO, stream=sys.stdout, formatter=ConsoleFormatter())

    # pylint: disable=bad-option-value
    parser = argparse.ArgumentParser(
        description="""Find all the unique characters or phonemes in a dataset.\n\n"""
        """
    Example runs:

    python TTS/bin/find_unique_chars.py --config_path config.json
    """,
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("--config_path", type=str, help="Path to dataset config file.", required=True)
    args = parser.parse_args()

    c = load_config(args.config_path)

    # load all datasets
    train_items, eval_items = load_tts_samples(
        c.datasets, eval_split=True, eval_split_max_size=c.eval_split_max_size, eval_split_size=c.eval_split_size
    )

    items = train_items + eval_items
    find_unique_chars(items)


if __name__ == "__main__":
    main()
