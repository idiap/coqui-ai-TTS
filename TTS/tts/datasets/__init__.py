import logging
import os
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from TTS.config import get_from_config_or_model_args
from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.tts.datasets.dataset import *
from TTS.tts.datasets.formatters import _FORMATTER_REGISTRY, Formatter, register_formatter

logger = logging.getLogger(__name__)


def split_dataset(
    items: list[dict[str, Any]], eval_split_max_size: int | None = None, eval_split_size: float = 0.01
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split a dataset into train and eval. Consider speaker distribution in multi-speaker training.

    Args:
        items (list[list]):
            A list of samples. Each sample is a list of `[audio_path, text, speaker_id]`.

        eval_split_max_size (int):
            Number maximum of samples to be used for evaluation in proportion split. Defaults to None (Disabled).

        eval_split_size (float):
            If between 0.0 and 1.0 represents the proportion of the dataset to include in the evaluation set.
            If > 1, represents the absolute number of evaluation samples. Defaults to 0.01 (1%).
    """
    speakers = [item["speaker_name"] for item in items]
    is_multi_speaker = len(set(speakers)) > 1
    if eval_split_size > 1:
        eval_split_size = int(eval_split_size)
    else:
        if eval_split_max_size:
            eval_split_size = min(eval_split_max_size, int(len(items) * eval_split_size))
        else:
            eval_split_size = int(len(items) * eval_split_size)

    assert eval_split_size > 0, (
        f" [!] You do not have enough samples for the evaluation set. You can work around this setting the 'eval_split_size' parameter to a minimum of {1 / len(items)}"
    )
    np.random.seed(0)
    np.random.shuffle(items)
    if is_multi_speaker:
        items_eval = []
        speakers = [item["speaker_name"] for item in items]
        speaker_counter = Counter(speakers)
        while len(items_eval) < eval_split_size:
            item_idx = np.random.randint(0, len(items))
            speaker_to_be_removed = items[item_idx]["speaker_name"]
            if speaker_counter[speaker_to_be_removed] > 1:
                items_eval.append(items[item_idx])
                speaker_counter[speaker_to_be_removed] -= 1
                del items[item_idx]
        return items_eval, items
    return items[:eval_split_size], items[eval_split_size:]


def add_extra_keys(metadata: list[dict[str, Any]], language: str, dataset_name: str):
    for item in metadata:
        # add language name
        item["language"] = language
        # add unique audio name
        relfilepath = Path(item["audio_file"]).relative_to(item["root_path"]).with_suffix("")
        item["audio_unique_name"] = f"{dataset_name}#{relfilepath}"
    return metadata


def load_tts_samples(
    config: BaseTTSConfig,
    *,
    eval_split: bool = True,
    formatter: Formatter | None = None,
    eval_split_max_size: int | None = None,
    eval_split_size: float = 0.01,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse the datasets from config and automatically populate speaker info.

    Load the samples as a list and load the attention alignments if provided. If
    `formatter` is not None, apply the formatter to the samples else pick the
    formatter from the available ones based on the dataset name.

    If the dataset contains speaker information, it will be extracted and stored
    in config.speakers. If no speaker information is present, the config
    remains unchanged.

    Args:
        config: BaseTTSConfig instance.

        eval_split (bool, optional): If true, create a evaluation split. If an eval split provided explicitly, generate
            an eval split automatically. Defaults to True.

        formatter (Callable, optional): The preprocessing function to be applied to create the list of samples. It
            must take the root_path and the meta_file name and return a list of samples in the format of
            `[[text, audio_path, speaker_id], ...]]`. See the available formatters in `TTS.tts.dataset.formatter` as
            example. Defaults to None.

        eval_split_max_size (int):
            Number maximum of samples to be used for evaluation in proportion split. Defaults to None (Disabled).

        eval_split_size (float):
            If between 0.0 and 1.0 represents the proportion of the dataset to include in the evaluation set.
            If > 1, represents the absolute number of evaluation samples. Defaults to 0.01 (1%).

    Returns:
        tuple[list[dict], list[dict]]: training and evaluation splits of the dataset.
    """
    if not config.has("datasets"):
        msg = (
            "From coqui-tts 0.28.0 you need to pass a config instance to"
            "`load_tts_samples()` that contains a `datasets` field, instead of directly"
            "passing a BaseDatasetConfig list as before."
        )
        raise TypeError(msg)
    meta_data_train_all = []
    meta_data_eval_all = []
    datasets = config.datasets
    for dataset in datasets:
        formatter_name = dataset["formatter"]
        dataset_name = dataset["dataset_name"]
        root_path = dataset["path"]
        meta_file_train = dataset["meta_file_train"]
        meta_file_val = dataset["meta_file_val"]
        ignored_speakers = dataset["ignored_speakers"]
        language = dataset["language"]

        # setup the right data processor
        if formatter is None:
            formatter = _get_formatter_by_name(formatter_name)
        # load train set
        meta_data_train = formatter(root_path, meta_file_train, ignored_speakers=ignored_speakers)
        assert len(meta_data_train) > 0, f" [!] No training samples found in {root_path}/{meta_file_train}"

        meta_data_train = add_extra_keys(meta_data_train, language, dataset_name)

        logger.info("Found %d files in %s", len(meta_data_train), Path(root_path).resolve())
        # load evaluation split if set
        if eval_split:
            if meta_file_val:
                meta_data_eval = formatter(root_path, meta_file_val, ignored_speakers=ignored_speakers)
                meta_data_eval = add_extra_keys(meta_data_eval, language, dataset_name)
            else:
                eval_size_per_dataset = eval_split_max_size // len(datasets) if eval_split_max_size else None
                meta_data_eval, meta_data_train = split_dataset(meta_data_train, eval_size_per_dataset, eval_split_size)
            meta_data_eval_all += meta_data_eval
        meta_data_train_all += meta_data_train
        # load attention masks for the duration predictor training
        if dataset.meta_file_attn_mask:
            meta_data = dict(load_attention_mask_meta_data(dataset["meta_file_attn_mask"]))
            for meta_data_all in (meta_data_train_all, meta_data_eval_all):
                for idx, ins in enumerate(meta_data_all):
                    attn_file = meta_data[ins["audio_file"]].strip()
                    meta_data_all[idx].update({"alignment_file": attn_file})
        # set none for the next iter
        formatter = None

    # Parse speaker info
    if get_from_config_or_model_args(config, "use_speaker_embedding"):
        speakers = sorted(
            set(s["speaker_name"].strip() for s in meta_data_train_all + meta_data_eval_all if "speaker_name" in s)
        )
        logger.debug("Found %d speakers in dataset", len(speakers))
        if speakers:
            config.speakers = speakers
    return meta_data_train_all, meta_data_eval_all


def load_attention_mask_meta_data(metafile_path: str | os.PathLike[Any]):
    """Load meta data file created by compute_attention_masks.py"""
    with open(metafile_path, encoding="utf-8") as f:
        lines = f.readlines()

    meta_data = []
    for line in lines:
        wav_file, attn_file = line.split("|")
        meta_data.append([wav_file, attn_file])
    return meta_data


def _get_formatter_by_name(name: str) -> Formatter:
    """Returns the respective preprocessing function."""
    if name.lower() not in _FORMATTER_REGISTRY:
        msg = f"{name} formatter not found. If it is a custom formatter, make sure to call register_formatter() first."
        raise ValueError(msg)
    return _FORMATTER_REGISTRY[name.lower()]


def find_unique_chars(data_samples: list[dict[str, Any]]) -> set[str]:
    texts = "".join(item["text"] for item in data_samples)
    chars = set(texts)
    lower_chars = filter(lambda c: c.islower(), chars)
    chars_force_lower = [c.lower() for c in chars]
    chars_force_lower = set(chars_force_lower)

    logger.info("Number of unique characters: %d", len(chars))
    logger.info("Unique characters: %s", "".join(sorted(chars)))
    logger.info("Unique lower characters: %s", "".join(sorted(lower_chars)))
    logger.info("Unique all forced to lower characters: %s", "".join(sorted(chars_force_lower)))
    return chars_force_lower
