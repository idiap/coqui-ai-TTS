import logging
from typing import Any

import numpy as np
import torch

from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.tts.utils.managers import BaseIDManager

logger = logging.getLogger(__name__)


def normalize_language(language: str | None) -> str | None:
    """Remove any region codes from the language, e.g. 'en-US' -> 'en'."""
    return None if language is None else language.split("-")[0]


class LanguageManager(BaseIDManager):
    """Manage the languages for multi-lingual 🐸TTS models.

    Args:
        ids_file_path: Path to the metafile that maps language names to ids used by TTS models. Defaults to "".

    Examples:
        >>> manager = LanguageManager("language_ids.json")
        >>> language_id_mapper = manager.name_to_id
    """

    @property
    def num_languages(self) -> int:
        return len(list(self.name_to_id.keys()))

    @property
    def language_names(self) -> list[str]:
        return sorted(self.name_to_id.keys())

    @staticmethod
    def parse_language_ids_from_config(c: BaseTTSConfig) -> dict[str, int]:
        """Set language id from config.

        1. Read config.languages
        2. Otherwise read language names from the dataset configs
        3. Otherwise read config.phoneme_language

        Args:
            c (BaseTTSConfig): Config

        Returns:
            Language ID mapping.
        """
        languages = c.languages
        if len(languages) == 0:
            dataset_languages = set({})
            for dataset in c.datasets:
                if language := dataset.get("language"):
                    dataset_languages.add(language)
                else:
                    logger.warning("Dataset `%s` has no language specified.", dataset.get("dataset_name"))
            languages = sorted(dataset_languages)
        if len(languages) == 0 and c.phoneme_language:
            languages = [c.phoneme_language]
        if len(languages) == 0:
            languages = ["en"]
            logger.warning("Could not identify language from config. Initializing with English for text processing.")
        logger.debug("Language manager initialized with: %s", languages)
        return {name: i for i, name in enumerate(languages)}

    @staticmethod
    def parse_ids_from_data(items: list[dict[str, Any]], parse_key: str) -> Any:
        raise NotImplementedError

    def set_ids_from_data(self, items: list[dict[str, Any]], parse_key: str) -> Any:
        raise NotImplementedError

    @staticmethod
    def init_from_config(config: BaseTTSConfig) -> "LanguageManager":
        """Initialize the language manager from the config and update config.languages.

        Args:
            config: BaseTTSConfig
        """
        if (path := config.model_args.get("language_ids_file")) and config.model_args.get("use_language_embedding"):
            language_manager = LanguageManager(path)
        else:
            language_manager = LanguageManager()
            language_manager.name_to_id = LanguageManager.parse_language_ids_from_config(config)
        # Do not sort this list to allow restoring the exact name_to_id mapping
        config.languages = list(language_manager.name_to_id.keys())
        return language_manager


def get_language_balancer_weights(items: list[dict[str, Any]]) -> torch.Tensor:
    language_names = np.array([item["language"] for item in items])
    unique_language_names = np.unique(language_names).tolist()
    language_ids = [unique_language_names.index(l) for l in language_names]
    language_count = np.array([len(np.where(language_names == l)[0]) for l in unique_language_names])
    weight_language = 1.0 / language_count
    # get weight for each sample
    dataset_samples_weight = np.array([weight_language[l] for l in language_ids])
    # normalize
    dataset_samples_weight = dataset_samples_weight / np.linalg.norm(dataset_samples_weight)
    return torch.from_numpy(dataset_samples_weight).float()
