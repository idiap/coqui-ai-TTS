from typing import Any

import numpy as np
import torch

from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.tts.utils.managers import BaseIDManager


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

        1. Check if the config contains a `languages` field.
        2. Otherwise read language names from the dataset configs.

        Args:
            c (BaseTTSConfig): Config

        Returns:
            Language ID mapping.
        """
        languages = c.get("languages", [])
        if len(languages) == 0:
            dataset_languages = set({})
            for dataset in c.datasets:
                if "language" in dataset:
                    dataset_languages.add(dataset["language"])
                else:
                    raise ValueError(f"Dataset {dataset['name']} has no language specified.")
            languages = sorted(dataset_languages)
        return {name: i for i, name in enumerate(languages)}

    @staticmethod
    def parse_ids_from_data(items: list[dict[str, Any]], parse_key: str) -> Any:
        raise NotImplementedError

    def set_ids_from_data(self, items: list[dict[str, Any]], parse_key: str) -> Any:
        raise NotImplementedError

    @staticmethod
    def init_from_config(config: BaseTTSConfig) -> "LanguageManager":
        """Initialize the language manager from a BaseTTSConfig.

        Args:
            config: BaseTTSConfig
        """
        if (path := config.model_args.get("language_ids_file")) and config.model_args.get("use_language_embedding"):
            return LanguageManager(path)
        # Fall back to parse language IDs from datasets listed in the config
        language_manager = LanguageManager()
        language_manager.name_to_id = LanguageManager.parse_language_ids_from_config(config)
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
