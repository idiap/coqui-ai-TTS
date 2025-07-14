import logging
import os
from typing import Any, Union

import numpy as np
import torch
from coqpit import Coqpit

from TTS.config import get_from_config_or_model_args_with_default
from TTS.tts.utils.managers import EmbeddingManager

logger = logging.getLogger(__name__)


class SpeakerManager(EmbeddingManager):
    """Manage the speakers for multi-speaker 🐸TTS models. Load a datafile and parse the information
    in a way that can be queried by speaker or clip.

    There are 3 different scenarios considered:

    1. Models using speaker embedding layers. The datafile only maps speaker names to ids used by the embedding layer.
    2. Models using d-vectors. The datafile includes a dictionary in the following format.

    ::

        {
            'clip_name.wav':{
                'name': 'speakerA',
                'embedding'[<d_vector_values>]
            },
            ...
        }


    3. Computing the d-vectors by the speaker encoder. It loads the speaker encoder model and
    computes the d-vectors for a given clip or speaker.

    Args:
        d_vectors_file_path (str, optional): Path to the metafile including x vectors. Defaults to "".
        speaker_id_file_path (str, optional): Path to the metafile that maps speaker names to ids used by
        TTS models. Defaults to "".
        encoder_model_path (str, optional): Path to the speaker encoder model file. Defaults to "".
        encoder_config_path (str, optional): Path to the spealer encoder config file. Defaults to "".

    Examples:
        >>> # load audio processor and speaker encoder
        >>> ap = AudioProcessor(**config.audio)
        >>> manager = SpeakerManager(encoder_model_path=encoder_model_path, encoder_config_path=encoder_config_path)
        >>> # load a sample audio and compute embedding
        >>> waveform = ap.load_wav(sample_wav_path)
        >>> mel = ap.melspectrogram(waveform)
        >>> d_vector = manager.compute_embeddings(mel.T)
    """

    def __init__(
        self,
        data_items: list[dict[str, Any]] | None = None,
        d_vectors_file_path: str = "",
        speaker_id_file_path: str | os.PathLike[Any] = "",
        encoder_model_path: str | os.PathLike[Any] = "",
        encoder_config_path: str | os.PathLike[Any] = "",
        use_cuda: bool = False,
    ):
        super().__init__(
            embedding_file_path=d_vectors_file_path,
            id_file_path=speaker_id_file_path,
            encoder_model_path=encoder_model_path,
            encoder_config_path=encoder_config_path,
            use_cuda=use_cuda,
        )

        if data_items:
            self.set_ids_from_data(data_items, parse_key="speaker_name")

    @property
    def num_speakers(self) -> int:
        return len(self.name_to_id)

    @property
    def speaker_names(self) -> list[str]:
        return list(self.name_to_id.keys())

    @staticmethod
    def init_from_config(
        config: "Coqpit", samples: list[dict[str, Any]] | None = None
    ) -> Union["SpeakerManager", None]:
        """Initialize a speaker manager from config

        Args:
            config (Coqpit): Config object.
            samples (Union[List[List], List[Dict]], optional): List of data samples to parse out the speaker names.
                Defaults to None.

        Returns:
            SpeakerEncoder: Speaker encoder object.
        """
        speaker_manager = None
        if get_from_config_or_model_args_with_default(config, "use_speaker_embedding", False):
            if samples:
                speaker_manager = SpeakerManager(data_items=samples)
            if speaker_file := get_from_config_or_model_args_with_default(config, "speaker_file", None):
                speaker_manager = SpeakerManager(speaker_id_file_path=speaker_file)
            if speakers_file := get_from_config_or_model_args_with_default(config, "speakers_file", None):
                speaker_manager = SpeakerManager(speaker_id_file_path=speakers_file)

        if get_from_config_or_model_args_with_default(config, "use_d_vector_file", False):
            speaker_manager = SpeakerManager()
            if d_vector_file := get_from_config_or_model_args_with_default(config, "d_vector_file", None):
                speaker_manager = SpeakerManager(d_vectors_file_path=d_vector_file)
        return speaker_manager


def get_speaker_balancer_weights(items: list):
    speaker_names = np.array([item["speaker_name"] for item in items])
    unique_speaker_names = np.unique(speaker_names).tolist()
    speaker_ids = [unique_speaker_names.index(l) for l in speaker_names]
    speaker_count = np.array([len(np.where(speaker_names == l)[0]) for l in unique_speaker_names])
    weight_speaker = 1.0 / speaker_count
    dataset_samples_weight = np.array([weight_speaker[l] for l in speaker_ids])
    # normalize
    dataset_samples_weight = dataset_samples_weight / np.linalg.norm(dataset_samples_weight)
    return torch.from_numpy(dataset_samples_weight).float()
