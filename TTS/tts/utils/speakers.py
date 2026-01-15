import logging
import os
from typing import Any, Union

import numpy as np
import torch

from TTS.config import get_from_config_or_model_args
from TTS.config.shared_configs import BaseTrainingConfig
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
        d_vectors_file_path: str | os.PathLike[Any] | list[str | os.PathLike[Any]] | None = None,
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
    def init_from_config(config: BaseTrainingConfig) -> Union["SpeakerManager", None]:
        """Initialize a speaker manager from config.

        If using speaker embeddings:
          - Read config.speakers
          - Otherwise read speaker IDs from speaker(s)_file if set
        Else if using d-vectors:
          - Read embeddings from d_vector_file

        Initialize speaker encoder if speaker_encoder_model_path is set.

        Args:
            config: Config object.
        """
        speaker_manager = None
        if get_from_config_or_model_args(config, "use_speaker_embedding"):
            if config.speakers:
                speaker_manager = SpeakerManager()
                speaker_manager.name_to_id = {name: i for i, name in enumerate(config.speakers)}
            # TODO: check if speaker_file is used by any model
            elif speaker_file := get_from_config_or_model_args(config, "speaker_file"):
                speaker_manager = SpeakerManager(speaker_id_file_path=speaker_file)
            elif speakers_file := get_from_config_or_model_args(config, "speakers_file"):
                speaker_manager = SpeakerManager(speaker_id_file_path=speakers_file)
        elif get_from_config_or_model_args(config, "use_d_vector_file"):
            speaker_manager = SpeakerManager()
            if d_vector_file := get_from_config_or_model_args(config, "d_vector_file"):
                speaker_manager = SpeakerManager(d_vectors_file_path=d_vector_file)

        if encoder_path := config.model_args.get("speaker_encoder_model_path"):
            if speaker_manager is None:
                speaker_manager = SpeakerManager()
            speaker_manager.init_encoder(encoder_path, config.model_args.speaker_encoder_config_path)
            logger.debug("Encoder initialized in speaker manager.")

        if speaker_manager is not None:
            logger.debug("Speaker manager initialized with: %s", speaker_manager.speaker_names)
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
