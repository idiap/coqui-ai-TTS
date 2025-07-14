"""Voice cloning utilities."""

import datetime
import importlib.metadata
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from TTS.utils.generic_utils import is_pytorch_at_least_2_4, slugify

logger = logging.getLogger(__name__)


@dataclass
class VoiceMetadata:
    """Holds metadata on a voice.

    Args:
        model:
            Model name, same as used in configs.
        speaker_id:
            User-defined speaker name.
        source_files:
            Paths to audio files used to generate the voice.
        created_at:
            ISO 8601 timestamp at which the voice was generated.
        coqui_version:
            Coqui TTS version that generated the voice.
    """

    model: dict[str, str | float | bool]
    speaker_id: str
    source_files: list[str] | None = None
    created_at: str | None = None
    coqui_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoiceMetadata":
        return cls(**data)


class CloningMixin:
    """Add voice cloning with caching support.

    To be used as a mixin in :py:class:`~TTS.tts.models.base_tts.BaseTTS` and
    :py:class:`~TTS.vc.models.base_vc.BaseVC`-derived models.
    """

    def _create_voice_metadata(
        self, model: dict[str, str | float | bool], speaker_id: str, source_files: list[str]
    ) -> VoiceMetadata:
        return VoiceMetadata(
            model=model,
            speaker_id=speaker_id,
            source_files=source_files,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="minutes"),
            coqui_version=importlib.metadata.version("coqui-tts"),
        )

    def clone_voice(
        self,
        speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]] | None,
        speaker_id: str | None = None,
        voice_dir: str | os.PathLike[Any] | None = None,
        **generate_kwargs: Any,
    ) -> dict[str, Any]:
        """Load a pregenerated voice or generate one from the given reference  audio.

        If ``speaker_wav`` is not specified, loads the voice from ``<voice_dir>/<speaker_id>.pth``.

        The voice will be cached in ``<voice_dir>/<speaker_id>.pth`` if those parameters are
        specified. If there already is a voice for ``speaker_id``, it will be overwritten with
        the newly generated one.

        Args:
            speaker_wav:
                Path(s) to the reference audio files.
            speaker_id:
                Speaker ID to be assigned when saving the voice (if not ``None``).
            voice_dir:
                Directory to save the voice (if not ``None``).
            **generate_kwargs:
                Arguments passed on to the model-specific ``_clone_voice()``.

        """
        if speaker_wav is None or (isinstance(speaker_wav, list) and len(speaker_wav) == 0):
            if speaker_id is None:
                msg = "Neither `speaker_wav` nor `speaker_id` was specified"
                raise RuntimeError(msg)
            if voice_dir is None:
                msg = "Specified only `speaker_id`, but no `voice_dir` to load the voice from"
                raise RuntimeError(msg)
            return self.load_voice_file(speaker_id, voice_dir)
        voice, model_metadata = self._clone_voice(speaker_wav, **generate_kwargs)
        logger.info("Generated voice from reference audio")
        if speaker_id is not None and voice_dir is not None:
            speaker_id = slugify(speaker_id)
            voice_fn = Path(voice_dir) / f"{speaker_id}.pth"
            voice_fn.parent.mkdir(exist_ok=True, parents=True)
            speaker_wav = speaker_wav if isinstance(speaker_wav, list) else [speaker_wav]
            metadata = self._create_voice_metadata(model_metadata, speaker_id, [str(p) for p in speaker_wav])
            voices = self.get_voices(voice_dir)
            if speaker_id in voices:
                logger.info("Voice `%s` already exists in `%s`, overwriting it", speaker_id, voice_fn)
            voice_dict = {**voice, "metadata": metadata.to_dict()}
            torch.save(voice_dict, voice_fn)
            logger.info("Voice `%s` saved to: %s", speaker_id, voice_fn)
        return voice

    def _clone_voice(
        self,
        speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]],
        **generate_kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Generate a voice from the given reference audio.

        To be implemented in model subclasses.

        Args:
            speaker_wav: Path(s) to the reference audio files.

        Returns:
            - dictionary with the embedding(s)
            - dictionary with any model-specific metadata, e.g. model name
        """
        raise NotImplementedError

    def load_voice_file(
        self,
        speaker_id: str,
        voice_dir: str | os.PathLike[Any],
    ) -> dict[str, Any]:
        """Load the voice for the given speaker.

        Args:
            speaker_id:
                Speaker ID to load.
            voice_dir:
                Directory where to look for the voice.
        """
        voices = self.get_voices(voice_dir)
        if speaker_id not in voices:
            msg = f"Voice file `{slugify(speaker_id)}.pth` for speaker `{speaker_id}` not found in: {voice_dir}"
            raise FileNotFoundError(msg)
        voice = torch.load(voices[speaker_id], map_location="cpu", weights_only=is_pytorch_at_least_2_4())
        logger.info("Loaded voice `%s` from: %s", speaker_id, voices[speaker_id])
        return voice

    def get_voices(self, voice_dir: str | os.PathLike[Any]) -> dict[str, Path]:
        """Return all available voices in the given directory.

        Args:
            voice_dir: Directory to search for voices.

        Returns:
            Dictionary mapping a speaker ID to its voice file.
        """
        return {path.stem: path for path in Path(voice_dir).glob("*.pth")}
