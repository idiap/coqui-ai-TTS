"""Coqui TTS Python API."""

import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

from torch import nn

from TTS.config import load_config
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

logger = logging.getLogger(__name__)


class TTS(nn.Module):
    """Coqui Python API."""

    def __init__(
        self,
        model_name: str = "",
        *,
        model_path: str | None = None,
        config_path: str | None = None,
        vocoder_name: str | None = None,
        vocoder_path: str | None = None,
        vocoder_config_path: str | None = None,
        encoder_path: str | None = None,
        encoder_config_path: str | None = None,
        speakers_file_path: str | None = None,
        language_ids_file_path: str | None = None,
        progress_bar: bool = True,
        gpu: bool = False,
    ) -> None:
        """🐸TTS python interface that allows to load and use the released models.

        Example with a multi-speaker model:
            >>> from TTS.api import TTS
            >>> tts = TTS(TTS.list_models()[0])
            >>> wav = tts.tts("This is a test! This is also a test!!", speaker=tts.speakers[0], language=tts.languages[0])
            >>> tts.tts_to_file(text="Hello world!", speaker=tts.speakers[0], language=tts.languages[0], file_path="output.wav")

        Example with a single-speaker model:
            >>> tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC", progress_bar=False)
            >>> tts.tts_to_file(text="Ich bin eine Testnachricht.", file_path="output.wav")

        Example loading a model from a path:
            >>> tts = TTS(model_path="/path/to/checkpoint_100000.pth", config_path="/path/to/config.json", progress_bar=False)
            >>> tts.tts_to_file(text="Ich bin eine Testnachricht.", file_path="output.wav")

        Example voice cloning with YourTTS in English, French and Portuguese:
            >>> tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to("cuda")
            >>> tts.tts_to_file("This is voice cloning.", speaker_wav="my/cloning/audio.wav", language="en", file_path="thisisit.wav")
            >>> tts.tts_to_file("C'est le clonage de la voix.", speaker_wav="my/cloning/audio.wav", language="fr", file_path="thisisit.wav")
            >>> tts.tts_to_file("Isso é clonagem de voz.", speaker_wav="my/cloning/audio.wav", language="pt", file_path="thisisit.wav")

        Example Fairseq TTS models (uses ISO language codes in https://dl.fbaipublicfiles.com/mms/tts/all-tts-languages.html):
            >>> tts = TTS(model_name="tts_models/eng/fairseq/vits", progress_bar=False).to("cuda")
            >>> tts.tts_to_file("This is a test.", file_path="output.wav")

        Args:
            model_name (str, optional): Model name to load. You can list models by ```tts.models```. Defaults to None.
            model_path (str, optional): Path to the model checkpoint. Defaults to None.
            config_path (str, optional): Path to the model config. Defaults to None.
            vocoder_name (str, optional): Pre-trained vocoder to use. Defaults to None, i.e. using the default vocoder.
            vocoder_path (str, optional): Path to the vocoder checkpoint. Defaults to None.
            vocoder_config_path (str, optional): Path to the vocoder config. Defaults to None.
            encoder_path: Path to speaker encoder checkpoint. Default to None.
            encoder_config_path: Path to speaker encoder config file. Defaults to None.
            speakers_file_path: JSON file for multi-speaker model. Defaults to None.
            language_ids_file_path: JSON file for multilingual model. Defaults to None
            progress_bar (bool, optional): Whether to print a progress bar while downloading a model. Defaults to True.
            gpu (bool, optional): Enable/disable GPU. Defaults to False. DEPRECATED, use TTS(...).to("cuda")
        """
        super().__init__()
        self.manager = ModelManager(models_file=self.get_models_file_path(), progress_bar=progress_bar)
        self.config = load_config(config_path) if config_path else None
        self.synthesizer: Synthesizer | None = None
        self.voice_converter: Synthesizer | None = None
        self.model_name = ""

        self.vocoder_path = vocoder_path
        self.vocoder_config_path = vocoder_config_path
        self.encoder_path = encoder_path
        self.encoder_config_path = encoder_config_path
        self.speakers_file_path = speakers_file_path
        self.language_ids_file_path = language_ids_file_path

        if gpu:
            warnings.warn("`gpu` will be deprecated. Please use `tts.to(device)` instead.")

        if model_name is not None and len(model_name) > 0:
            if "tts_models" in model_name:
                self.load_tts_model_by_name(model_name, vocoder_name, gpu=gpu)
            elif "voice_conversion_models" in model_name:
                self.load_vc_model_by_name(model_name, vocoder_name, gpu=gpu)
            # To allow just TTS("xtts")
            else:
                self.load_model_by_name(model_name, vocoder_name, gpu=gpu)

        if model_path:
            self.load_tts_model_by_path(model_path, config_path, gpu=gpu)

    @property
    def models(self) -> list[str]:
        return self.manager.list_tts_models()

    @property
    def is_multi_speaker(self) -> bool:
        if self.synthesizer is not None:
            if self.synthesizer.tts_model.config.supports_cloning:
                return True
            if hasattr(self.synthesizer.tts_model, "speaker_manager") and self.synthesizer.tts_model.speaker_manager:
                return self.synthesizer.tts_model.speaker_manager.num_speakers > 1
        return False

    @property
    def is_multi_lingual(self) -> bool:
        # Not sure what sets this to None, but applied a fix to prevent crashing.
        if (
            isinstance(self.model_name, str)
            and "xtts" in self.model_name
            or self.config
            and ("xtts" in self.config.model or "languages" in self.config and len(self.config.languages) > 1)
        ):
            return True
        if (
            self.synthesizer is not None
            and hasattr(self.synthesizer.tts_model, "language_manager")
            and self.synthesizer.tts_model.language_manager
        ):
            return self.synthesizer.tts_model.language_manager.num_languages > 1
        return False

    @property
    def speakers(self) -> list[str] | None:
        if not self.is_multi_speaker:
            return None
        speakers = []
        if self.synthesizer.tts_model.config.supports_cloning:
            speakers.extend(self.synthesizer.tts_model.get_voices(self.synthesizer.voice_dir).keys())
        if self.synthesizer.tts_model.speaker_manager is not None:
            speakers.extend(self.synthesizer.tts_model.speaker_manager.speaker_names)
        return speakers

    @property
    def languages(self) -> list[str] | None:
        if not self.is_multi_lingual:
            return None
        return self.synthesizer.tts_model.language_manager.language_names

    @staticmethod
    def get_models_file_path() -> Path:
        return Path(__file__).parent / ".models.json"

    @staticmethod
    def list_models() -> list[str]:
        return ModelManager(models_file=TTS.get_models_file_path(), progress_bar=False).list_models()

    def download_model_by_name(
        self, model_name: str, vocoder_name: str | None = None
    ) -> tuple[Path | None, Path | None, Path | None, Path | None, Path | None]:
        model_path, config_path, model_item = self.manager.download_model(model_name)
        if (
            "fairseq" in model_name
            or "openvoice" in model_name
            or (
                model_item is not None
                and isinstance(model_item["model_url"], list)
                and len(model_item["model_url"]) > 2
            )
        ):
            # return model directory if there are multiple files
            # we assume that the model knows how to load itself
            return None, None, None, None, model_path
        if model_item.get("default_vocoder") is None:
            return model_path, config_path, None, None, None
        if vocoder_name is None:
            vocoder_name = model_item["default_vocoder"]
        vocoder_path, vocoder_config_path = None, None
        # A local vocoder model will take precedence if already specified in __init__
        if model_item["model_type"] == "tts_models":
            vocoder_path = self.vocoder_path
            vocoder_config_path = self.vocoder_config_path
        if vocoder_path is None or vocoder_config_path is None:
            vocoder_path, vocoder_config_path, _ = self.manager.download_model(vocoder_name)
        return model_path, config_path, vocoder_path, vocoder_config_path, None

    def load_model_by_name(self, model_name: str, vocoder_name: str | None = None, *, gpu: bool = False) -> None:
        """Load one of the 🐸TTS models by name.

        Args:
            model_name (str): Model name to load. You can list models by ```tts.models```.
            gpu (bool, optional): Enable/disable GPU. Some models might be too slow on CPU. Defaults to False.
        """
        self.load_tts_model_by_name(model_name, vocoder_name, gpu=gpu)

    def load_vc_model_by_name(self, model_name: str, vocoder_name: str | None = None, *, gpu: bool = False) -> None:
        """Load one of the voice conversion models by name.

        Args:
            model_name (str): Model name to load. You can list models by ```tts.models```.
            gpu (bool, optional): Enable/disable GPU. Some models might be too slow on CPU. Defaults to False.
        """
        self.model_name = model_name
        model_path, config_path, vocoder_path, vocoder_config_path, model_dir = self.download_model_by_name(
            model_name, vocoder_name
        )
        self.voice_converter = Synthesizer(
            vc_checkpoint=model_path,
            vc_config=config_path,
            vocoder_checkpoint=vocoder_path,
            vocoder_config=vocoder_config_path,
            model_dir=model_dir,
            use_cuda=gpu,
        )

    def load_tts_model_by_name(self, model_name: str, vocoder_name: str | None = None, *, gpu: bool = False) -> None:
        """Load one of 🐸TTS models by name.

        Args:
            model_name (str): Model name to load. You can list models by ```tts.models```.
            gpu (bool, optional): Enable/disable GPU. Some models might be too slow on CPU. Defaults to False.

        TODO: Add tests
        """
        self.model_name = model_name

        model_path, config_path, vocoder_path, vocoder_config_path, model_dir = self.download_model_by_name(
            model_name, vocoder_name
        )

        # init synthesizer
        # None values are fetch from the model
        self.synthesizer = Synthesizer(
            tts_checkpoint=model_path,
            tts_config_path=config_path,
            tts_speakers_file=None,
            tts_languages_file=None,
            vocoder_checkpoint=vocoder_path,
            vocoder_config=vocoder_config_path,
            encoder_checkpoint=self.encoder_path,
            encoder_config=self.encoder_config_path,
            model_dir=model_dir,
            use_cuda=gpu,
        )

    def load_tts_model_by_path(self, model_path: str, config_path: str, *, gpu: bool = False) -> None:
        """Load a model from a path.

        Args:
            model_path (str): Path to the model checkpoint.
            config_path (str): Path to the model config.
            vocoder_path (str, optional): Path to the vocoder checkpoint. Defaults to None.
            vocoder_config (str, optional): Path to the vocoder config. Defaults to None.
            gpu (bool, optional): Enable/disable GPU. Some models might be too slow on CPU. Defaults to False.
        """

        self.synthesizer = Synthesizer(
            tts_checkpoint=model_path,
            tts_config_path=config_path,
            tts_speakers_file=self.speakers_file_path,
            tts_languages_file=self.language_ids_file_path,
            vocoder_checkpoint=self.vocoder_path,
            vocoder_config=self.vocoder_config_path,
            encoder_checkpoint=self.encoder_path,
            encoder_config=self.encoder_config_path,
            use_cuda=gpu,
        )

    def _check_arguments(
        self,
        speaker: str | None = None,
        language: str | None = None,
        speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]] | None = None,
        emotion: str | None = None,
        **kwargs,
    ) -> None:
        """Check if the arguments are valid for the model."""
        # check for the coqui tts models
        if self.is_multi_lingual and language is None:
            raise ValueError("Model is multi-lingual but no `language` is provided.")
        if not self.is_multi_speaker and speaker is not None:
            raise ValueError("Model is not multi-speaker but `speaker` is provided.")
        if not self.is_multi_lingual and language is not None:
            raise ValueError("Model is not multi-lingual but `language` is provided.")
        if emotion is not None:
            raise ValueError("Emotion can only be used with Coqui Studio models. Which is discontinued.")

    def tts(
        self,
        text: str,
        speaker: str | None = None,
        language: str | None = None,
        speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]] | None = None,
        emotion: str | None = None,
        split_sentences: bool = True,
        **kwargs,
    ):
        """Convert text to speech.

        Args:
            text (str):
                Input text to synthesize.
            speaker (str, optional):
                Speaker name for multi-speaker. You can check whether loaded model is multi-speaker by
                `tts.is_multi_speaker` and list speakers by `tts.speakers`. Defaults to None.
            language (str): Language of the text. If None, the default language of the speaker is used. Language is only
                supported by `XTTS` model.
            speaker_wav (str, optional):
                Path to a reference wav file to use for voice cloning with supporting models like YourTTS.
                Defaults to None.
            emotion (str, optional):
                Emotion to use for 🐸Coqui Studio models. If None, Studio models use "Neutral". Defaults to None.
            split_sentences (bool, optional):
                Split text into sentences, synthesize them separately and concatenate the file audio.
                Setting it False uses more VRAM and possibly hit model specific text length or VRAM limits. Only
                applicable to the 🐸TTS models. Defaults to True.
            **kwargs (optional):
                Additional arguments for the model.
        """
        if self.synthesizer is None:
            msg = "The selected model does not support speech synthesis."
            raise RuntimeError(msg)
        self._check_arguments(speaker=speaker, language=language, speaker_wav=speaker_wav, emotion=emotion, **kwargs)
        wav = self.synthesizer.tts(
            text=text,
            speaker_name=speaker,
            language_name=language,
            speaker_wav=speaker_wav,
            split_sentences=split_sentences,
            **kwargs,
        )
        return wav

    def tts_to_file(
        self,
        text: str,
        speaker: str | None = None,
        language: str | None = None,
        speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]] | None = None,
        emotion: str | None = None,
        pipe_out=None,
        file_path: str = "output.wav",
        split_sentences: bool = True,
        **kwargs,
    ) -> str:
        """Convert text to speech.

        Args:
            text (str):
                Input text to synthesize.
            speaker (str, optional):
                Speaker name for multi-speaker. You can check whether loaded model is multi-speaker by
                `tts.is_multi_speaker` and list speakers by `tts.speakers`. Defaults to None.
            language (str, optional):
                Language code for multi-lingual models. You can check whether loaded model is multi-lingual
                `tts.is_multi_lingual` and list available languages by `tts.languages`. Defaults to None.
            speaker_wav (str, optional):
                Path to a reference wav file to use for voice cloning with supporting models like YourTTS.
                Defaults to None.
            emotion (str, optional):
                Emotion to use for 🐸Coqui Studio models. Defaults to "Neutral".
            pipe_out (BytesIO, optional):
                Flag to stdout the generated TTS wav file for shell pipe.
            file_path (str, optional):
                Output file path. Defaults to "output.wav".
            split_sentences (bool, optional):
                Split text into sentences, synthesize them separately and concatenate the file audio.
                Setting it False uses more VRAM and possibly hit model specific text length or VRAM limits. Only
                applicable to the 🐸TTS models. Defaults to True.
            **kwargs (optional):
                Additional arguments for the model.
        """
        self._check_arguments(speaker=speaker, language=language, speaker_wav=speaker_wav, **kwargs)

        wav = self.tts(
            text=text,
            speaker=speaker,
            language=language,
            speaker_wav=speaker_wav,
            split_sentences=split_sentences,
            **kwargs,
        )
        self.synthesizer.save_wav(wav=wav, path=file_path, pipe_out=pipe_out)
        return file_path

    def voice_conversion(
        self,
        source_wav: str | os.PathLike[Any],
        target_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]] | None = None,
        *,
        speaker: str | None = None,
        voice_dir: str | os.PathLike[Any] | None = None,
        source_speaker: str | None = None,
        **kwargs,
    ):
        """Convert source wav to target speaker.

        Target speaker voices can be cached by assigning a ``speaker`` argument
        for later reuse without ``target_wav``.

        Args:
            source_wav:
                Path to the source wav file.
            target_wav:
                Path(s) to the target wav file(s).
            speaker:
                Custom target speaker ID to cache the cloned voice.
            voice_dir:
                Cache folder for cloned voices.
            source_speaker:
                Source speaker ID. Only needed for embedding-based models like Vits.
            **kwargs:
                Additional arguments for the model.
        """
        if self.voice_converter is not None:
            return self.voice_converter.voice_conversion(
                source_wav=source_wav, target_wav=target_wav, speaker_id=speaker, voice_dir=voice_dir, **kwargs
            )
        if self.synthesizer is not None and hasattr(self.synthesizer.tts_model, "voice_conversion"):
            return self.synthesizer.tts(
                source_wav=source_wav,
                source_speaker_name=source_speaker,
                speaker_wav=target_wav,
                speaker_name=speaker,
                voice_dir=voice_dir,
                **kwargs,
            )
        msg = "The selected model does not support voice conversion."
        raise RuntimeError(msg)

    def voice_conversion_to_file(
        self,
        source_wav: str | os.PathLike[Any],
        target_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]] | None = None,
        *,
        file_path: str = "output.wav",
        speaker: str | None = None,
        voice_dir: str | os.PathLike[Any] | None = None,
        source_speaker: str | None = None,
        pipe_out=None,
        **kwargs,
    ) -> str:
        """Convert source wav to target speaker.

        Target speaker voices can be cached by assigning a ``speaker`` argument
        for later reuse without ``target_wav``.

        Args:
            source_wav:
                Path to the source wav file.
            target_wav:
                Path to the target wav file.
            file_path:
                Output file path. Defaults to "output.wav".
            speaker:
                Custom speaker ID to cache the cloned voice.
            voice_dir:
                Cache folder for cloned voices.
            source_speaker:
                Source speaker ID. Only needed for embedding-based models like Vits.
            pipe_out (BytesIO, optional):
                Flag to stdout the generated TTS wav file for shell pipe.
            **kwargs:
                Additional arguments for the model.
        """
        wav = self.voice_conversion(
            source_wav=source_wav, target_wav=target_wav, speaker=speaker, voice_dir=voice_dir, **kwargs
        )
        if self.voice_converter is not None:
            self.voice_converter.save_wav(wav=wav, path=file_path, pipe_out=pipe_out)
        else:
            self.synthesizer.save_wav(wav=wav, path=file_path, pipe_out=pipe_out)
        return file_path

    def tts_with_vc(
        self,
        text: str,
        *,
        language: str | None = None,
        speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]],
        speaker: str | None = None,
        split_sentences: bool = True,
    ):
        """Convert text to speech with voice conversion.

        It combines tts with voice conversion to fake voice cloning.

        - Convert text to speech with tts.
        - Convert the output wav to target speaker with voice conversion.

        Args:
            text (str):
                Input text to synthesize.
            language (str, optional):
                Language code for multi-lingual models. You can check whether loaded model is multi-lingual
                `tts.is_multi_lingual` and list available languages by `tts.languages`. Defaults to None.
            speaker_wav (str, optional):
                Path to a reference wav file to use for voice cloning with supporting models like YourTTS.
                Defaults to None.
            speaker (str, optional):
                Speaker name for multi-speaker. You can check whether loaded model is multi-speaker by
                `tts.is_multi_speaker` and list speakers by `tts.speakers`. Defaults to None.
            split_sentences (bool, optional):
                Split text into sentences, synthesize them separately and concatenate the file audio.
                Setting it False uses more VRAM and possibly hit model specific text length or VRAM limits. Only
                applicable to the 🐸TTS models. Defaults to True.
        """
        if self.synthesizer.tts_model.config.supports_cloning:
            warnings.warn(
                "This TTS model directly supports voice cloning, for better quality call it with "
                "tts/tts_to_file(..., speaker_wav=...) instead."
            )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            # Lazy code... save it to a temp file to resample it while reading it for VC
            self.tts_to_file(
                text=text, speaker=speaker, language=language, file_path=fp.name, split_sentences=split_sentences
            )
        if self.voice_converter is None:
            self.load_vc_model_by_name("voice_conversion_models/multilingual/vctk/freevc24")
        wav = self.voice_converter.voice_conversion(source_wav=fp.name, target_wav=speaker_wav)
        return wav

    def tts_with_vc_to_file(
        self,
        text: str,
        *,
        language: str | None = None,
        speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]],
        file_path: str = "output.wav",
        speaker: str | None = None,
        split_sentences: bool = True,
        pipe_out=None,
    ) -> str:
        """Convert text to speech with voice conversion and save to file.

        Check `tts_with_vc` for more details.

        Args:
            text (str):
                Input text to synthesize.
            language (str, optional):
                Language code for multi-lingual models. You can check whether loaded model is multi-lingual
                `tts.is_multi_lingual` and list available languages by `tts.languages`. Defaults to None.
            speaker_wav (str, optional):
                Path to a reference wav file to use for voice cloning with supporting models like YourTTS.
                Defaults to None.
            file_path (str, optional):
                Output file path. Defaults to "output.wav".
            speaker (str, optional):
                Speaker name for multi-speaker. You can check whether loaded model is multi-speaker by
                `tts.is_multi_speaker` and list speakers by `tts.speakers`. Defaults to None.
            split_sentences (bool, optional):
                Split text into sentences, synthesize them separately and concatenate the file audio.
                Setting it False uses more VRAM and possibly hit model specific text length or VRAM limits. Only
                applicable to the 🐸TTS models. Defaults to True.
            pipe_out (BytesIO, optional):
                Flag to stdout the generated TTS wav file for shell pipe.
        """
        wav = self.tts_with_vc(
            text=text, language=language, speaker_wav=speaker_wav, speaker=speaker, split_sentences=split_sentences
        )
        self.voice_converter.save_wav(wav=wav, path=file_path, pipe_out=pipe_out)
        return file_path
