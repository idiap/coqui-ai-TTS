import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pysbd
import torch
from torch import nn

from TTS.config import load_config
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models import setup_model as setup_tts_model
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.models.vits import Vits
from TTS.utils.audio import AudioProcessor
from TTS.utils.audio.numpy_transforms import save_wav
from TTS.utils.generic_utils import optional_to_str
from TTS.vc.configs.openvoice_config import OpenVoiceConfig
from TTS.vc.models import setup_model as setup_vc_model
from TTS.vc.models.base_vc import BaseVC
from TTS.vc.models.openvoice import OpenVoice
from TTS.vocoder.models import setup_model as setup_vocoder_model
from TTS.vocoder.models.base_vocoder import BaseVocoder
from TTS.vocoder.utils.generic_utils import interpolate_vocoder_input

logger = logging.getLogger(__name__)


class Synthesizer(nn.Module):
    def __init__(
        self,
        *,
        tts_checkpoint: str | os.PathLike[Any] | None = None,
        tts_config_path: str | os.PathLike[Any] | None = None,
        tts_speakers_file: str | os.PathLike[Any] | None = None,
        tts_languages_file: str | os.PathLike[Any] | None = None,
        vocoder_checkpoint: str | os.PathLike[Any] | None = None,
        vocoder_config: str | os.PathLike[Any] | None = None,
        encoder_checkpoint: str | os.PathLike[Any] | None = None,
        encoder_config: str | os.PathLike[Any] | None = None,
        vc_checkpoint: str | os.PathLike[Any] | None = None,
        vc_config: str | os.PathLike[Any] | None = None,
        model_dir: str | os.PathLike[Any] | None = None,
        voice_dir: str | os.PathLike[Any] | None = None,
        use_cuda: bool = False,
    ) -> None:
        """General 🐸 TTS interface for inference. It takes a tts and a vocoder
        model and synthesize speech from the provided text.

        The text is divided into a list of sentences using `pysbd` and synthesize
        speech on each sentence separately.

        If you have certain special characters in your text, you need to handle
        them before providing the text to Synthesizer.

        TODO: set the segmenter based on the source language

        Args:
            tts_checkpoint (str, optional): path to the tts model file.
            tts_config_path (str, optional): path to the tts config file.
            vocoder_checkpoint (str, optional): path to the vocoder model file. Defaults to None.
            vocoder_config (str, optional): path to the vocoder config file. Defaults to None.
            encoder_checkpoint (str, optional): path to the speaker encoder model file. Defaults to `""`,
            encoder_config (str, optional): path to the speaker encoder config file. Defaults to `""`,
            vc_checkpoint (str, optional): path to the voice conversion model file. Defaults to `""`,
            vc_config (str, optional): path to the voice conversion config file. Defaults to `""`,
            use_cuda (bool, optional): enable/disable cuda. Defaults to False.
        """
        super().__init__()
        self.tts_checkpoint = optional_to_str(tts_checkpoint)
        self.tts_config_path = optional_to_str(tts_config_path)
        self.tts_speakers_file = optional_to_str(tts_speakers_file)
        self.tts_languages_file = optional_to_str(tts_languages_file)
        self.vocoder_checkpoint = optional_to_str(vocoder_checkpoint)
        self.vocoder_config = optional_to_str(vocoder_config)
        self.encoder_checkpoint = optional_to_str(encoder_checkpoint)
        self.encoder_config = optional_to_str(encoder_config)
        self.vc_checkpoint = optional_to_str(vc_checkpoint)
        self.vc_config = optional_to_str(vc_config)
        model_dir = optional_to_str(model_dir)
        self.use_cuda = use_cuda

        self.tts_model: BaseTTS | None = None
        self.vocoder_model: BaseVocoder | None = None
        self.vc_model: BaseVC | None = None
        self.seg = self._get_segmenter("en")
        self.use_cuda = use_cuda
        if self.use_cuda:
            assert torch.cuda.is_available(), "CUDA is not availabe on this machine."

        self.checkpoint_dir = None
        if tts_checkpoint:
            self._load_tts(self.tts_checkpoint, self.tts_config_path, use_cuda)
            self.checkpoint_dir = Path(self.tts_checkpoint)

        if vc_checkpoint and model_dir == "":
            self._load_vc(self.vc_checkpoint, self.vc_config, use_cuda)
            self.checkpoint_dir = Path(self.vc_checkpoint)

        if vocoder_checkpoint:
            self._load_vocoder(self.vocoder_checkpoint, self.vocoder_config, use_cuda)

        if model_dir:
            path = Path(model_dir)
            self.checkpoint_dir = path if path.is_dir() else path.parent
            if "fairseq" in model_dir:
                self._load_fairseq_from_dir(model_dir, use_cuda)
            elif "openvoice" in model_dir:
                self._load_openvoice_from_dir(Path(model_dir), use_cuda)
            else:
                self._load_tts_from_dir(model_dir, use_cuda)

        if self.checkpoint_dir is None:
            msg = "Need to initialize a TTS or VC model via tts_checkpoint/vc_checkpoint/model_dir"
            raise RuntimeError(msg)

        self.voice_dir = Path(voice_dir) if voice_dir is not None else self.checkpoint_dir / "voices"

    @staticmethod
    def _get_segmenter(lang: str):
        """get the sentence segmenter for the given language.

        Args:
            lang (str): target language code.

        Returns:
            [type]: [description]
        """
        return pysbd.Segmenter(language=lang, clean=True)

    def _load_vc(self, vc_checkpoint: str, vc_config_path: str, use_cuda: bool) -> None:
        """Load the voice conversion model.

        1. Load the model config.
        2. Init the model from the config.
        3. Load the model weights.
        4. Move the model to the GPU if CUDA is enabled.

        Args:
            vc_checkpoint (str): path to the model checkpoint.
            tts_config_path (str): path to the model config file.
            use_cuda (bool): enable/disable CUDA use.
        """
        # pylint: disable=global-statement
        self.vc_config = load_config(vc_config_path)
        self.output_sample_rate = self.vc_config.audio.get(
            "output_sample_rate", self.vc_config.audio.get("sample_rate", None)
        )
        self.vc_model = setup_vc_model(config=self.vc_config)
        self.vc_model.load_checkpoint(self.vc_config, vc_checkpoint)
        if use_cuda:
            self.vc_model.cuda()

    def _load_fairseq_from_dir(self, model_dir: str, use_cuda: bool) -> None:
        """Load the fairseq model from a directory.

        We assume it is VITS and the model knows how to load itself from the directory and there is a config.json file in the directory.
        """
        self.tts_config = VitsConfig()
        self.tts_model = Vits.init_from_config(self.tts_config)
        self.tts_model.load_fairseq_checkpoint(self.tts_config, checkpoint_dir=model_dir, eval=True)
        self.tts_config = self.tts_model.config
        self.output_sample_rate = self.tts_config.audio["sample_rate"]
        if use_cuda:
            self.tts_model.cuda()

    def _load_openvoice_from_dir(self, checkpoint: Path, use_cuda: bool) -> None:
        """Load the OpenVoice model from a directory.

        We assume the model knows how to load itself from the directory and
        there is a config.json file in the directory.
        """
        self.vc_config = OpenVoiceConfig()
        self.vc_model = OpenVoice.init_from_config(self.vc_config)
        self.vc_model.load_checkpoint(self.vc_config, checkpoint, eval=True)
        self.vc_config = self.vc_model.config
        self.output_sample_rate = self.vc_config.audio["output_sample_rate"]
        if use_cuda:
            self.vc_model.cuda()

    def _load_tts_from_dir(self, model_dir: str, use_cuda: bool) -> None:
        """Load the TTS model from a directory.

        We assume the model knows how to load itself from the directory and there is a config.json file in the directory.
        """
        config = load_config(os.path.join(model_dir, "config.json"))
        self.tts_config = config
        self.output_sample_rate = self.tts_config.audio["output_sample_rate"]
        self.tts_model = setup_tts_model(config)
        self.tts_model.load_checkpoint(config, checkpoint_dir=model_dir, eval=True)
        if use_cuda:
            self.tts_model.cuda()

    def _load_tts(self, tts_checkpoint: str, tts_config_path: str, use_cuda: bool) -> None:
        """Load the TTS model.

        1. Load the model config.
        2. Init the model from the config.
        3. Load the model weights.
        4. Move the model to the GPU if CUDA is enabled.
        5. Init the speaker manager in the model.

        Args:
            tts_checkpoint (str): path to the model checkpoint.
            tts_config_path (str): path to the model config file.
            use_cuda (bool): enable/disable CUDA use.
        """
        # pylint: disable=global-statement
        self.tts_config = load_config(tts_config_path)
        self.output_sample_rate = self.tts_config.audio["sample_rate"]
        if self.tts_config["use_phonemes"] and self.tts_config["phonemizer"] is None:
            raise ValueError("Phonemizer is not defined in the TTS config.")

        self.tts_model = setup_tts_model(config=self.tts_config)

        if not self.encoder_checkpoint:
            self._set_speaker_encoder_paths_from_tts_config()

        self.tts_model.load_checkpoint(self.tts_config, tts_checkpoint, eval=True)
        if use_cuda:
            self.tts_model.cuda()

        if self.encoder_checkpoint and hasattr(self.tts_model, "speaker_manager"):
            self.tts_model.speaker_manager.init_encoder(self.encoder_checkpoint, self.encoder_config, use_cuda)

    def _set_speaker_encoder_paths_from_tts_config(self):
        """Set the encoder paths from the tts model config for models with speaker encoders."""
        if self.tts_config.model_args is not None and hasattr(
            self.tts_config.model_args, "speaker_encoder_config_path"
        ):
            self.encoder_checkpoint = self.tts_config.model_args.speaker_encoder_model_path
            self.encoder_config = self.tts_config.model_args.speaker_encoder_config_path

    def _load_vocoder(self, model_file: str, model_config: str, use_cuda: bool) -> None:
        """Load the vocoder model.

        1. Load the vocoder config.
        2. Init the AudioProcessor for the vocoder.
        3. Init the vocoder model from the config.
        4. Move the model to the GPU if CUDA is enabled.

        Args:
            model_file (str): path to the model checkpoint.
            model_config (str): path to the model config file.
            use_cuda (bool): enable/disable CUDA use.
        """
        self.vocoder_config = load_config(model_config)
        self.output_sample_rate = self.vocoder_config.audio["sample_rate"]
        self.vocoder_ap = AudioProcessor(**self.vocoder_config.audio)
        self.vocoder_model = setup_vocoder_model(self.vocoder_config)
        self.vocoder_model.load_checkpoint(self.vocoder_config, model_file, eval=True)
        if use_cuda:
            self.vocoder_model.cuda()

    def split_into_sentences(self, text) -> list[str]:
        """Split give text into sentences.

        Args:
            text (str): input text in string format.

        Returns:
            List[str]: list of sentences.
        """
        return self.seg.segment(text)

    def save_wav(self, wav: list[int] | torch.Tensor | np.ndarray, path: str, pipe_out=None) -> None:
        """Save the waveform as a file.

        Args:
            wav (List[int]): waveform as a list of values.
            path (str): output path to save the waveform.
            pipe_out (BytesIO, optional): Flag to stdout the generated TTS wav file for shell pipe.
        """
        # if tensor convert to numpy
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()
        if isinstance(wav, list):
            wav = np.array(wav)
        save_wav(wav=wav, path=path, sample_rate=self.output_sample_rate, pipe_out=pipe_out)

    def voice_conversion(
        self,
        source_wav: str,
        target_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]] | None = None,
        *,
        speaker_id: str | None = None,
        voice_dir: str | os.PathLike[Any] | None = None,
        **kwargs,
    ) -> list[int]:
        """Run a voice conversion model."""
        start_time = time.time()
        if self.vc_model is None:
            msg = "Voice conversion model not loaded"
            raise RuntimeError(msg)
        if target_wav is None and speaker_id is None:
            msg = "Need to specify at least one of `target_wav` and `speaker_id`"
            raise RuntimeError(msg)

        voice_dir = Path(voice_dir) if voice_dir is not None else self.voice_dir
        if target_wav is not None and not isinstance(target_wav, list):
            target_wav = [target_wav]
        output = self.vc_model.voice_conversion(
            source_wav, target_wav, speaker_id=speaker_id, voice_dir=voice_dir, **kwargs
        )
        if self.vocoder_model is not None:
            output = self.vocoder_model.inference(output)

        output = output.squeeze()
        process_time = time.time() - start_time
        audio_time = len(output) / self.output_sample_rate
        logger.info("Processing time: %.3f", process_time)
        logger.info("Real-time factor: %.3f", process_time / audio_time)
        return output

    def tts(
        self,
        text: str = "",
        speaker_name: str | None = "",
        language_name: str = "",
        speaker_wav=None,
        style_wav=None,
        style_text=None,
        source_wav=None,
        source_speaker_name=None,
        split_sentences: bool = True,
        **kwargs,
    ) -> list[int]:
        """🐸 TTS magic. Run all the models and generate speech.

        Args:
            text (str): input text.
            speaker_name (str, optional): speaker id for multi-speaker models. Defaults to "".
            language_name (str, optional): language id for multi-language models. Defaults to "".
            speaker_wav (Union[str, List[str]], optional): path to the speaker wav for voice cloning. Defaults to None.
            style_wav ([type], optional): style waveform for GST. Defaults to None.
            style_text ([type], optional): transcription of style_wav for Capacitron. Defaults to None.
            source_wav ([type], optional): source waveform for voice conversion. Defaults to None.
            source_speaker_name ([type], optional): speaker id of source waveform. Defaults to None.
            split_sentences (bool, optional): split the input text into sentences. Defaults to True.
            **kwargs: additional arguments to pass to the TTS model.
        Returns:
            List[int]: [description]
        """
        if self.tts_model is None:
            msg = "Text-to-speech model not loaded"
            raise RuntimeError(msg)
        start_time = time.time()
        wavs = []

        if not text and not speaker_wav and not speaker_name:
            msg = (
                "You need to define at least either `text` (for synthesis) or a "
                "`speaker_wav` or cached `speaker` name (for voice conversion) to use the Coqui TTS API."
            )
            raise ValueError(msg)

        if text:
            sens = [text]
            if split_sentences:
                sens = self.split_into_sentences(text)
                logger.info("Text split into sentences.")
            logger.info("Input: %s", sens)

        voice_dir = Path(d) if (d := kwargs.pop("voice_dir", None)) is not None else self.voice_dir
        vocoder_device = "cpu"
        use_gl = self.vocoder_model is None
        if not use_gl:
            vocoder_device = next(self.vocoder_model.parameters()).device
        if self.use_cuda:
            vocoder_device = "cuda"

        if not source_wav:  # not voice conversion
            for sen in sens:
                outputs = self.tts_model.synthesize(
                    text=sen,
                    speaker=speaker_name,
                    voice_dir=voice_dir,
                    speaker_wav=speaker_wav,
                    language=language_name,
                    use_griffin_lim=use_gl,
                    **kwargs,
                )
                waveform = outputs["wav"]
                if not use_gl:
                    mel_postnet_spec = outputs["outputs"]["model_outputs"][0].detach().cpu().numpy()
                    # denormalize tts output based on tts audio config
                    mel_postnet_spec = self.tts_model.ap.denormalize(mel_postnet_spec.T).T
                    # renormalize spectrogram based on vocoder config
                    vocoder_input = self.vocoder_ap.normalize(mel_postnet_spec.T)
                    # compute scale factor for possible sample rate mismatch
                    scale_factor = [
                        1,
                        self.vocoder_config["audio"]["sample_rate"] / self.tts_model.ap.sample_rate,
                    ]
                    if scale_factor[1] != 1:
                        logger.info("Interpolating TTS model output.")
                        vocoder_input = interpolate_vocoder_input(scale_factor, vocoder_input)
                    else:
                        vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)  # pylint: disable=not-callable
                    # run vocoder model
                    # [1, T, C]
                    waveform = self.vocoder_model.inference(vocoder_input.to(vocoder_device))
                if isinstance(waveform, torch.Tensor) and waveform.device != torch.device("cpu") and not use_gl:
                    waveform = waveform.cpu()
                if not use_gl:
                    waveform = waveform.numpy()
                waveform = waveform.squeeze()

                # trim silence
                if "do_trim_silence" in self.tts_config.audio and self.tts_config.audio["do_trim_silence"]:
                    waveform = waveform[: self.tts_model.ap.find_endpoint(waveform)]

                wavs += list(waveform)
                wavs += [0] * 10000
        else:
            outputs = self.tts_model.voice_conversion(
                source_wav, speaker_wav, source_speaker=source_speaker_name, speaker=speaker_name, voice_dir=voice_dir
            )
            waveform = outputs
            if not use_gl:
                mel_postnet_spec = outputs[0].detach().cpu().numpy()
                # denormalize tts output based on tts audio config
                mel_postnet_spec = self.tts_model.ap.denormalize(mel_postnet_spec.T).T
                # renormalize spectrogram based on vocoder config
                vocoder_input = self.vocoder_ap.normalize(mel_postnet_spec.T)
                # compute scale factor for possible sample rate mismatch
                scale_factor = [
                    1,
                    self.vocoder_config["audio"]["sample_rate"] / self.tts_model.ap.sample_rate,
                ]
                if scale_factor[1] != 1:
                    logger.info("Interpolating TTS model output.")
                    vocoder_input = interpolate_vocoder_input(scale_factor, vocoder_input)
                else:
                    vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)  # pylint: disable=not-callable
                # run vocoder model
                # [1, T, C]
                waveform = self.vocoder_model.inference(vocoder_input.to(vocoder_device))
            if isinstance(waveform, torch.Tensor) and waveform.device != torch.device("cpu"):
                waveform = waveform.cpu()
            if not use_gl:
                waveform = waveform.numpy()
            wavs = waveform.squeeze()

        # compute stats
        process_time = time.time() - start_time
        audio_time = len(wavs) / self.tts_config.audio["sample_rate"]
        logger.info("Processing time: %.3f", process_time)
        logger.info("Real-time factor: %.3f", process_time / audio_time)
        return wavs
