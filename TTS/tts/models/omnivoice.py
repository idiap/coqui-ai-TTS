"""Thin Coqui TTS adapter for the optional OmniVoice package."""

import os
from typing import Any

import torch
from coqpit import Coqpit

from TTS.tts.models.base_tts import BaseTTS
from TTS.utils.generic_utils import warn_synthesize_config_deprecated, warn_synthesize_speaker_id_deprecated


class Omnivoice(BaseTTS):
    """Expose k2-fsa OmniVoice through Coqui's synthesis and voice-caching APIs."""

    def __init__(self, config: Coqpit) -> None:
        self.tokenizer = None  # OmniVoice provides its own tokenizer.
        super().__init__(config)
        self.model = None

    def load_checkpoint(
        self,
        config: Coqpit,
        checkpoint_dir: str | os.PathLike[Any],
        *,
        eval: bool = False,
        **kwargs: Any,
    ) -> None:
        try:
            from omnivoice import OmniVoice as OmniVoiceModel
        except ImportError as e:
            raise ImportError(
                "OmniVoice support requires its optional dependency: pip install 'coqui-tts[omnivoice]'"
            ) from e

        self.model = OmniVoiceModel.from_pretrained(str(checkpoint_dir), dtype=torch.float32)
        self.config.audio.sample_rate = self.model.sampling_rate
        self.config.audio.output_sample_rate = self.model.sampling_rate
        languages = sorted(self.model.supported_language_ids())
        self.language_manager.name_to_id = {language: index for index, language in enumerate(languages)}
        self.config.languages = languages
        if eval:
            self.eval()

    def _require_model(self):
        if self.model is None:
            raise RuntimeError("OmniVoice checkpoint has not been loaded.")
        return self.model

    def _apply(self, fn, recurse=True):
        """Move the generator while keeping the large audio tokenizer on CPU.

        OmniVoice requires this on Apple Silicon because its tokenizer has an
        output channel count unsupported by MPS. It is also a lower-memory
        default on CUDA devices.
        """
        audio_tokenizer = getattr(self.model, "audio_tokenizer", None)
        if audio_tokenizer is not None:
            self.model.audio_tokenizer = None
        try:
            return super()._apply(fn, recurse=recurse)
        finally:
            if audio_tokenizer is not None:
                self.model.audio_tokenizer = audio_tokenizer

    def _clone_voice(
        self,
        speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]],
        **generate_kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        model = self._require_model()
        if isinstance(speaker_wav, list):
            if len(speaker_wav) != 1:
                raise ValueError("OmniVoice accepts exactly one reference audio file.")
            speaker_wav = speaker_wav[0]
        prompt = model.create_voice_clone_prompt(
            str(speaker_wav),
            ref_text=generate_kwargs.get("ref_text"),
            preprocess_prompt=generate_kwargs.get("preprocess_prompt", True),
        )
        voice = {
            "ref_audio_tokens": prompt.ref_audio_tokens,
            "ref_text": prompt.ref_text,
            "ref_rms": prompt.ref_rms,
        }
        return voice, {"name": self.config.model}

    def synthesize(
        self,
        text: str,
        config: Coqpit | None = None,
        *,
        speaker: str | None = None,
        speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]] | None = None,
        voice_dir: str | os.PathLike[Any] | None = None,
        language: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if config is not None:
            warn_synthesize_config_deprecated()
        if (speaker_id := kwargs.pop("speaker_id", None)) is not None:
            speaker = speaker_id
            warn_synthesize_speaker_id_deprecated()

        model = self._require_model()
        for key in ("use_griffin_lim", "do_trim_silence", "extra_aux_input"):
            kwargs.pop(key, None)

        voice_clone_prompt = None
        if speaker_wav is not None or speaker is not None:
            from omnivoice.models.omnivoice import VoiceClonePrompt

            voice = self.clone_voice(
                speaker_wav,
                speaker,
                voice_dir,
                ref_text=kwargs.pop("ref_text", None),
                preprocess_prompt=kwargs.get("preprocess_prompt", True),
            )
            voice_clone_prompt = VoiceClonePrompt(
                ref_audio_tokens=voice["ref_audio_tokens"],
                ref_text=voice["ref_text"],
                ref_rms=voice["ref_rms"],
            )

        audio = model.generate(
            text=text,
            language=language,
            voice_clone_prompt=voice_clone_prompt,
            **kwargs,
        )[0]
        return {"wav": audio, "text_inputs": text}

    def forward(self): ...

    def inference(self, input: torch.Tensor, aux_input: dict[str, Any] = {}) -> dict[str, Any]: ...
