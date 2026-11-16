"""
Kokoro implementation.

Kokoro is based on StyleTTS 2 without the style encoders and diffusion:
speakers are represented by precomputed style vectors ("voices").

Original code: https://github.com/hexgrad/kokoro
Pre-trained models: https://huggingface.co/hexgrad/Kokoro-82M
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import torch

from TTS.tts.configs.kokoro_config import KokoroConfig
from TTS.tts.configs.shared_configs import BaseTTSConfig, CharactersConfig
from TTS.tts.configs.styletts2_config import PLBertConfig, StyleTTS2DecoderConfig
from TTS.tts.layers.styletts2.decoder import StyleTTSDecoder
from TTS.tts.layers.styletts2.duration_predictor import DifferentiableDurationPredictor
from TTS.tts.layers.styletts2.prosody_predictor import ProsodyPredictor
from TTS.tts.layers.styletts2.text_encoders import AcousticTextEncoder, ProsodicTextEncoder
from TTS.tts.models.styletts2 import StyleTTS2
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.characters import BaseCharacters
from TTS.tts.utils.text.phonemizers.kokoro_phonemizer import KokoroPhonemizer
from TTS.utils.generic_utils import warn_synthesize_config_deprecated, warn_synthesize_speaker_id_deprecated

logger = logging.getLogger(__name__)

# The first letter of Kokoro voice names indicates their language
_VOICE_PREFIX_TO_LANGUAGE = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "p": "pt-br",
    "j": "ja",
    "z": "zh",
}


class Kokoro(StyleTTS2):
    """Kokoro model, only supports inference with pre-trained models.

    Speakers are selected by voice name, e.g. ``af_heart``. Several voices can be
    mixed by separating them with commas, e.g. ``af_heart,af_bella``.
    """

    def __init__(self, config: KokoroConfig):
        super().__init__(config)
        del self.to_mel  # only needed for voice cloning in StyleTTS 2
        self.voices: dict[str, torch.Tensor] = {}

    def _init_model_components(self) -> None:
        self.acoustic_text_encoder = AcousticTextEncoder(
            num_chars=self.args.num_chars,
            hidden_channels=self.args.hidden_channels,
            num_layers=self.args.acoustic_text_encoder_layers,
        )
        self.prosodic_text_encoder = ProsodicTextEncoder(
            plbert_config=self.args.plbert_config,
            hidden_channels=self.args.hidden_channels,
        )
        self.duration_predictor = DifferentiableDurationPredictor(
            hidden_channels=self.args.hidden_channels,
            num_layers=self.args.duration_predictor_layers,
            max_duration=self.args.max_duration,
            style_dim=self.args.style_dim,
        )
        self.prosody_predictor = ProsodyPredictor(
            hidden_dim=self.args.hidden_channels, style_dim=self.args.style_dim, dropout=0.1
        )
        self.decoder = StyleTTSDecoder(self.args.decoder_config, self.args.hidden_channels, self.args.style_dim)
        # Kokoro uses torch's Hann window, which differs from scipy's (StyleTTS 2)
        # by ~1e-7. This is enough to flip the phase of near-zero STFT bins of the
        # harmonic source, which the generator takes as input.
        self.decoder.generator.stft.window = torch.hann_window(self.args.decoder_config.gen_istft_n_fft, periodic=True)

    def forward(self, *args, **kwargs) -> dict[str, Any]:
        msg = "Training Kokoro is not supported."
        raise NotImplementedError(msg)

    @torch.inference_mode()
    def inference(
        self,
        tokens: torch.Tensor,
        aux_input: dict[str, Any] | None = None,
        *,
        speed: float = 1.0,
    ) -> dict[str, Any]:
        """Inference forward pass.

        Args:
            tokens: Input token sequences [1, T_text], including padding at start and end.
            aux_input: Must contain the ``style`` vector [1, 2 * style_dim].
            speed: Speaking rate, higher is faster.

        Returns:
            Dictionary with the ``model_outputs`` waveform [1, 1, T_audio], the
            ``alignments`` and the predicted ``durations`` per token.
        """
        if aux_input is None or aux_input.get("style") is None:
            msg = "Kokoro requires a `style` vector in `aux_input`."
            raise ValueError(msg)
        style = aux_input["style"]
        acoustic_style = style[:, : self.args.style_dim]
        prosodic_style = style[:, self.args.style_dim :]

        acoustic_text_features = self.acoustic_text_encoder.inference(tokens)
        prosodic_text_features, _ = self.prosodic_text_encoder.inference(tokens)

        duration_outputs = self.duration_predictor(prosodic_text_features, prosodic_style, training=False, speed=speed)
        alignments = duration_outputs["alignments"].unsqueeze(0)
        en = duration_outputs["encoder_output"].transpose(-1, -2) @ alignments
        aligned_text_features = acoustic_text_features @ alignments

        f0, energy = self.prosody_predictor(en, prosodic_style)
        model_outputs = self.decoder(aligned_text_features, acoustic_style, f0, energy)
        return {
            "model_outputs": model_outputs,
            "alignments": duration_outputs["alignments"],
            "durations": duration_outputs["predicted_durations"],
        }

    def _clone_voice(self, speaker_wav: Any, **kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
        msg = "Kokoro does not support voice cloning, choose one of the built-in voices instead."
        raise NotImplementedError(msg)

    def get_voice(self, speaker: str) -> torch.Tensor:
        """Return the style vectors of the given voice, averaging comma-separated voices.

        Returns:
            Tensor [max_phonemes, 1, 2 * style_dim] with one style per input length.
        """
        names = [name.strip() for name in speaker.split(",")]
        for name in names:
            if name not in self.voices:
                msg = f"Voice `{name}` not found, choose one of: {sorted(self.voices)}"
                raise ValueError(msg)
        if len(names) == 1:
            return self.voices[names[0]]
        return torch.stack([self.voices[name] for name in names]).mean(dim=0)

    def load_voices(self, voice_dir: str | os.PathLike[Any]) -> None:
        """Load all voices (``*.pt`` files) from the given directory."""
        for voice_file in sorted(Path(voice_dir).glob("*.pt")):
            self.voices[voice_file.stem] = torch.load(voice_file, map_location="cpu", weights_only=True)
        if len(self.voices) == 0:
            msg = f"No Kokoro voices (*.pt files) found in {voice_dir}"
            raise FileNotFoundError(msg)
        self.speaker_manager = SpeakerManager()
        self.speaker_manager.name_to_id = {name: i for i, name in enumerate(self.voices)}

    def synthesize(
        self,
        text: str,
        config: BaseTTSConfig | None = None,
        *,
        speaker: str | None = None,
        speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]] | None = None,
        voice_dir: str | os.PathLike[Any] | None = None,
        language: str | None = None,
        speed: float = 1.0,
        **kwargs,
    ) -> dict[str, Any]:
        """Synthesize speech for the given text.

        Args:
            text: Input text to convert to speech.
            config: DEPRECATED. Not used.
            speaker: Voice name, or several comma-separated voice names to mix them.
            speaker_wav: Not supported.
            voice_dir: Not used.
            language: Language name. Defaults to the language of the voice.
            speed: Speaking rate, higher is faster.
        """
        if config is not None:
            warn_synthesize_config_deprecated()
        if (speaker_id := kwargs.pop("speaker_id", None)) is not None:
            speaker = speaker_id
            warn_synthesize_speaker_id_deprecated()
        if speaker_wav is not None:
            msg = "Kokoro does not support voice cloning, choose one of the built-in voices with `speaker`."
            raise ValueError(msg)
        if not speaker:
            msg = f"Kokoro requires a voice, pass one of these as `speaker`: {sorted(self.voices)}"
            raise ValueError(msg)
        voice = self.get_voice(speaker)
        if language is None:
            language = _VOICE_PREFIX_TO_LANGUAGE.get(speaker[0], "en-us")

        # The tokenizer adds a pad token at start and end
        token_ids = self.tokenizer.text_to_ids(text, language=language)
        max_length = self.args.plbert_config.max_position_embeddings
        if len(token_ids) > max_length:
            logger.warning("Input of %d phonemes is too long, truncating to %d", len(token_ids) - 2, max_length - 2)
            token_ids = [*token_ids[: max_length - 1], token_ids[-1]]
        num_phonemes = len(token_ids) - 2
        style = voice[max(num_phonemes, 1) - 1].to(self.device)

        text_inputs = torch.as_tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        outputs = self.inference(text_inputs, aux_input={"style": style}, speed=speed)
        wav = outputs["model_outputs"].squeeze().cpu().numpy()
        return {
            "wav": wav,
            "alignments": outputs["alignments"],
            "text_inputs": text_inputs,
            "outputs": outputs,
        }

    @classmethod
    def from_original(cls, checkpoint_dir: str | os.PathLike[Any]) -> "Kokoro":
        """Load a checkpoint in the original Kokoro format.

        Args:
            checkpoint_dir: Directory containing the original ``config.json``, the
                ``.pth`` checkpoint and a ``voices`` folder, as in
                https://huggingface.co/hexgrad/Kokoro-82M

        Returns:
            Kokoro model in eval mode.
        """
        checkpoint_dir = Path(checkpoint_dir)
        with open(checkpoint_dir / "config.json", encoding="utf-8") as f:
            orig_config = json.load(f)
        checkpoint_files = list(checkpoint_dir.glob("*.pth"))
        if len(checkpoint_files) != 1:
            msg = f"Expected exactly one .pth checkpoint file in {checkpoint_dir}, found {len(checkpoint_files)}"
            raise FileNotFoundError(msg)
        logger.info("Loading checkpoint from %s", checkpoint_files[0])
        orig_state_dict = torch.load(checkpoint_files[0], map_location="cpu", weights_only=True)

        model = cls(cls._convert_original_config(orig_config))
        state_dict = cls._convert_state_dict(orig_state_dict)
        model.load_state_dict(state_dict, strict=True)
        model.load_voices(checkpoint_dir / "voices")
        model.eval()
        return model

    @staticmethod
    def _convert_original_config(orig_config: dict[str, Any]) -> KokoroConfig:
        """Convert the original Kokoro ``config.json`` to a Coqui config."""
        config = KokoroConfig()
        args = config.model_args
        args.num_chars = orig_config["n_token"]
        args.hidden_channels = orig_config["hidden_dim"]
        args.style_dim = orig_config["style_dim"]
        args.max_duration = orig_config["max_dur"]
        args.acoustic_text_encoder_layers = orig_config["n_layer"]
        args.duration_predictor_layers = orig_config["n_layer"]
        args.plbert_config = PLBertConfig(vocab_size=orig_config["n_token"], **orig_config["plbert"])
        args.decoder_config = StyleTTS2DecoderConfig(generator_type="istftnet", **orig_config["istftnet"])
        config.audio.sample_rate = 24000
        config.audio.num_mels = orig_config["n_mels"]

        config.use_phonemes = True
        config.phonemizer = KokoroPhonemizer.name()
        config.phoneme_language = "en-us"
        config.languages = list(KokoroPhonemizer.supported_languages())
        config.text_cleaner = None
        config.enable_eos_bos_chars = True
        config.add_blank = False
        config.characters = _create_kokoro_character_config(orig_config["vocab"], orig_config["n_token"])
        return config


class KokoroCharacters(BaseCharacters):
    """Character set with the vocabulary of Kokoro models.

    The original vocabulary maps phonemes to explicit IDs with gaps, so it is
    stored as a list in ``vocab_dict`` with empty strings for unused IDs.
    """

    def __init__(self, vocab: list[str], pad: str = "$") -> None:
        self._vocab_list = list(vocab)
        super().__init__(pad=pad, eos=pad, bos=pad, blank=None, is_unique=False, is_sorted=False)

    def _create_vocab(self) -> None:
        self.vocab = self._vocab_list

    @staticmethod
    def init_from_config(config: BaseTTSConfig) -> tuple["KokoroCharacters", BaseTTSConfig]:
        if config.characters is None or not config.characters.vocab_dict:
            msg = "KokoroCharacters requires `characters.vocab_dict` in the config."
            raise ValueError(msg)
        return KokoroCharacters(config.characters.vocab_dict, config.characters.pad), config

    def to_config(self) -> CharactersConfig:
        return _create_kokoro_character_config(
            {char: i for i, char in enumerate(self.vocab) if char and char != self.pad}, len(self.vocab), self.pad
        )


def _create_kokoro_character_config(vocab: dict[str, int], num_chars: int, pad: str = "$") -> CharactersConfig:
    """Create a CharactersConfig from the original Kokoro vocabulary (ID 0 is the pad token)."""
    vocab_list = [""] * num_chars
    vocab_list[0] = pad
    for char, idx in vocab.items():
        vocab_list[idx] = char
    return CharactersConfig(
        characters_class="TTS.tts.models.kokoro.KokoroCharacters",
        vocab_dict=vocab_list,
        pad=pad,
        eos=pad,
        bos=pad,
        blank=None,
        is_unique=False,
        is_sorted=False,
    )
