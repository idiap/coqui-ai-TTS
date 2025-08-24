import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from coqpit import Coqpit
from encodec import EncodecModel
from encodec.utils import convert_audio
from transformers import BertTokenizer

from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.tts.layers.bark.hubert.hubert_manager import HubertManager
from TTS.tts.layers.bark.hubert.kmeans_hubert import CustomHubert
from TTS.tts.layers.bark.hubert.tokenizer import HubertTokenizer
from TTS.tts.layers.bark.inference_funcs import (
    codec_decode,
    generate_coarse,
    generate_fine,
    generate_text_semantic,
)
from TTS.tts.layers.bark.load_model import load_model
from TTS.tts.layers.bark.model import GPT
from TTS.tts.layers.bark.model_fine import FineGPT
from TTS.tts.models.base_tts import BaseTTS
from TTS.utils.generic_utils import (
    is_pytorch_at_least_2_4,
    slugify,
    warn_synthesize_config_deprecated,
    warn_synthesize_speaker_id_deprecated,
)

logger = logging.getLogger(__name__)


@dataclass
class BarkAudioConfig(Coqpit):
    sample_rate: int = 24000
    output_sample_rate: int = 24000


class Bark(BaseTTS):
    def __init__(
        self,
        config: Coqpit,
        tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased"),
    ) -> None:
        super().__init__(config=config, ap=None, tokenizer=None, speaker_manager=None, language_manager=None)
        self.config.num_chars = len(tokenizer)
        self.tokenizer = tokenizer
        self.semantic_model = GPT(config.semantic_config)
        self.coarse_model = GPT(config.coarse_config)
        self.fine_model = FineGPT(config.fine_config)
        self.encodec = EncodecModel.encodec_model_24khz()
        self.encodec.set_target_bandwidth(6.0)

    def load_bark_models(self):
        self.semantic_model, self.config = load_model(
            ckpt_path=self.config.LOCAL_MODEL_PATHS["text"], device=self.device, config=self.config, model_type="text"
        )
        self.coarse_model, self.config = load_model(
            ckpt_path=self.config.LOCAL_MODEL_PATHS["coarse"],
            device=self.device,
            config=self.config,
            model_type="coarse",
        )
        self.fine_model, self.config = load_model(
            ckpt_path=self.config.LOCAL_MODEL_PATHS["fine"], device=self.device, config=self.config, model_type="fine"
        )

    def train_step(self):
        pass

    def text_to_semantic(
        self,
        text: str,
        history_prompt: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
        temp: float = 0.7,
        base: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        allow_early_stop: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Generate semantic array from text.

        Args:
            text: text to be turned into audio
            history_prompt: history choice for audio cloning
            temp: generation temperature (1.0 more diverse, 0.0 more conservative)

        Returns:
            numpy semantic array to be fed into `semantic_to_waveform`
        """
        x_semantic = generate_text_semantic(
            text,
            self,
            history_prompt=history_prompt,
            temp=temp,
            base=base,
            allow_early_stop=allow_early_stop,
            **kwargs,
        )
        return x_semantic

    def semantic_to_waveform(
        self,
        semantic_tokens: torch.Tensor,
        history_prompt: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
        temp: float = 0.7,
        base: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate audio array from semantic input.

        Args:
            semantic_tokens: semantic token output from `text_to_semantic`
            history_prompt: history choice for audio cloning
            temp: generation temperature (1.0 more diverse, 0.0 more conservative)

        Returns:
            numpy audio array at sample frequency 24khz
        """
        x_coarse_gen = generate_coarse(
            semantic_tokens,
            self,
            history_prompt=history_prompt,
            temp=temp,
            base=base,
        )
        x_fine_gen = generate_fine(
            x_coarse_gen,
            self,
            history_prompt=history_prompt,
            temp=0.5,
            base=base,
        )
        audio_arr = codec_decode(x_fine_gen, self)
        return audio_arr, x_coarse_gen, x_fine_gen

    def generate_audio(
        self,
        text: str,
        history_prompt: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
        text_temp: float = 0.7,
        waveform_temp: float = 0.7,
        base: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        allow_early_stop: bool = True,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate audio array from input text.

        Args:
            text: text to be turned into audio
            history_prompt: history choice for audio cloning
            text_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
            waveform_temp: generation temperature (1.0 more diverse, 0.0 more conservative)

        Returns:
            numpy audio array at sample frequency 24khz
        """
        x_semantic = self.text_to_semantic(
            text,
            history_prompt=history_prompt,
            temp=text_temp,
            base=base,
            allow_early_stop=allow_early_stop,
            **kwargs,
        )
        audio_arr, coarse, fine = self.semantic_to_waveform(
            x_semantic, history_prompt=history_prompt, temp=waveform_temp, base=base
        )
        return audio_arr, x_semantic, coarse, fine

    def _generate_voice(self, speaker_wav: str | os.PathLike[Any]) -> dict[str, torch.Tensor]:
        """Generate a new voice from the given audio."""
        audio, sr = torchaudio.load(speaker_wav)
        audio = convert_audio(audio, sr, self.config.sample_rate, self.encodec.channels)
        audio = audio.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            encoded_frames = self.encodec.encode(audio)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]

        # generate semantic tokens
        # Load the HuBERT model
        hubert_manager = HubertManager()
        hubert_manager.make_sure_tokenizer_installed(model_path=self.config.LOCAL_MODEL_PATHS["hubert_tokenizer"])

        hubert_model = CustomHubert().to(self.device)

        # Load the CustomTokenizer model
        tokenizer = HubertTokenizer.load_from_checkpoint(
            self.config.LOCAL_MODEL_PATHS["hubert_tokenizer"], map_location=self.device
        )
        # semantic_tokens = self.text_to_semantic(
        #     text, max_gen_duration_s=seconds, top_k=50, top_p=0.95, temp=0.7
        # )  # not 100%
        with torch.inference_mode():
            semantic_vectors = hubert_model.forward(audio[0], input_sample_hz=self.config.sample_rate)
        semantic_tokens = tokenizer.get_token(semantic_vectors)
        return {
            "semantic_prompt": semantic_tokens,
            "coarse_prompt": codes[:2, :],
            "fine_prompt": codes,
        }

    def _clone_voice(
        self, speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]], **generate_kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if isinstance(speaker_wav, list):
            warnings.warn(
                "Bark supports only a single reference audio file, but list was provided. Using only first file."
            )
            speaker_wav = speaker_wav[0]
        voice = self._generate_voice(speaker_wav)
        metadata = {"name": self.config["model"]}
        return voice, metadata

    def get_voices(self, voice_dir: str | os.PathLike[Any]) -> dict[str, Path]:
        """Return all available voices in the given directory.

        Args:
            voice_dir: Directory to search for voices.

        Returns:
            Dictionary mapping a speaker ID to its voice file.
        """
        # For Bark we overwrite the base method to also allow loading the npz
        # files included with the original model.
        return {path.stem: path for path in Path(voice_dir).iterdir() if path.suffix in (".npz", ".pth")}

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
        # For Bark we overwrite the base method to also allow loading the npz
        # files included with the original model.
        voices = self.get_voices(voice_dir)
        if speaker_id not in voices:
            msg = f"Voice file `{slugify(speaker_id)}.pth` or .npz for speaker `{speaker_id}` not found in: {voice_dir}"
            raise FileNotFoundError(msg)
        if voices[speaker_id].suffix == ".npz":
            np_voice = np.load(voices[speaker_id])
            voice = {key: torch.tensor(np_voice[key]) for key in np_voice.keys()}
        else:
            voice = torch.load(voices[speaker_id], map_location="cpu", weights_only=is_pytorch_at_least_2_4())
        logger.info("Loaded voice `%s` from: %s", speaker_id, voices[speaker_id])
        return voice

    def synthesize(
        self,
        text: str,
        config: BaseTTSConfig | None = None,
        *,
        speaker: str | None = None,
        speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]] | None = None,
        voice_dir: str | os.PathLike[Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Synthesize speech with the given input text.

        Args:
            text (str): Input text.
            config: DEPRECATED. Not used.
            speaker: Custom speaker ID to cache or retrieve a voice.
            speaker_wav: Path(s) to reference audio.
            voice_dir: Folder for cached voices.
            **kwargs: Model specific inference settings used by `generate_audio()` and
                      `TTS.tts.layers.bark.inference_funcs.generate_text_semantic()`.

        Returns:
            A dictionary of the output values with `wav` as output waveform,
            `deterministic_seed` as seed used at inference, `text_input` as text token IDs
            after tokenizer, `voice_samples` as samples used for cloning,
            `conditioning_latents` as latents used at inference.

        """
        if config is not None:
            warn_synthesize_config_deprecated()
        if (speaker_id := kwargs.pop("speaker_id", None)) is not None:
            speaker = speaker_id
            warn_synthesize_speaker_id_deprecated()
        history_prompt = None, None, None
        if speaker_wav is not None or speaker is not None:
            voice = self.clone_voice(speaker_wav, speaker, voice_dir)
            history_prompt = (voice["semantic_prompt"], voice["coarse_prompt"], voice["fine_prompt"])
        outputs = self.generate_audio(text, history_prompt=history_prompt, **kwargs)
        return {
            "wav": outputs[0],
            "text_inputs": text,
        }

    def forward(self): ...

    def inference(self): ...

    @staticmethod
    def init_from_config(config: "BarkConfig", **kwargs):  # pylint: disable=unused-argument
        return Bark(config)

    # pylint: disable=unused-argument, redefined-builtin
    def load_checkpoint(
        self,
        config,
        checkpoint_dir,
        text_model_path=None,
        coarse_model_path=None,
        fine_model_path=None,
        hubert_tokenizer_path=None,
        eval=False,
        strict=True,
        **kwargs,
    ):
        """Load a model checkpoints from a directory. This model is with multiple checkpoint files and it
        expects to have all the files to be under the given `checkpoint_dir` with the rigth names.
        If eval is True, set the model to eval mode.

        Args:
            config (BarkConfig): The model config.
            checkpoint_dir (str): The directory where the checkpoints are stored.
            text_model_path (str, optional): The path to the text model checkpoint. Defaults to None.
            coarse_model_path (str, optional): The path to the coarse model checkpoint. Defaults to None.
            fine_model_path (str, optional): The path to the fine model checkpoint. Defaults to None.
            hubert_tokenizer_path (str, optional): The path to the tokenizer checkpoint. Defaults to None.
            eval (bool, optional): Whether to set the model to eval mode. Defaults to False.
            strict (bool, optional): Whether to load the model strictly. Defaults to True.
        """
        text_model_path = text_model_path or os.path.join(checkpoint_dir, "text_2.pt")
        coarse_model_path = coarse_model_path or os.path.join(checkpoint_dir, "coarse_2.pt")
        fine_model_path = fine_model_path or os.path.join(checkpoint_dir, "fine_2.pt")
        hubert_tokenizer_path = hubert_tokenizer_path or os.path.join(checkpoint_dir, "tokenizer.pth")

        # The paths in the default config start with /root/.local/share/tts and need to be fixed
        self.config.LOCAL_MODEL_PATHS["text"] = text_model_path
        self.config.LOCAL_MODEL_PATHS["coarse"] = coarse_model_path
        self.config.LOCAL_MODEL_PATHS["fine"] = fine_model_path
        self.config.LOCAL_MODEL_PATHS["hubert_tokenizer"] = hubert_tokenizer_path
        self.config.CACHE_DIR = str(Path(text_model_path).parent)

        self.load_bark_models()

        if eval:
            self.eval()
