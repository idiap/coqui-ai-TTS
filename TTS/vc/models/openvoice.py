import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import numpy.typing as npt
import torch
from coqpit import Coqpit
from torch import nn
from torch.nn import functional as F

from TTS.tts.layers.vits.networks import PosteriorEncoder
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.audio.torch_transforms import wav_to_spec
from TTS.utils.voices import CloningMixin
from TTS.vc.configs.openvoice_config import OpenVoiceConfig
from TTS.vc.models.base_vc import BaseVC
from TTS.vc.models.freevc import Generator, ResidualCouplingBlock

logger = logging.getLogger(__name__)


class ReferenceEncoder(nn.Module):
    """NN module creating a fixed size prosody embedding from a spectrogram.

    inputs: mel spectrograms [batch_size, num_spec_frames, num_mel]
    outputs: [batch_size, embedding_dim]
    """

    def __init__(self, spec_channels: int, embedding_dim: int = 0, layernorm: bool = True) -> None:
        super().__init__()
        self.spec_channels = spec_channels
        ref_enc_filters = [32, 32, 64, 64, 128, 128]
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters
        convs = [
            torch.nn.utils.parametrizations.weight_norm(
                nn.Conv2d(
                    in_channels=filters[i],
                    out_channels=filters[i + 1],
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                )
            )
            for i in range(K)
        ]
        self.convs = nn.ModuleList(convs)

        out_channels = self.calculate_channels(spec_channels, 3, 2, 1, K)
        self.gru = nn.GRU(
            input_size=ref_enc_filters[-1] * out_channels,
            hidden_size=256 // 2,
            batch_first=True,
        )
        self.proj = nn.Linear(128, embedding_dim)
        self.layernorm = nn.LayerNorm(self.spec_channels) if layernorm else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        N = inputs.size(0)

        out = inputs.view(N, 1, -1, self.spec_channels)  # [N, 1, Ty, n_freqs]
        if self.layernorm is not None:
            out = self.layernorm(out)

        for conv in self.convs:
            out = conv(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        _memory, out = self.gru(out)  # out --- [1, N, 128]

        return self.proj(out.squeeze(0))

    def calculate_channels(self, L: int, kernel_size: int, stride: int, pad: int, n_convs: int) -> int:
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class OpenVoice(CloningMixin, BaseVC):
    """
    OpenVoice voice conversion model (inference only).

    Source: https://github.com/myshell-ai/OpenVoice
    Paper: https://arxiv.org/abs/2312.01479

    Paper abstract:
    We introduce OpenVoice, a versatile voice cloning approach that requires
    only a short audio clip from the reference speaker to replicate their voice and
    generate speech in multiple languages. OpenVoice represents a significant
    advancement in addressing the following open challenges in the field: 1)
    Flexible Voice Style Control. OpenVoice enables granular control over voice
    styles, including emotion, accent, rhythm, pauses, and intonation, in addition
    to replicating the tone color of the reference speaker. The voice styles are not
    directly copied from and constrained by the style of the reference speaker.
    Previous approaches lacked the ability to flexibly manipulate voice styles after
    cloning. 2) Zero-Shot Cross-Lingual Voice Cloning. OpenVoice achieves zero-shot
    cross-lingual voice cloning for languages not included in the massive-speaker
    training set. Unlike previous approaches, which typically require extensive
    massive-speaker multi-lingual (MSML) dataset for all languages, OpenVoice can
    clone voices into a new language without any massive-speaker training data for
    that language. OpenVoice is also computationally efficient, costing tens of
    times less than commercially available APIs that offer even inferior
    performance. To foster further research in the field, we have made the source
    code and trained model publicly accessible. We also provide qualitative results
    in our demo website. Prior to its public release, our internal version of
    OpenVoice was used tens of millions of times by users worldwide between May and
    October 2023, serving as the backend of MyShell.
    """

    def __init__(self, config: Coqpit, speaker_manager: SpeakerManager | None = None) -> None:
        super().__init__(config, None, speaker_manager, None)

        self.init_multispeaker(config)

        self.zero_g = self.args.zero_g
        self.inter_channels = self.args.inter_channels
        self.hidden_channels = self.args.hidden_channels
        self.filter_channels = self.args.filter_channels
        self.n_heads = self.args.n_heads
        self.n_layers = self.args.n_layers
        self.kernel_size = self.args.kernel_size
        self.p_dropout = self.args.p_dropout
        self.resblock = self.args.resblock
        self.resblock_kernel_sizes = self.args.resblock_kernel_sizes
        self.resblock_dilation_sizes = self.args.resblock_dilation_sizes
        self.upsample_rates = self.args.upsample_rates
        self.upsample_initial_channel = self.args.upsample_initial_channel
        self.upsample_kernel_sizes = self.args.upsample_kernel_sizes
        self.n_layers_q = self.args.n_layers_q
        self.use_spectral_norm = self.args.use_spectral_norm
        self.gin_channels = self.args.gin_channels
        self.tau = self.args.tau

        self.spec_channels = config.audio.fft_size // 2 + 1

        self.dec = Generator(
            self.inter_channels,
            self.resblock,
            self.resblock_kernel_sizes,
            self.resblock_dilation_sizes,
            self.upsample_rates,
            self.upsample_initial_channel,
            self.upsample_kernel_sizes,
            gin_channels=self.gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            self.spec_channels,
            self.inter_channels,
            self.hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            num_layers=16,
            cond_channels=self.gin_channels,
        )

        self.flow = ResidualCouplingBlock(
            self.inter_channels,
            self.hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=4,
            gin_channels=self.gin_channels,
        )

        self.ref_enc = ReferenceEncoder(self.spec_channels, self.gin_channels)

    @staticmethod
    def init_from_config(config: OpenVoiceConfig) -> "OpenVoice":
        return OpenVoice(config)

    def init_multispeaker(self, config: Coqpit, data: list[Any] | None = None) -> None:
        """Initialize multi-speaker modules of a model. A model can be trained either with a speaker embedding layer
        or with external `d_vectors` computed from a speaker encoder model.

        You must provide a `speaker_manager` at initialization to set up the multi-speaker modules.

        Args:
            config (Coqpit): Model configuration.
            data (list, optional): Dataset items to infer number of speakers. Defaults to None.
        """
        self.num_spks = config.num_speakers
        if self.speaker_manager:
            self.num_spks = self.speaker_manager.num_speakers

    def load_checkpoint(
        self,
        config: Coqpit,
        checkpoint_path: str | os.PathLike[Any],
        *,
        eval: bool = False,
        strict: bool = True,
        cache: bool = False,
    ) -> None:
        """Map from OpenVoice's config structure."""
        config_path = Path(checkpoint_path).parent / "config.json"
        with open(config_path, encoding="utf-8") as f:
            config_org = json.load(f)
        self.config.audio.input_sample_rate = config_org["data"]["sampling_rate"]
        self.config.audio.output_sample_rate = config_org["data"]["sampling_rate"]
        self.config.audio.fft_size = config_org["data"]["filter_length"]
        self.config.audio.hop_length = config_org["data"]["hop_length"]
        self.config.audio.win_length = config_org["data"]["win_length"]

        super().load_checkpoint(config, checkpoint_path, eval=eval, strict=strict, cache=cache)

    def forward(self) -> None: ...
    def train_step(self) -> None: ...
    def eval_step(self) -> None: ...

    @staticmethod
    def _set_x_lengths(x: torch.Tensor, aux_input: Mapping[str, torch.Tensor | None]) -> torch.Tensor:
        if "x_lengths" in aux_input and aux_input["x_lengths"] is not None:
            return aux_input["x_lengths"]
        return torch.tensor(x.shape[-1:]).to(x.device)

    @torch.inference_mode()
    def inference(
        self,
        x: torch.Tensor,
        aux_input: Mapping[str, torch.Tensor | None] = {"x_lengths": None, "g_src": None, "g_tgt": None},
    ) -> dict[str, torch.Tensor]:
        """
        Inference pass of the model

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch_size, c_seq_len).
            x_lengths (torch.Tensor): Lengths of the input tensor. Shape: (batch_size,).
            g_src (torch.Tensor): Source speaker embedding tensor. Shape: (batch_size, spk_emb_dim).
            g_tgt (torch.Tensor): Target speaker embedding tensor. Shape: (batch_size, spk_emb_dim).

        Returns:
            o_hat: Output spectrogram tensor. Shape: (batch_size, spec_seq_len, spec_dim).
            x_mask: Spectrogram mask. Shape: (batch_size, spec_seq_len).
            (z, z_p, z_hat): A tuple of latent variables.
        """
        x_lengths = self._set_x_lengths(x, aux_input)
        if "g_src" in aux_input and aux_input["g_src"] is not None:
            g_src = aux_input["g_src"]
        else:
            raise ValueError("aux_input must define g_src")
        if "g_tgt" in aux_input and aux_input["g_tgt"] is not None:
            g_tgt = aux_input["g_tgt"]
        else:
            raise ValueError("aux_input must define g_tgt")
        z, _m_q, _logs_q, y_mask = self.enc_q(
            x, x_lengths, g=g_src if not self.zero_g else torch.zeros_like(g_src), tau=self.tau
        )
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.dec(z_hat * y_mask, g=g_tgt if not self.zero_g else torch.zeros_like(g_tgt))
        return {
            "model_outputs": o_hat,
            "y_mask": y_mask,
            "z": z,
            "z_p": z_p,
            "z_hat": z_hat,
        }

    def load_audio(
        self, wav: str | os.PathLike[Any] | npt.NDArray[np.float32] | torch.Tensor | list[float]
    ) -> torch.Tensor:
        """Read and format the input audio."""
        if isinstance(wav, (str, os.PathLike)):
            out = torch.from_numpy(librosa.load(wav, sr=self.config.audio.input_sample_rate)[0])
        elif isinstance(wav, np.ndarray):
            out = torch.from_numpy(wav)
        elif isinstance(wav, list):
            out = torch.from_numpy(np.array(wav))
        else:
            out = wav
        return out.to(self.device).float()

    def extract_se(self, audio: str | os.PathLike[Any] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.load_audio(audio)
        y = y.to(self.device)
        y = y.unsqueeze(0)
        spec = wav_to_spec(
            y,
            n_fft=self.config.audio.fft_size,
            hop_length=self.config.audio.hop_length,
            win_length=self.config.audio.win_length,
            center=False,
        ).to(self.device)
        with torch.no_grad():
            g = self.ref_enc(spec.transpose(1, 2)).unsqueeze(-1)

        return g, spec

    def _extract_target_se(self, tgt: list[str | os.PathLike[Any] | torch.Tensor]) -> torch.Tensor:
        tgt_ses = []
        for tg in tgt:
            tgt_se, _ = self.extract_se(tg)
            tgt_ses.append(tgt_se)
        return torch.stack(tgt_ses).mean(dim=0)

    def _clone_voice(
        self, speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]], **generate_kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not isinstance(speaker_wav, list):
            speaker_wav = [speaker_wav]
        voice = {"speaker_embedding": self._extract_target_se(speaker_wav)}
        metadata = {"name": self.config["model"]}
        return voice, metadata

    @torch.inference_mode()
    def voice_conversion(
        self,
        src: str | os.PathLike[Any] | torch.Tensor,
        tgt: list[str | os.PathLike[Any] | torch.Tensor] | None = None,
        *,
        speaker_id: str | None = None,
        voice_dir: str | os.PathLike[Any] | None = None,
    ) -> npt.NDArray[np.float32]:
        """
        Voice conversion pass of the model.

        Args:
            src (str or torch.Tensor): Source utterance.
            tgt (list of str or torch.Tensor): Target utterance.

        Returns:
            Output numpy array.
        """
        src_se, src_spec = self.extract_se(src)
        if tgt is None or all(isinstance(x, (str, os.PathLike)) for x in tgt):
            tgt_se = self.clone_voice(tgt, speaker_id, voice_dir)["speaker_embedding"]
        else:
            tgt_se = self._extract_target_se(tgt)

        aux_input = {"g_src": src_se, "g_tgt": tgt_se}
        audio = self.inference(src_spec, aux_input)
        return audio["model_outputs"][0, 0].data.cpu().float().numpy()
