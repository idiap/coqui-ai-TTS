import logging
import math
import os
from dataclasses import replace
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torchaudio
from coqpit import Coqpit
from monotonic_alignment_search import maximum_path
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from trainer.io import load_fsspec
from trainer.torch import DistributedSampler, DistributedSamplerWrapper
from trainer.trainer_utils import get_optimizer, get_scheduler

from TTS.tts.configs.shared_configs import BaseTTSConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsArgs, VitsConfig
from TTS.tts.datasets.dataset import TTSDataset, _parse_sample, get_attribute_balancer_weights
from TTS.tts.layers.glow_tts.duration_predictor import DurationPredictor
from TTS.tts.layers.vits.discriminator import VitsDiscriminator
from TTS.tts.layers.vits.networks import PosteriorEncoder, ResidualCouplingBlocks, TextEncoder
from TTS.tts.layers.vits.stochastic_duration_predictor import StochasticDurationPredictor
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.fairseq import rehash_fairseq_vits_checkpoint
from TTS.tts.utils.helpers import generate_path, rand_segments, segment, sequence_mask
from TTS.tts.utils.text.characters import BaseCharacters, BaseVocabulary, _characters, _pad, _phonemes, _punctuations
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio.torch_transforms import spec_to_mel, wav_to_mel, wav_to_spec
from TTS.utils.samplers import BucketBatchSampler
from TTS.vocoder.models.hifigan_generator import HifiganGenerator
from TTS.vocoder.utils.generic_utils import plot_results

logger = logging.getLogger(__name__)

##############################
# IO / Feature extraction
##############################


@torch.no_grad()
def weights_reset(m: nn.Module):
    # check if the current module has reset_parameters and if it is reset the weight
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


def get_module_weights_sum(mdl: nn.Module):
    dict_sums = {}
    for name, w in mdl.named_parameters():
        if "weight" in name:
            value = w.data.sum().item()
            dict_sums[name] = value
    return dict_sums


def load_audio(file_path):
    """Load the audio file normalized in [-1, 1]

    Return Shapes:
        - x: :math:`[1, T]`
    """
    x, sr = torchaudio.load(file_path)
    assert (x > 1).sum() + (x < -1).sum() == 0
    return x, sr


##############################
# DATASET
##############################


class VitsDataset(TTSDataset):
    def __init__(self, model_args, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_id = self.tokenizer.characters.pad_id
        self.model_args = model_args

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        raw_text = item["text"]

        wav, _ = load_audio(item["audio_file"])
        if self.model_args.encoder_sample_rate is not None:
            if wav.size(1) % self.model_args.encoder_sample_rate != 0:
                wav = wav[:, : -int(wav.size(1) % self.model_args.encoder_sample_rate)]

        wav_filename = os.path.basename(item["audio_file"])

        token_ids = self.get_token_ids(idx, item["text"], item["language"])

        # after phonemization the text length may change
        # this is a shameful 🤭 hack to prevent longer phonemes
        # TODO: find a better fix
        if len(token_ids) > self.max_text_len or wav.shape[1] < self.min_audio_len:
            self.rescue_item_idx += 1
            return self.__getitem__(self.rescue_item_idx)

        return {
            "raw_text": raw_text,
            "token_ids": token_ids,
            "token_len": len(token_ids),
            "wav": wav,
            "wav_file": wav_filename,
            "speaker_name": item["speaker_name"],
            "language_name": item["language"],
            "audio_unique_name": item["audio_unique_name"],
        }

    @property
    def lengths(self):
        lens = []
        for item in self.samples:
            _, wav_file, *_ = _parse_sample(item)
            audio_len = os.path.getsize(wav_file) / 16 * 8  # assuming 16bit audio
            lens.append(audio_len)
        return lens

    def collate_fn(self, batch):
        """
        Return Shapes:
            - tokens: :math:`[B, T]`
            - token_lens :math:`[B]`
            - token_rel_lens :math:`[B]`
            - waveform: :math:`[B, 1, T]`
            - waveform_lens: :math:`[B]`
            - waveform_rel_lens: :math:`[B]`
            - speaker_names: :math:`[B]`
            - language_names: :math:`[B]`
            - audiofile_paths: :math:`[B]`
            - raw_texts: :math:`[B]`
            - audio_unique_names: :math:`[B]`
        """
        # convert list of dicts to dict of lists
        B = len(batch)
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}

        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x.size(1) for x in batch["wav"]]), dim=0, descending=True
        )

        max_text_len = max([len(x) for x in batch["token_ids"]])
        token_lens = torch.LongTensor(batch["token_len"])
        token_rel_lens = token_lens / token_lens.max()

        wav_lens = [w.shape[1] for w in batch["wav"]]
        wav_lens = torch.LongTensor(wav_lens)
        wav_lens_max = torch.max(wav_lens)
        wav_rel_lens = wav_lens / wav_lens_max

        token_padded = torch.LongTensor(B, max_text_len)
        wav_padded = torch.FloatTensor(B, 1, wav_lens_max)
        token_padded = token_padded.zero_() + self.pad_id
        wav_padded = wav_padded.zero_() + self.pad_id
        for i in range(len(ids_sorted_decreasing)):
            token_ids = batch["token_ids"][i]
            token_padded[i, : batch["token_len"][i]] = torch.LongTensor(token_ids)

            wav = batch["wav"][i]
            wav_padded[i, :, : wav.size(1)] = torch.FloatTensor(wav)

        return {
            "tokens": token_padded,
            "token_lens": token_lens,
            "token_rel_lens": token_rel_lens,
            "waveform": wav_padded,  # (B x T)
            "waveform_lens": wav_lens,  # (B)
            "waveform_rel_lens": wav_rel_lens,
            "speaker_names": batch["speaker_name"],
            "language_names": batch["language_name"],
            "audio_files": batch["wav_file"],
            "raw_text": batch["raw_text"],
            "audio_unique_names": batch["audio_unique_name"],
        }


##############################
# MODEL DEFINITION
##############################


class Vits(BaseTTS):
    """VITS TTS model

    Paper::
        https://arxiv.org/pdf/2106.06103.pdf

    Paper Abstract::
        Several recent end-to-end text-to-speech (TTS) models enabling single-stage training and parallel
        sampling have been proposed, but their sample quality does not match that of two-stage TTS systems.
        In this work, we present a parallel endto-end TTS method that generates more natural sounding audio than
        current two-stage models. Our method adopts variational inference augmented with normalizing flows and
        an adversarial training process, which improves the expressive power of generative modeling. We also propose a
        stochastic duration predictor to synthesize speech with diverse rhythms from input text. With the
        uncertainty modeling over latent variables and the stochastic duration predictor, our method expresses the
        natural one-to-many relationship in which a text input can be spoken in multiple ways
        with different pitches and rhythms. A subjective human evaluation (mean opinion score, or MOS)
        on the LJ Speech, a single speaker dataset, shows that our method outperforms the best publicly
        available TTS systems and achieves a MOS comparable to ground truth.

    Check :class:`TTS.tts.configs.vits_config.VitsConfig` for class arguments.

    Examples:
        >>> from TTS.tts.configs.vits_config import VitsConfig
        >>> from TTS.tts.models.vits import Vits
        >>> config = VitsConfig()
        >>> model = Vits(config)
    """

    config: VitsConfig
    args: VitsArgs

    def __init__(
        self,
        config: Coqpit,
        ap: None = None,
        tokenizer: None = None,
        speaker_manager: None = None,
    ):
        super().__init__(config, ap, tokenizer, speaker_manager)

        upsample_rate = torch.prod(torch.as_tensor(config.model_args.upsample_rates_decoder)).item()

        if not config.model_args.encoder_sample_rate:
            assert upsample_rate == config.audio.hop_length, (
                f" [!] Product of upsample rates must be equal to the hop length - {upsample_rate} vs {config.audio.hop_length}"
            )
        else:
            encoder_to_vocoder_upsampling_factor = config.audio.sample_rate / config.model_args.encoder_sample_rate
            effective_hop_length = config.audio.hop_length * encoder_to_vocoder_upsampling_factor
            assert config.model_args.interpolate_z or upsample_rate == effective_hop_length, (
                f" [!] Product of upsample rates must be equal to the hop length - {upsample_rate} vs {effective_hop_length}"
            )

        self.init_multispeaker()
        self.init_multilingual(self.config)
        self.init_upsampling()

        self.length_scale = self.args.length_scale
        self.noise_scale = self.args.noise_scale
        self.inference_noise_scale = self.args.inference_noise_scale
        self.inference_noise_scale_dp = self.args.inference_noise_scale_dp
        self.noise_scale_dp = self.args.noise_scale_dp
        self.max_inference_len = self.args.max_inference_len
        self.spec_segment_size = self.args.spec_segment_size

        self.text_encoder = TextEncoder(
            self.args.num_chars,
            self.args.hidden_channels,
            self.args.hidden_channels,
            self.args.hidden_channels_ffn_text_encoder,
            self.args.num_heads_text_encoder,
            self.args.num_layers_text_encoder,
            self.args.kernel_size_text_encoder,
            self.args.dropout_p_text_encoder,
            language_emb_dim=self.embedded_language_dim,
        )

        self.posterior_encoder = PosteriorEncoder(
            self.args.out_channels,
            self.args.hidden_channels,
            self.args.hidden_channels,
            kernel_size=self.args.kernel_size_posterior_encoder,
            dilation_rate=self.args.dilation_rate_posterior_encoder,
            num_layers=self.args.num_layers_posterior_encoder,
            cond_channels=self.embedded_speaker_dim,
        )

        self.flow = ResidualCouplingBlocks(
            self.args.hidden_channels,
            self.args.hidden_channels,
            kernel_size=self.args.kernel_size_flow,
            dilation_rate=self.args.dilation_rate_flow,
            num_layers=self.args.num_layers_flow,
            cond_channels=self.embedded_speaker_dim,
        )

        if self.args.use_sdp:
            self.duration_predictor = StochasticDurationPredictor(
                self.args.hidden_channels,
                192,
                3,
                self.args.dropout_p_duration_predictor,
                4,
                cond_channels=self.embedded_speaker_dim if self.args.condition_dp_on_speaker else 0,
                language_emb_dim=self.embedded_language_dim,
            )
        else:
            self.duration_predictor = DurationPredictor(
                self.args.hidden_channels,
                256,
                3,
                self.args.dropout_p_duration_predictor,
                cond_channels=self.embedded_speaker_dim,
                language_emb_dim=self.embedded_language_dim,
            )

        self.waveform_decoder = HifiganGenerator(
            self.args.hidden_channels,
            1,
            self.args.resblock_type_decoder,
            self.args.resblock_dilation_sizes_decoder,
            self.args.resblock_kernel_sizes_decoder,
            self.args.upsample_kernel_sizes_decoder,
            self.args.upsample_initial_channel_decoder,
            self.args.upsample_rates_decoder,
            inference_padding=0,
            cond_channels=self.embedded_speaker_dim,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
        )

        if self.args.init_discriminator:
            self.disc = VitsDiscriminator(
                periods=self.args.periods_multi_period_discriminator,
                use_spectral_norm=self.args.use_spectral_norm_disriminator,
            )

    def init_multispeaker(self) -> None:
        """Initialize multi-speaker modules of a model.

        A model can be trained either with a speaker embedding layer
        or with external `d_vectors` computed from a speaker encoder model.

        You must provide a `speaker_manager` at initialization to set up the multi-speaker modules.
        """
        super().init_multispeaker()
        self.audio_transform = None

        # TODO: make this a function
        if self.args.use_speaker_encoder_as_loss:
            if self.speaker_manager.encoder is None and (
                not self.args.speaker_encoder_model_path or not self.args.speaker_encoder_config_path
            ):
                raise RuntimeError(
                    " [!] To use the speaker consistency loss (SCL) you need to specify speaker_encoder_model_path and speaker_encoder_config_path !!"
                )

            self.speaker_manager.encoder.eval()
            logger.info("External Speaker Encoder Loaded !!")

            if (
                hasattr(self.speaker_manager.encoder, "audio_config")
                and self.config.audio.sample_rate != self.speaker_manager.encoder.audio_config["sample_rate"]
            ):
                self.audio_transform = torchaudio.transforms.Resample(
                    orig_freq=self.config.audio.sample_rate,
                    new_freq=self.speaker_manager.encoder.audio_config["sample_rate"],
                )

    def _init_speaker_embedding(self):
        if self.num_speakers > 0:
            logger.info("Initialization of speaker-embedding layers.")
            self.embedded_speaker_dim = self.args.speaker_embedding_channels
            self.emb_g = nn.Embedding(self.num_speakers, self.embedded_speaker_dim)

    def _init_d_vector(self):
        if hasattr(self, "emb_g"):
            raise ValueError("[!] Speaker embedding layer already initialized before d_vector settings.")
        self.embedded_speaker_dim = self.args.d_vector_dim

    def init_multilingual(self, config: Coqpit):
        """Initialize multilingual modules of a model.

        Args:
            config (Coqpit): Model configuration.
        """
        # For one language this does not necessarily make sense, but need to support
        # it for existing models that do this.
        if self.args.use_language_embedding and self.language_manager.num_languages > 0:
            logger.info("Initialization of language-embedding layers.")
            self.embedded_language_dim = self.args.embedded_language_dim
            self.emb_l = nn.Embedding(self.language_manager.num_languages, self.embedded_language_dim)
            torch.nn.init.xavier_uniform_(self.emb_l.weight)
        else:
            self.embedded_language_dim = 0

    def init_upsampling(self):
        """
        Initialize upsampling modules of a model.
        """
        if self.args.encoder_sample_rate:
            self.interpolate_factor = self.config.audio["sample_rate"] / self.args.encoder_sample_rate
            self.audio_resampler = torchaudio.transforms.Resample(
                orig_freq=self.config.audio["sample_rate"], new_freq=self.args.encoder_sample_rate
            )  # pylint: disable=W0201

    def on_epoch_start(self, trainer):  # pylint: disable=W0613
        """Freeze layers at the beginning of an epoch"""
        self._freeze_layers()
        # set the device of speaker encoder
        if self.args.use_speaker_encoder_as_loss:
            self.speaker_manager.encoder = self.speaker_manager.encoder.to(self.device)

    def on_init_end(self, trainer):  # pylint: disable=W0613
        """Reinit layes if needed"""
        if self.args.reinit_DP:
            before_dict = get_module_weights_sum(self.duration_predictor)
            # Applies weights_reset recursively to every submodule of the duration predictor
            self.duration_predictor.apply(fn=weights_reset)
            after_dict = get_module_weights_sum(self.duration_predictor)
            for key, value in after_dict.items():
                if value == before_dict[key]:
                    raise RuntimeError(" [!] The weights of Duration Predictor was not reinit check it !")
            logger.info("Duration Predictor was reinit.")

        if self.args.reinit_text_encoder:
            before_dict = get_module_weights_sum(self.text_encoder)
            # Applies weights_reset recursively to every submodule of the duration predictor
            self.text_encoder.apply(fn=weights_reset)
            after_dict = get_module_weights_sum(self.text_encoder)
            for key, value in after_dict.items():
                if value == before_dict[key]:
                    raise RuntimeError(" [!] The weights of Text Encoder was not reinit check it !")
            logger.info("Text Encoder was reinit.")

    def _freeze_layers(self):
        if self.args.freeze_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

            if hasattr(self, "emb_l"):
                for param in self.emb_l.parameters():
                    param.requires_grad = False

        if self.args.freeze_PE:
            for param in self.posterior_encoder.parameters():
                param.requires_grad = False

        if self.args.freeze_DP:
            for param in self.duration_predictor.parameters():
                param.requires_grad = False

        if self.args.freeze_flow_decoder:
            for param in self.flow.parameters():
                param.requires_grad = False

        if self.args.freeze_waveform_decoder:
            for param in self.waveform_decoder.parameters():
                param.requires_grad = False

    def forward_mas(self, outputs, z_p, m_p, logs_p, x, x_mask, y_mask, g, lang_emb):
        # find the alignment path
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        with torch.no_grad():
            o_scale = torch.exp(-2 * logs_p)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1]).unsqueeze(-1)  # [b, t, 1]
            logp2 = torch.einsum("klm, kln -> kmn", [o_scale, -0.5 * (z_p**2)])
            logp3 = torch.einsum("klm, kln -> kmn", [m_p * o_scale, z_p])
            logp4 = torch.sum(-0.5 * (m_p**2) * o_scale, [1]).unsqueeze(-1)  # [b, t, 1]
            logp = logp2 + logp3 + logp1 + logp4
            attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()  # [b, 1, t, t']

        # duration predictor
        attn_durations = attn.sum(3)
        if self.args.use_sdp:
            loss_duration = self.duration_predictor(
                x.detach() if self.args.detach_dp_input else x,
                x_mask,
                attn_durations,
                g=g.detach() if self.args.detach_dp_input and g is not None else g,
                lang_emb=lang_emb.detach() if self.args.detach_dp_input and lang_emb is not None else lang_emb,
            )
            loss_duration = loss_duration / torch.sum(x_mask)
        else:
            attn_log_durations = torch.log(attn_durations + 1e-6) * x_mask
            log_durations = self.duration_predictor(
                x.detach() if self.args.detach_dp_input else x,
                x_mask,
                g=g.detach() if self.args.detach_dp_input and g is not None else g,
                lang_emb=lang_emb.detach() if self.args.detach_dp_input and lang_emb is not None else lang_emb,
            )
            loss_duration = torch.sum((log_durations - attn_log_durations) ** 2, [1, 2]) / torch.sum(x_mask)
        outputs["loss_duration"] = loss_duration
        return outputs, attn

    def upsampling_z(self, z, slice_ids=None, y_lengths=None, y_mask=None):
        spec_segment_size = self.spec_segment_size
        if self.args.encoder_sample_rate:
            # recompute the slices and spec_segment_size if needed
            slice_ids = slice_ids * int(self.interpolate_factor) if slice_ids is not None else slice_ids
            spec_segment_size = spec_segment_size * int(self.interpolate_factor)
            # interpolate z if needed
            if self.args.interpolate_z:
                z = torch.nn.functional.interpolate(z, scale_factor=[self.interpolate_factor], mode="linear").squeeze(0)
                # recompute the mask if needed
                if y_lengths is not None and y_mask is not None:
                    y_mask = (
                        sequence_mask(y_lengths * self.interpolate_factor, None).to(y_mask.dtype).unsqueeze(1)
                    )  # [B, 1, T_dec_resampled]

        return z, spec_segment_size, slice_ids, y_mask

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        waveform: torch.Tensor,
        aux_input: dict[str, Any] | None = None,
    ) -> dict:
        """Forward pass of the model.

        Args:
            x: Batch of input character sequence IDs.
            x_lengths: Batch of input character sequence lengths.
            y: Batch of input spectrograms.
            y_lengths: Batch of input spectrogram lengths.
            waveform: Batch of ground truth waveforms per sample.
            aux_input: Auxiliary inputs for multi-speaker and multi-lingual training.
                Defaults to {"d_vectors": None, "speaker_ids": None, "language_ids": None}.

        Returns:
            Dict: model outputs keyed by the output name.

        Shapes:
            - x: :math:`[B, T_seq]`
            - x_lengths: :math:`[B]`
            - y: :math:`[B, C, T_spec]`
            - y_lengths: :math:`[B]`
            - waveform: :math:`[B, 1, T_wav]`
            - d_vectors: :math:`[B, C, 1]`
            - speaker_ids: :math:`[B]`
            - language_ids: :math:`[B]`

        Return Shapes:
            - model_outputs: :math:`[B, 1, T_wav]`
            - alignments: :math:`[B, T_seq, T_dec]`
            - z: :math:`[B, C, T_dec]`
            - z_p: :math:`[B, C, T_dec]`
            - m_p: :math:`[B, C, T_dec]`
            - logs_p: :math:`[B, C, T_dec]`
            - m_q: :math:`[B, C, T_dec]`
            - logs_q: :math:`[B, C, T_dec]`
            - waveform_seg: :math:`[B, 1, spec_seg_size * hop_length]`
            - gt_spk_emb: :math:`[B, 1, speaker_encoder.proj_dim]`
            - syn_spk_emb: :math:`[B, 1, speaker_encoder.proj_dim]`
        """
        if aux_input is None:
            aux_input = {"d_vectors": None, "speaker_ids": None, "language_ids": None}
        outputs = {}
        g = self._get_speaker_conditioning(aux_input, "emb_g", normalize_embedding=False)

        # language embedding
        lang_emb = None
        lid = aux_input.get("language_ids")
        if self.args.use_language_embedding and lid is not None:
            lang_emb = self.emb_l(lid).unsqueeze(-1)

        x, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths, lang_emb=lang_emb)

        # posterior encoder
        z, m_q, logs_q, y_mask = self.posterior_encoder(y, y_lengths, g=g)

        # flow layers
        z_p = self.flow(z, y_mask, g=g)

        # duration predictor
        outputs, attn = self.forward_mas(outputs, z_p, m_p, logs_p, x, x_mask, y_mask, g=g, lang_emb=lang_emb)

        # expand prior
        m_p = torch.einsum("klmn, kjm -> kjn", [attn, m_p])
        logs_p = torch.einsum("klmn, kjm -> kjn", [attn, logs_p])

        # select a random feature segment for the waveform decoder
        z_slice, slice_ids = rand_segments(z, y_lengths, self.spec_segment_size, let_short_samples=True, pad_short=True)

        # interpolate z if needed
        z_slice, spec_segment_size, slice_ids, _ = self.upsampling_z(z_slice, slice_ids=slice_ids)

        o = self.waveform_decoder(z_slice, g=g)

        wav_seg = segment(
            waveform,
            slice_ids * self.config.audio.hop_length,
            spec_segment_size * self.config.audio.hop_length,
            pad_short=True,
        )

        if self.args.use_speaker_encoder_as_loss and self.speaker_manager.encoder is not None:
            # concate generated and GT waveforms
            wavs_batch = torch.cat((wav_seg, o), dim=0)

            # resample audio to speaker encoder sample_rate
            # pylint: disable=W0105
            if self.audio_transform is not None:
                wavs_batch = self.audio_transform(wavs_batch)

            pred_embs = self.speaker_manager.encoder.forward(wavs_batch, l2_norm=True)

            # split generated and GT speaker embeddings
            gt_spk_emb, syn_spk_emb = torch.chunk(pred_embs, 2, dim=0)
        else:
            gt_spk_emb, syn_spk_emb = None, None

        outputs.update(
            {
                "model_outputs": o,
                "alignments": attn.squeeze(1),
                "m_p": m_p,
                "logs_p": logs_p,
                "z": z,
                "z_p": z_p,
                "m_q": m_q,
                "logs_q": logs_q,
                "waveform_seg": wav_seg,
                "gt_spk_emb": gt_spk_emb,
                "syn_spk_emb": syn_spk_emb,
                "slice_ids": slice_ids,
            }
        )
        return outputs

    @staticmethod
    def _set_x_lengths(x, aux_input):
        if "x_lengths" in aux_input and aux_input["x_lengths"] is not None:
            return aux_input["x_lengths"]
        return torch.tensor(x.shape[1:2]).to(x.device)

    @torch.inference_mode()
    def inference(
        self,
        x: torch.Tensor,
        aux_input: dict[str, Any] | None = None,
    ):
        """
        Note:
            To run in batch mode, provide `x_lengths` else model assumes that the batch size is 1.

        Shapes:
            - x: :math:`[B, T_seq]`
            - x_lengths: :math:`[B]`
            - d_vectors: :math:`[B, C]`
            - speaker_ids: :math:`[B]`

        Return Shapes:
            - model_outputs: :math:`[B, 1, T_wav]`
            - alignments: :math:`[B, T_seq, T_dec]`
            - z: :math:`[B, C, T_dec]`
            - z_p: :math:`[B, C, T_dec]`
            - m_p: :math:`[B, C, T_dec]`
            - logs_p: :math:`[B, C, T_dec]`
        """
        if aux_input is None:
            aux_input = {
                "x_lengths": None,
                "d_vectors": None,
                "speaker_ids": None,
                "language_ids": None,
                "durations": None,
            }
        g = self._get_speaker_conditioning(aux_input, "emb_g", normalize_embedding=False)
        x_lengths = self._set_x_lengths(x, aux_input)

        # language embedding
        lang_emb = None
        lid = aux_input.get("language_ids")
        if self.args.use_language_embedding and lid is not None:
            lang_emb = self.emb_l(lid).unsqueeze(-1)

        x, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths, lang_emb=lang_emb)

        durations = aux_input.get("durations")
        if durations is None:
            if self.args.use_sdp:
                logw = self.duration_predictor(
                    x,
                    x_mask,
                    g=g if self.args.condition_dp_on_speaker else None,
                    reverse=True,
                    noise_scale=self.inference_noise_scale_dp,
                    lang_emb=lang_emb,
                )
            else:
                logw = self.duration_predictor(
                    x, x_mask, g=g if self.args.condition_dp_on_speaker else None, lang_emb=lang_emb
                )
            w = torch.exp(logw) * x_mask * self.length_scale
        else:
            assert durations.shape[-1] == x.shape[-1]
            w = durations.unsqueeze(0)

        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).to(x_mask.dtype).unsqueeze(1)  # [B, 1, T_dec]

        attn_mask = x_mask * y_mask.transpose(1, 2)  # [B, 1, T_enc] * [B, T_dec, 1]
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1).transpose(1, 2))

        m_p = torch.matmul(attn.transpose(1, 2), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.transpose(1, 2), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * self.inference_noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)

        # upsampling if needed
        z, _, _, y_mask = self.upsampling_z(z, y_lengths=y_lengths, y_mask=y_mask)

        o = self.waveform_decoder((z * y_mask)[:, :, : self.max_inference_len], g=g)

        outputs = {
            "model_outputs": o,
            "alignments": attn.squeeze(1),
            "durations": w_ceil,
            "z": z,
            "z_p": z_p,
            "m_p": m_p,
            "logs_p": logs_p,
            "y_mask": y_mask,
        }
        return outputs

    @torch.inference_mode()
    def voice_conversion(
        self,
        source_wav,
        target_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]] | None,
        *,
        source_speaker=None,
        speaker=None,
        voice_dir: str | os.PathLike[Any] | None = None,
    ):
        """Inference for voice conversion

        Args:
            source_wav: Path to source audio file.
            target_wav: Path(s) to target audio file(s).
            source_speaker: Source speaker name.
            speaker: Target speaker name.
            voice_dir: Directory to cache voices.
        """
        speaker_id, d_vector = self._get_speaker_id_or_dvector(speaker, target_wav, voice_dir)
        source_speaker_id, source_d_vector = self._get_speaker_id_or_dvector(source_speaker, source_wav, voice_dir)
        y = torch.tensor(
            self.ap.load_wav(
                source_wav, sr=self.args.encoder_sample_rate if self.args.encoder_sample_rate else self.ap.sample_rate
            ),
            dtype=torch.float,
            device=self.device,
        ).unsqueeze(0)
        # compute spectrograms
        y = wav_to_spec(
            y,
            self.config.audio.fft_size,
            self.config.audio.hop_length,
            self.config.audio.win_length,
            center=False,
        )
        y_lengths = torch.tensor([y.size(-1)]).to(y.device)
        speaker_cond_src = source_speaker_id if source_speaker_id is not None else source_d_vector
        speaker_cond_tgt = speaker_id if speaker_id is not None else d_vector
        wav, _, _ = self.inference_voice_conversion(y, y_lengths, speaker_cond_src, speaker_cond_tgt)
        return wav.squeeze()

    @torch.inference_mode()
    def inference_voice_conversion(self, y, y_lengths, speaker_cond_src, speaker_cond_tgt):
        """Forward pass for voice conversion

        Args:
            y (Tensor): Source spectrograms. Tensor of shape [B, T, C]
            y_lengths (Tensor): Length of each source spectrogram. Tensor of shape [B]
            speaker_cond_src (Tensor): Source speaker ID. Tensor of shape [B,]
            speaker_cond_tgt (Tensor): Target speaker ID. Tensor of shape [B,]
        """
        assert self.num_speakers > 0, "num_speakers have to be larger than 0."
        # speaker embedding
        if self.args.use_speaker_embedding and not self.args.use_d_vector_file:
            g_src = self.emb_g(torch.from_numpy(np.array(speaker_cond_src)).unsqueeze(0)).unsqueeze(-1)
            g_tgt = self.emb_g(torch.from_numpy(np.array(speaker_cond_tgt)).unsqueeze(0)).unsqueeze(-1)
        elif not self.args.use_speaker_embedding and self.args.use_d_vector_file:
            g_src = F.normalize(speaker_cond_src).unsqueeze(-1)
            g_tgt = F.normalize(speaker_cond_tgt).unsqueeze(-1)
        else:
            raise RuntimeError(" [!] Voice conversion is only supported on multi-speaker models.")

        z, _, _, y_mask = self.posterior_encoder(y, y_lengths, g=g_src)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.waveform_decoder(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)

    def train_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int) -> tuple[dict, dict]:
        """Perform a single training step. Run the model forward pass and compute losses.

        Args:
            batch (Dict): Input tensors.
            criterion (nn.Module): Loss layer designed for the model.
            optimizer_idx (int): Index of optimizer to use. 0 for the discriminator and 1 for the generator networks.

        Returns:
            Tuple[Dict, Dict]: Model ouputs and computed losses.
        """

        spec_lens = batch["spec_lens"]

        if optimizer_idx == 0:
            tokens = batch["tokens"]
            token_lenghts = batch["token_lens"]
            spec = batch["spec"]

            d_vectors = batch["d_vectors"]
            speaker_ids = batch["speaker_ids"]
            language_ids = batch["language_ids"]
            waveform = batch["waveform"]

            # generator pass
            outputs = self.forward(
                tokens,
                token_lenghts,
                spec,
                spec_lens,
                waveform,
                aux_input={"d_vectors": d_vectors, "speaker_ids": speaker_ids, "language_ids": language_ids},
            )

            # cache tensors for the generator pass
            self.model_outputs_cache = outputs  # pylint: disable=attribute-defined-outside-init

            # compute scores and features
            scores_disc_fake, _, scores_disc_real, _ = self.disc(
                outputs["model_outputs"].detach(), outputs["waveform_seg"]
            )

            # compute loss
            with torch.autocast("cuda", enabled=False):  # use float32 for the criterion
                loss_dict = criterion[optimizer_idx](
                    scores_disc_real,
                    scores_disc_fake,
                )
            return outputs, loss_dict

        if optimizer_idx == 1:
            mel = batch["mel"]

            # compute melspec segment
            with torch.autocast("cuda", enabled=False):
                if self.args.encoder_sample_rate:
                    spec_segment_size = self.spec_segment_size * int(self.interpolate_factor)
                else:
                    spec_segment_size = self.spec_segment_size

                mel_slice = segment(
                    mel.float(), self.model_outputs_cache["slice_ids"], spec_segment_size, pad_short=True
                )
                mel_slice_hat = wav_to_mel(
                    y=self.model_outputs_cache["model_outputs"].float(),
                    n_fft=self.config.audio.fft_size,
                    sample_rate=self.config.audio.sample_rate,
                    num_mels=self.config.audio.num_mels,
                    hop_length=self.config.audio.hop_length,
                    win_length=self.config.audio.win_length,
                    fmin=self.config.audio.mel_fmin,
                    fmax=self.config.audio.mel_fmax,
                    center=False,
                )

            # compute discriminator scores and features
            scores_disc_fake, feats_disc_fake, _, feats_disc_real = self.disc(
                self.model_outputs_cache["model_outputs"], self.model_outputs_cache["waveform_seg"]
            )

            # compute losses
            with torch.autocast("cuda", enabled=False):  # use float32 for the criterion
                loss_dict = criterion[optimizer_idx](
                    mel_slice_hat=mel_slice.float(),
                    mel_slice=mel_slice_hat.float(),
                    z_p=self.model_outputs_cache["z_p"].float(),
                    logs_q=self.model_outputs_cache["logs_q"].float(),
                    m_p=self.model_outputs_cache["m_p"].float(),
                    logs_p=self.model_outputs_cache["logs_p"].float(),
                    z_len=spec_lens,
                    scores_disc_fake=scores_disc_fake,
                    feats_disc_fake=feats_disc_fake,
                    feats_disc_real=feats_disc_real,
                    loss_duration=self.model_outputs_cache["loss_duration"],
                    use_speaker_encoder_as_loss=self.args.use_speaker_encoder_as_loss,
                    gt_spk_emb=self.model_outputs_cache["gt_spk_emb"],
                    syn_spk_emb=self.model_outputs_cache["syn_spk_emb"],
                )

            return self.model_outputs_cache, loss_dict

        raise ValueError(" [!] Unexpected `optimizer_idx`.")

    def _create_logs(self, batch, outputs: list[dict[str, Any]]):
        from TTS.tts.utils.visual import plot_alignment

        y_hat = outputs[1]["model_outputs"]
        y = outputs[1]["waveform_seg"]
        figures = plot_results(y_hat, y, self.ap)
        sample_voice = y_hat[0].squeeze(0).detach().cpu().numpy()
        audios = {"audio": sample_voice}

        alignments = outputs[1]["alignments"]
        align_img = alignments[0].data.cpu().numpy().T

        figures.update({"alignment": plot_alignment(align_img, output_fig=False)})
        return figures, audios

    @torch.inference_mode()
    def test_run(self, assets) -> dict[str, Any]:
        """Vits-specific test run method.

        Returns:
            Dictionary with test figures and audios to be projected to Tensorboard.
        """
        from TTS.tts.utils.visual import plot_alignment

        logger.info("Synthesizing test sentences.")
        test_audios = {}
        test_figures = {}
        test_sentences = self.config.test_sentences
        for idx, s_info in enumerate(test_sentences):
            aux_inputs = self.get_aux_input_from_test_sentences(s_info)
            outputs = self.synthesize(
                aux_inputs["text"],
                speaker=aux_inputs.get("speaker", None),
                language=aux_inputs.get("language", None),
                use_griffin_lim=True,
            )
            test_audios[f"{idx}-audio"] = outputs["wav"]
            test_figures[f"{idx}-alignment"] = plot_alignment(outputs["alignments"].permute(2, 1, 0), output_fig=False)
        return {"figures": test_figures, "audios": test_audios}

    def format_batch(self, batch: dict) -> dict:
        """Compute speaker, langugage IDs and d_vector for the batch if necessary."""
        speaker_ids = None
        language_ids = None
        d_vectors = None

        # get numerical speaker ids from speaker names
        if self.speaker_manager is not None and self.speaker_manager.name_to_id and self.args.use_speaker_embedding:
            speaker_ids = [self.speaker_manager.name_to_id[sn] for sn in batch["speaker_names"]]

        if speaker_ids is not None:
            speaker_ids = torch.LongTensor(speaker_ids)

        # get d_vectors from audio file names
        if self.speaker_manager is not None and self.speaker_manager.embeddings and self.args.use_d_vector_file:
            d_vector_mapping = self.speaker_manager.embeddings
            d_vectors = [d_vector_mapping[w]["embedding"] for w in batch["audio_unique_names"]]
            d_vectors = torch.FloatTensor(d_vectors)

        # get language ids from language names
        if self.language_manager.name_to_id and self.args.use_language_embedding:
            language_ids = [self.language_manager.name_to_id[ln] for ln in batch["language_names"]]

        if language_ids is not None:
            language_ids = torch.LongTensor(language_ids)

        batch["language_ids"] = language_ids
        batch["d_vectors"] = d_vectors
        batch["speaker_ids"] = speaker_ids
        return batch

    def format_batch_on_device(self, batch):
        """Compute spectrograms on the device."""
        ac = self.config.audio

        if self.args.encoder_sample_rate:
            wav = self.audio_resampler(batch["waveform"])
        else:
            wav = batch["waveform"]

        # compute spectrograms
        batch["spec"] = wav_to_spec(wav, ac.fft_size, ac.hop_length, ac.win_length, center=False)

        if self.args.encoder_sample_rate:
            # recompute spec with high sampling rate to the loss
            spec_mel = wav_to_spec(batch["waveform"], ac.fft_size, ac.hop_length, ac.win_length, center=False)
            # remove extra stft frames if needed
            if spec_mel.size(2) > int(batch["spec"].size(2) * self.interpolate_factor):
                spec_mel = spec_mel[:, :, : int(batch["spec"].size(2) * self.interpolate_factor)]
            else:
                batch["spec"] = batch["spec"][:, :, : int(spec_mel.size(2) / self.interpolate_factor)]
        else:
            spec_mel = batch["spec"]

        batch["mel"] = spec_to_mel(
            spec=spec_mel,
            n_fft=ac.fft_size,
            num_mels=ac.num_mels,
            sample_rate=ac.sample_rate,
            fmin=ac.mel_fmin,
            fmax=ac.mel_fmax,
        )

        if self.args.encoder_sample_rate:
            assert batch["spec"].shape[2] == int(batch["mel"].shape[2] / self.interpolate_factor), (
                f"{batch['spec'].shape[2]}, {batch['mel'].shape[2]}"
            )
        else:
            assert batch["spec"].shape[2] == batch["mel"].shape[2], f"{batch['spec'].shape[2]}, {batch['mel'].shape[2]}"

        # compute spectrogram frame lengths
        batch["spec_lens"] = (batch["spec"].shape[2] * batch["waveform_rel_lens"]).int()
        batch["mel_lens"] = (batch["mel"].shape[2] * batch["waveform_rel_lens"]).int()

        if self.args.encoder_sample_rate:
            assert (batch["spec_lens"] - (batch["mel_lens"] / self.interpolate_factor).int()).sum() == 0
        else:
            assert (batch["spec_lens"] - batch["mel_lens"]).sum() == 0

        # zero the padding frames
        batch["spec"] = batch["spec"] * sequence_mask(batch["spec_lens"]).unsqueeze(1)
        batch["mel"] = batch["mel"] * sequence_mask(batch["mel_lens"]).unsqueeze(1)
        return batch

    def get_sampler(self, config: Coqpit, dataset: TTSDataset, num_gpus=1, is_eval=False):
        weights = None
        data_items = dataset.samples
        if getattr(config, "use_weighted_sampler", False):
            for attr_name, alpha in config.weighted_sampler_attrs.items():
                logger.info("Using weighted sampler for attribute '%s' with alpha %.3f", attr_name, alpha)
                multi_dict = config.weighted_sampler_multipliers.get(attr_name, None)
                logger.info(multi_dict)
                weights, attr_names, attr_weights = get_attribute_balancer_weights(
                    attr_name=attr_name, items=data_items, multi_dict=multi_dict
                )
                weights = weights * alpha
                logger.info("Attribute weights for '%s' \n | > %s", attr_names, attr_weights)

        # input_audio_lenghts = [os.path.getsize(x["audio_file"]) for x in data_items]

        if weights is not None:
            w_sampler = WeightedRandomSampler(weights, len(weights))
            batch_sampler = BucketBatchSampler(
                w_sampler,
                data=data_items,
                batch_size=config.eval_batch_size if is_eval else config.batch_size,
                sort_key=lambda x: os.path.getsize(x["audio_file"]),
                drop_last=True,
            )
        else:
            batch_sampler = None
        # sampler for DDP
        if batch_sampler is None:
            batch_sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        else:  # If a sampler is already defined use this sampler and DDP sampler together
            batch_sampler = (
                DistributedSamplerWrapper(batch_sampler) if num_gpus > 1 else batch_sampler
            )  # TODO: check batch_sampler with multi-gpu
        return batch_sampler

    def get_data_loader(
        self,
        config: Coqpit,
        assets: dict,
        is_eval: bool,
        samples: list[dict] | list[list],
        verbose: bool,
        num_gpus: int,
        rank: int | None = None,
    ) -> "DataLoader":
        # init dataloader
        dataset = VitsDataset(
            model_args=self.args,
            samples=samples,
            batch_group_size=0 if is_eval else config.batch_group_size * config.batch_size,
            min_text_len=config.min_text_len,
            max_text_len=config.max_text_len,
            min_audio_len=config.min_audio_len,
            max_audio_len=config.max_audio_len,
            phoneme_cache_path=config.phoneme_cache_path,
            precompute_num_workers=config.precompute_num_workers,
            tokenizer=self.tokenizer,
            start_by_longest=config.start_by_longest,
        )

        # wait all the DDP process to be ready
        if num_gpus > 1:
            dist.barrier()

        # sort input sequences from short to long
        dataset.preprocess_samples()

        # get samplers
        sampler = self.get_sampler(config, dataset, num_gpus)
        if sampler is None:
            loader = DataLoader(
                dataset,
                batch_size=config.eval_batch_size if is_eval else config.batch_size,
                shuffle=False,  # shuffle is done in the dataset.
                collate_fn=dataset.collate_fn,
                drop_last=False,  # setting this False might cause issues in AMP training.
                num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                pin_memory=False,
            )
        else:
            if num_gpus > 1:
                loader = DataLoader(
                    dataset,
                    sampler=sampler,
                    batch_size=config.eval_batch_size if is_eval else config.batch_size,
                    collate_fn=dataset.collate_fn,
                    num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                    pin_memory=False,
                )
            else:
                loader = DataLoader(
                    dataset,
                    batch_sampler=sampler,
                    collate_fn=dataset.collate_fn,
                    num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                    pin_memory=False,
                )
        return loader

    def get_optimizer(self) -> list:
        """Initiate and return the GAN optimizers based on the config parameters.

        It returns 2 optimizers in a list. First one is for the discriminator
        and the second one is for the generator.

        Returns:
            List: optimizers.
        """
        optimizer0 = get_optimizer(self.config.optimizer, self.config.optimizer_params, self.config.lr_disc, self.disc)

        # select generator parameters
        gen_parameters = chain(params for k, params in self.named_parameters() if not k.startswith("disc."))
        optimizer1 = get_optimizer(
            self.config.optimizer, self.config.optimizer_params, self.config.lr_gen, parameters=gen_parameters
        )
        return [optimizer0, optimizer1]

    def get_lr(self) -> list:
        """Set the initial learning rates for each optimizer.

        Returns:
            List: learning rates for each optimizer.
        """
        return [self.config.lr_disc, self.config.lr_gen]

    def get_scheduler(self, optimizer) -> list:
        """Set the schedulers for each optimizer.

        Args:
            optimizer (List[`torch.optim.Optimizer`]): List of optimizers.

        Returns:
            List: Schedulers, one for each optimizer.
        """
        scheduler_D = get_scheduler(self.config.lr_scheduler_disc, self.config.lr_scheduler_disc_params, optimizer[0])
        scheduler_G = get_scheduler(self.config.lr_scheduler_gen, self.config.lr_scheduler_gen_params, optimizer[1])
        return [scheduler_D, scheduler_G]

    def get_criterion(self):
        """Get criterions for each optimizer. The index in the output list matches the optimizer idx used in
        `train_step()`"""
        from TTS.tts.layers.losses import (  # pylint: disable=import-outside-toplevel
            VitsDiscriminatorLoss,
            VitsGeneratorLoss,
        )

        return [VitsDiscriminatorLoss(self.config), VitsGeneratorLoss(self.config)]

    def load_checkpoint(self, config, checkpoint_path, eval=False, strict=True, cache=False):  # pylint: disable=unused-argument, redefined-builtin
        """Load the model checkpoint and setup for training or inference"""
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"), cache=cache)
        # compat band-aid for the pre-trained models to not use the encoder baked into the model
        # TODO: consider baking the speaker encoder into the model and call it from there.
        # as it is probably easier for model distribution.
        state["model"] = {k: v for k, v in state["model"].items() if "speaker_encoder" not in k}

        if self.args.encoder_sample_rate is not None and eval:
            # audio resampler is not used in inference time
            self.audio_resampler = None

        # handle fine-tuning from a checkpoint with additional speakers
        if hasattr(self, "emb_g") and state["model"]["emb_g.weight"].shape != self.emb_g.weight.shape:
            num_new_speakers = self.emb_g.weight.shape[0] - state["model"]["emb_g.weight"].shape[0]
            logger.info("Loading checkpoint with %d additional speakers.", num_new_speakers)
            emb_g = state["model"]["emb_g.weight"]
            new_row = torch.randn(num_new_speakers, emb_g.shape[1])
            emb_g = torch.cat([emb_g, new_row], axis=0)
            state["model"]["emb_g.weight"] = emb_g
        # load the model weights
        self.load_state_dict(state["model"], strict=strict)

        if eval:
            self.eval()
            assert not self.training

    def load_fairseq_checkpoint(self, config, checkpoint_path: Path, eval=False, strict=True):  # pylint: disable=unused-argument, redefined-builtin
        """Load VITS checkpoints released by fairseq here: https://github.com/facebookresearch/fairseq/tree/main/examples/mms
        Performs some changes for compatibility.

        Args:
            config (Coqpit): 🐸TTS model config.
            checkpoint_dir (str): Path to the checkpoint directory.
            eval (bool, optional): Set to True for evaluation. Defaults to False.
        """
        import json

        from TTS.tts.utils.text.cleaners import basic_cleaners, uroman_cleaners

        self.disc = None
        # set paths
        checkpoint_dir = checkpoint_path.parent
        config_file = checkpoint_dir / "config.json"
        vocab_file = checkpoint_dir / "vocab.txt"
        # set config params
        with open(config_file, encoding="utf-8") as f:
            # Load the JSON data as a dictionary
            config_org = json.load(f)
        self.config.audio.sample_rate = config_org["data"]["sampling_rate"]
        is_uroman = config_org["data"]["training_files"].endswith("uroman")
        # set tokenizer
        vocab = FairseqVocab.init_from_vocab_file(vocab_file)
        self.text_encoder.emb = nn.Embedding(vocab.num_chars, config.model_args.hidden_channels)
        self.tokenizer = TTSTokenizer(
            use_phonemes=False,
            text_cleaner=uroman_cleaners if is_uroman else basic_cleaners,
            characters=vocab,
            phonemizer=None,
            add_blank=config_org["data"]["add_blank"],
            use_eos_bos=False,
        )
        # load fairseq checkpoint
        new_chk = rehash_fairseq_vits_checkpoint(checkpoint_path)
        self.load_state_dict(new_chk, strict=strict)
        if eval:
            self.eval()
            assert not self.training

    def export_onnx(self, output_path: str = "coqui_vits.onnx", verbose: bool = True):
        """Export model to ONNX format for inference

        Args:
            output_path (str): Path to save the exported model.
            verbose (bool): Print verbose information. Defaults to True.
        """

        # rollback values
        _forward = self.forward
        disc = None
        if hasattr(self, "disc"):
            disc = self.disc
        training = self.training

        # set export mode
        self.disc = None
        self.eval()

        def onnx_inference(text, text_lengths, scales, sid=None, langid=None):
            noise_scale = scales[0]
            length_scale = scales[1]
            noise_scale_dp = scales[2]
            self.noise_scale = noise_scale
            self.length_scale = length_scale
            self.noise_scale_dp = noise_scale_dp
            return self.inference(
                text,
                aux_input={
                    "x_lengths": text_lengths,
                    "d_vectors": None,
                    "speaker_ids": sid,
                    "language_ids": langid,
                    "durations": None,
                },
            )["model_outputs"]

        self.forward = onnx_inference

        # set dummy inputs
        dummy_input_length = 100
        sequences = torch.randint(low=0, high=2, size=(1, dummy_input_length), dtype=torch.long)
        sequence_lengths = torch.LongTensor([sequences.size(1)])
        scales = torch.FloatTensor([self.inference_noise_scale, self.length_scale, self.inference_noise_scale_dp])
        dummy_input = (sequences, sequence_lengths, scales)
        input_names = ["input", "input_lengths", "scales"]

        if self.num_speakers > 0:
            speaker_id = torch.LongTensor([0])
            dummy_input += (speaker_id,)
            input_names.append("sid")

        if self.language_manager.num_languages > 0 and self.embedded_language_dim > 0:
            language_id = torch.LongTensor([0])
            dummy_input += (language_id,)
            input_names.append("langid")

        # export to ONNX
        torch.onnx.export(
            model=self,
            args=dummy_input,
            opset_version=18,
            f=output_path,
            verbose=verbose,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 1: "phonemes"},
                "input_lengths": {0: "batch_size"},
                "output": {0: "batch_size", 1: "time1", 2: "time2"},
            },
            dynamo=False,
        )

        # rollback
        self.forward = _forward
        if training:
            self.train()
        if disc is not None:
            self.disc = disc

    def load_onnx(self, model_path: str, cuda=False):
        import onnxruntime as ort

        providers = [
            (
                "CPUExecutionProvider"
                if cuda is False
                else ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"})
            )
        ]
        sess_options = ort.SessionOptions()
        self.onnx_sess = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers,
        )

    def inference_onnx(self, x, x_lengths=None, speaker_id=None, language_id=None):
        """ONNX inference"""

        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        if x_lengths is None:
            x_lengths = np.array([x.shape[1]], dtype=np.int64)

        if isinstance(x_lengths, torch.Tensor):
            x_lengths = x_lengths.cpu().numpy()
        scales = np.array(
            [self.inference_noise_scale, self.length_scale, self.inference_noise_scale_dp],
            dtype=np.float32,
        )
        input_params = {"input": x, "input_lengths": x_lengths, "scales": scales}
        if speaker_id is not None:
            input_params["sid"] = torch.tensor([speaker_id]).cpu().numpy()
        if language_id is not None:
            input_params["langid"] = torch.tensor([language_id]).cpu().numpy()

        audio = self.onnx_sess.run(
            ["output"],
            input_params,
        )
        return audio[0][0]


##################################
# VITS CHARACTERS
##################################


class VitsCharacters(BaseCharacters):
    """Characters class for VITs model for compatibility with pre-trained models"""

    def __init__(
        self,
        graphemes: str = _characters,
        punctuations: str = _punctuations,
        pad: str = _pad,
        ipa_characters: str = _phonemes,
    ) -> None:
        if ipa_characters is not None:
            graphemes += ipa_characters
        super().__init__(
            characters=graphemes,
            punctuations=punctuations,
            pad=pad,
            eos=None,
            bos=None,
            blank="<BLNK>",
            is_unique=False,
            is_sorted=True,
        )

    def _create_vocab(self):
        self._vocab = [self.pad, *self.punctuations, *self.characters, self.blank]
        self._char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self._id_to_char = dict(enumerate(self.vocab))

    @staticmethod
    def init_from_config(config: BaseTTSConfig):
        if config.characters is not None:
            return (
                VitsCharacters(
                    graphemes=config.characters["characters"],
                    ipa_characters=config.characters["phonemes"],
                    punctuations=config.characters["punctuations"],
                    pad=config.characters["pad"],
                ),
                config,
            )
        characters = VitsCharacters()
        new_config = replace(config, characters=characters.to_config())
        return characters, new_config

    def to_config(self) -> "CharactersConfig":
        return CharactersConfig(
            characters=self.characters,
            punctuations=self.punctuations,
            pad=self.pad,
            eos=None,
            bos=None,
            blank=self.blank,
            is_unique=False,
            is_sorted=True,
        )


class FairseqVocab(BaseVocabulary):
    @staticmethod
    def init_from_vocab_file(vocab_file: Path) -> "FairseqVocab":
        """Create vocabulary from a Fairseq vocab.txt file."""
        vocab = FairseqVocab(vocab=None)
        with vocab_file.open(encoding="utf-8") as f:
            vocab._vocab = [line.replace("\n", "") for line in f]
        vocab.blank = vocab._vocab[0]
        vocab.pad = " "
        vocab._char_to_id = {s: i for i, s in enumerate(vocab._vocab)}
        vocab._id_to_char = dict(enumerate(vocab._vocab))
        return vocab
