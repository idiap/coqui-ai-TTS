import logging
import os
from typing import Any, TypeAlias

import torch
import torch.nn.functional as F
import torchaudio
from coqpit import Coqpit

from TTS.utils.voices import CloningMixin
from TTS.vc.configs.knnvc_config import KNNVCConfig
from TTS.vc.layers.freevc.wavlm import get_wavlm
from TTS.vc.models.base_vc import BaseVC

logger = logging.getLogger(__name__)

PathOrTensor: TypeAlias = str | os.PathLike[Any] | torch.Tensor


class KNNVC(CloningMixin, BaseVC):
    """
    Paper::
        https://arxiv.org/abs/2305.18975

    Paper Abstract::
        Any-to-any voice conversion aims to transform source speech
        into a target voice with just a few examples of the target speaker as a
        reference. Recent methods produce convincing conversions, but at the cost of
        increased complexity -- making results difficult to reproduce and build on.
        Instead, we keep it simple. We propose k-nearest neighbors voice conversion
        (kNN-VC): a straightforward yet effective method for any-to-any conversion.
        First, we extract self-supervised representations of the source and reference
        speech. To convert to the target speaker, we replace each frame of the source
        representation with its nearest neighbor in the reference. Finally, a pretrained
        vocoder synthesizes audio from the converted representation. Objective and
        subjective evaluations show that kNN-VC improves speaker similarity with similar
        intelligibility scores to existing methods.

    Samples::
        https://bshall.github.io/knn-vc

    Original code::
        https://github.com/bshall/knn-vc

    Examples:
        >>> from TTS.vc.configs.knnvc_config import KNNVCConfig
        >>> from TTS.vc.models.knnvc import KNNVC
        >>> config = KNNVCConfig()
        >>> model = KNNVC(config)
    """

    def __init__(self, config: Coqpit):
        super().__init__(config)
        self.ssl_dim = self.args.ssl_dim
        self.wavlm = get_wavlm()

    @staticmethod
    def init_from_config(config: KNNVCConfig) -> "KNNVC":
        return KNNVC(config)

    @torch.inference_mode()
    def get_features(self, audio: PathOrTensor, vad_trigger_level=0) -> torch.Tensor:
        """Return features for the given waveform with output shape (seq_len, dim).

        Optionally perform VAD trimming on start/end with `vad_trigger_level`.
        """
        # load audio
        if isinstance(audio, torch.Tensor):
            x: torch.Tensor = audio
            sr = self.config.audio.sample_rate
            if x.dim() == 1:
                x = x[None]
        else:
            x, sr = torchaudio.load(audio, normalize=True)

        if not sr == self.config.audio.sample_rate:
            logger.info("Resampling %d to %d in %s", sr, self.config.audio.sample_rate, audio)
            x = torchaudio.functional.resample(x, orig_freq=sr, new_freq=self.config.audio.sample_rate)
            sr = self.config.audio.sample_rate

        # trim silence from front and back
        if vad_trigger_level > 1e-3:
            transform = torchaudio.transforms.Vad(sample_rate=sr, trigger_level=vad_trigger_level)
            x_front_trim = transform(x)
            waveform_reversed = torch.flip(x_front_trim, (-1,))
            waveform_reversed_front_trim = transform(waveform_reversed)
            x = torch.flip(waveform_reversed_front_trim, (-1,))

        # extract the representation of each layer
        wav_input_16khz = x.to(self.device)
        features = self.wavlm.extract_features(
            wav_input_16khz, output_layer=self.config.wavlm_layer, ret_layer_results=False
        )[0]
        return features.squeeze(0)

    def get_matching_set(self, wavs: list[PathOrTensor], vad_trigger_level=7) -> torch.Tensor:
        """Get concatenated wavlm features for the matching set using all waveforms in `wavs`.

        Wavs are specified as either a list of paths or list of loaded waveform tensors of
        shape (channels, T), assumed to be of 16kHz sample rate.
        """
        feats = []
        for p in wavs:
            feats.append(self.get_features(p, vad_trigger_level=vad_trigger_level))

        feats = torch.concat(feats, dim=0).cpu()
        return feats

    @staticmethod
    def fast_cosine_dist(source_feats: torch.Tensor, matching_pool: torch.Tensor) -> torch.Tensor:
        """Like torch.cdist, but fixed dim=-1 and for cosine distance."""
        source_norms = torch.norm(source_feats, p=2, dim=-1)
        matching_norms = torch.norm(matching_pool, p=2, dim=-1)
        dotprod = (
            -(torch.cdist(source_feats[None], matching_pool[None], p=2)[0] ** 2)
            + source_norms[:, None] ** 2
            + matching_norms[None] ** 2
        )
        dotprod /= 2

        dists = 1 - (dotprod / (source_norms[:, None] * matching_norms[None]))
        return dists

    @torch.inference_mode()
    def match(
        self,
        query_seq: torch.Tensor,
        matching_set: torch.Tensor,
        synth_set: torch.Tensor | None = None,
        topk: int | None = None,
        target_duration: float | None = None,
    ) -> torch.Tensor:
        """Given `query_seq`, `matching_set`, and `synth_set` tensors of shape (N, dim), perform kNN regression matching
        with k=`topk`.

        Args:
            `query_seq`: Tensor (N1, dim) of the input/source query features.
            `matching_set`: Tensor (N2, dim) of the matching set used as the 'training set' for the kNN algorithm.
            `synth_set`: optional Tensor (N2, dim) corresponding to the matching set. We use the matching set to assign
                         each query vector to a vector in the matching set, and then use the corresponding vector from
                         the synth set during HiFiGAN synthesis.
                         By default, and for best performance, this should be identical to the matching set.
            `topk`: k in the kNN -- the number of nearest neighbors to average over.
            `target_duration`: if set to a float, interpolate waveform duration to be equal to this value in seconds.

        Returns:
            - converted features (1, N, dim)
        """
        if topk is None:
            topk = self.config.topk
        synth_set = matching_set.to(self.device) if synth_set is None else synth_set.to(self.device)
        matching_set = matching_set.to(self.device)
        query_seq = query_seq.to(self.device)

        if target_duration is not None:
            target_samples = int(target_duration * self.config.audio.sample_rate)
            scale_factor = (target_samples / self.hop_length) / query_seq.shape[0]  # n_targ_feats / n_input_feats
            query_seq = F.interpolate(query_seq.T[None], scale_factor=scale_factor, mode="linear")[0].T

        dists = self.fast_cosine_dist(query_seq, matching_set)
        best = dists.topk(k=topk, largest=False, dim=-1)
        out_feats = synth_set[best.indices].mean(dim=1)
        return out_feats.unsqueeze(0)

    def load_checkpoint(
        self,
        config: Coqpit,
        checkpoint_path: str | os.PathLike[Any],
        *,
        eval: bool = False,
        strict: bool = True,
        cache: bool = False,
    ) -> None:
        """kNN-VC does not use checkpoints."""

    def forward(self) -> None: ...
    def inference(self) -> None: ...

    def _clone_voice(
        self, speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]], **generate_kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not isinstance(speaker_wav, list):
            speaker_wav = [speaker_wav]
        matching_set = self.get_matching_set(speaker_wav)
        voice = {"matching_set": matching_set}
        metadata = {"name": "wavlm", "layer": self.config.wavlm_layer}
        return voice, metadata

    @torch.inference_mode()
    def voice_conversion(
        self,
        source: PathOrTensor,
        target: list[PathOrTensor] | None = None,
        *,
        speaker_id: str | None = None,
        voice_dir: str | os.PathLike[Any] | None = None,
        topk: int | None = None,
    ) -> torch.Tensor:
        if target is not None and not isinstance(target, list):
            target = [target]
        source_features = self.get_features(source)

        if target is None or all(isinstance(x, (str, os.PathLike)) for x in target):
            voice = self.clone_voice(target, speaker_id, voice_dir)
            matching_set = voice["matching_set"]
        else:
            matching_set = self.get_matching_set(target)
        return self.match(source_features, matching_set, topk=topk)
