import logging
from typing import Any, cast

import torch
import torch.distributed as dist
from coqpit import Coqpit
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from trainer.torch import DistributedSampler, DistributedSamplerWrapper
from trainer.trainer import Trainer

from TTS.config.shared_configs import ModelArgs
from TTS.model import BaseTrainerModel
from TTS.tts.datasets.dataset import TTSDataset
from TTS.tts.utils.data import get_length_balancer_weights
from TTS.vc.configs.shared_configs import BaseVCConfig

# pylint: skip-file

logger = logging.getLogger(__name__)


class BaseVC(BaseTrainerModel):
    """Base VC class. Every new voice conversion model must inherit this.

    It defines common VC-specific functions on top of the :py:class:`~TTS.model.BaseTrainerModel`.
    """

    MODEL_TYPE = "vc"
    config: BaseVCConfig

    def __init__(self, config: Coqpit) -> None:
        super().__init__()
        self.config = cast(BaseVCConfig, config)
        self._set_model_args()

    def _set_model_args(self) -> None:
        """Set up model args based on the config type (``ModelConfig`` or ``ModelArgs``).

        ``ModelArgs`` has all the fields required to initialize the model architecture.

        ``ModelConfig`` has all the fields required for training, inference and containes ``ModelArgs``.

        If the config is for training with a name like ``*Config``, then the model args are embeded in the
        ``config.model_args``

        If the config is for the model with a name like ``*Args``, then we assign them directly.
        """
        if isinstance(self.config, BaseVCConfig):
            self.args = self.config.model_args
        elif isinstance(self.config, ModelArgs):
            self.args = self.config
        else:
            raise ValueError("config must be either a *Config or *Args")

    def format_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Generic batch formatting for ``VCDataset``.

        You must override this if you use a custom dataset.

        Args:
            batch: [description]

        Returns:
            dict: [description]
        """
        # setup input batch
        text_input = batch["token_id"]
        text_lengths = batch["token_id_lengths"]
        linear_input = batch["linear"]
        mel_input = batch["mel"]
        mel_lengths = batch["mel_lengths"]
        stop_targets = batch["stop_targets"]
        item_idx = batch["item_idxs"]
        d_vectors = batch["d_vectors"]
        attn_mask = batch["attns"]
        waveform = batch["waveform"]
        pitch = batch["pitch"]
        energy = batch["energy"]
        max_text_length = torch.max(text_lengths.float())
        max_spec_length = torch.max(mel_lengths.float())

        # compute durations from attention masks
        durations = None
        if attn_mask is not None:
            durations = torch.zeros(attn_mask.shape[0], attn_mask.shape[2])
            for idx, am in enumerate(attn_mask):
                # compute raw durations
                c_idxs = am[:, : text_lengths[idx], : mel_lengths[idx]].max(1)[1]
                # c_idxs, counts = torch.unique_consecutive(c_idxs, return_counts=True)
                c_idxs, counts = torch.unique(c_idxs, return_counts=True)
                dur = torch.ones([text_lengths[idx]]).to(counts.dtype)
                dur[c_idxs] = counts
                # smooth the durations and set any 0 duration to 1
                # by cutting off from the largest duration indeces.
                extra_frames = dur.sum() - mel_lengths[idx]
                largest_idxs = torch.argsort(-dur)[:extra_frames]
                dur[largest_idxs] -= 1
                assert dur.sum() == mel_lengths[idx], (
                    f" [!] total duration {dur.sum()} vs spectrogram length {mel_lengths[idx]}"
                )
                durations[idx, : text_lengths[idx]] = dur

        # set stop targets wrt reduction factor
        stop_targets = stop_targets.view(text_input.shape[0], stop_targets.size(1) // self.config.r, -1)
        stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze(2)
        stop_target_lengths = torch.divide(mel_lengths, self.config.r).ceil_()

        return {
            "text_input": text_input,
            "text_lengths": text_lengths,
            "mel_input": mel_input,
            "mel_lengths": mel_lengths,
            "linear_input": linear_input,
            "stop_targets": stop_targets,
            "stop_target_lengths": stop_target_lengths,
            "attn_mask": attn_mask,
            "durations": durations,
            "d_vectors": d_vectors,
            "max_text_length": float(max_text_length),
            "max_spec_length": float(max_spec_length),
            "item_idx": item_idx,
            "waveform": waveform,
            "pitch": pitch,
            "energy": energy,
            "audio_unique_names": batch["audio_unique_names"],
        }

    def get_sampler(self, config: Coqpit, dataset: TTSDataset, num_gpus: int = 1):
        weights = None
        data_items = dataset.samples

        if getattr(config, "use_length_weighted_sampler", False):
            alpha = getattr(config, "length_weighted_sampler_alpha", 1.0)
            logger.info("Using Length weighted sampler with alpha: %.2f", alpha)
            if weights is not None:
                weights += get_length_balancer_weights(data_items) * alpha
            else:
                weights = get_length_balancer_weights(data_items) * alpha

        if weights is not None:
            sampler = WeightedRandomSampler(weights, len(weights))
        else:
            sampler = None

        # sampler for DDP
        if sampler is None:
            sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        else:  # If a sampler is already defined use this sampler and DDP sampler together
            sampler = DistributedSamplerWrapper(sampler) if num_gpus > 1 else sampler

        return sampler

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
        dataset = TTSDataset(
            outputs_per_step=config.r if "r" in config else 1,
            compute_linear_spec=config.model.lower() == "tacotron" or config.compute_linear_spec,
            compute_f0=config.get("compute_f0", False),
            f0_cache_path=config.get("f0_cache_path", None),
            compute_energy=config.get("compute_energy", False),
            energy_cache_path=config.get("energy_cache_path", None),
            samples=samples,
            return_wav=config.return_wav if "return_wav" in config else False,
            batch_group_size=0 if is_eval else config.batch_group_size * config.batch_size,
            min_text_len=config.min_text_len,
            max_text_len=config.max_text_len,
            min_audio_len=config.min_audio_len,
            max_audio_len=config.max_audio_len,
            phoneme_cache_path=config.phoneme_cache_path,
            precompute_num_workers=config.precompute_num_workers,
            use_noise_augment=False if is_eval else config.use_noise_augment,
            tokenizer=None,
            start_by_longest=config.start_by_longest,
        )

        # wait all the DDP process to be ready
        if num_gpus > 1:
            dist.barrier()

        # sort input sequences from short to long
        dataset.preprocess_samples()

        # get samplers
        sampler = self.get_sampler(config, dataset, num_gpus)

        return DataLoader(
            dataset,
            batch_size=config.eval_batch_size if is_eval else config.batch_size,
            shuffle=config.shuffle if sampler is None else False,  # if there is no other sampler
            collate_fn=dataset.collate_fn,
            drop_last=config.drop_last,  # setting this False might cause issues in AMP training.
            sampler=sampler,
            num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
            pin_memory=False,
        )

    def test_run(self, assets: dict) -> tuple[dict, dict]:
        """Generic test run for ``vc`` models used by ``Trainer``.

        You can override this for a different behaviour.

        Args:
            assets (dict): A dict of training assets. For ``vc`` models, it must include ``{'audio_processor': ap}``.

        Returns:
            tuple[dict, dict]: Test figures and audios to be projected to Tensorboard.
        """
        raise NotImplementedError

    def on_init_start(self, trainer: Trainer) -> None:
        """Run before training starts."""
