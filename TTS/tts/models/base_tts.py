import logging
import os
import random
from typing import Any

import torch
import torch.distributed as dist
from coqpit import Coqpit
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from trainer.logging.base_dash_logger import BaseDashboardLogger
from trainer.torch import DistributedSampler, DistributedSamplerWrapper

from TTS.config import get_from_config_or_model_args
from TTS.model import BaseTrainerModel
from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.tts.datasets.dataset import TTSDataset
from TTS.tts.utils.data import get_length_balancer_weights
from TTS.tts.utils.languages import LanguageManager, get_language_balancer_weights
from TTS.tts.utils.speakers import SpeakerManager, get_speaker_balancer_weights
from TTS.tts.utils.synthesis import inv_spectrogram
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram
from TTS.utils.generic_utils import warn_synthesize_config_deprecated, warn_synthesize_speaker_id_deprecated
from TTS.utils.voices import CloningMixin

logger = logging.getLogger(__name__)

# pylint: skip-file


class BaseTTS(CloningMixin, BaseTrainerModel):
    """Base `tts` class. Every new `tts` model must inherit this.

    It defines common `tts` specific functions on top of `Model` implementation.
    """

    MODEL_TYPE = "tts"

    def __init__(
        self,
        config: Coqpit,
        ap: "AudioProcessor",
        tokenizer: "TTSTokenizer",
        speaker_manager: SpeakerManager | None = None,
        language_manager: LanguageManager | None = None,
    ):
        super().__init__()
        self.config = config
        self.ap = ap
        self.tokenizer = tokenizer
        self.speaker_manager = speaker_manager
        self.language_manager = language_manager
        self._set_model_args(config)

    def _set_model_args(self, config: Coqpit):
        """Setup model args based on the config type (`ModelConfig` or `ModelArgs`).

        `ModelArgs` has all the fields required to initialize the model architecture.

        `ModelConfig` has all the fields required for training, inference and contains `ModelArgs`.

        If the config is for training with a name like "*Config", then the model args are embeded in the
        config.model_args

        If the config is for the model with a name like "*Args", then we assign them directly.
        """
        # don't use isinstance not to import recursively
        if "Config" in config.__class__.__name__:
            config_num_chars = (
                self.config.model_args.num_chars if self.config.model_args is not None else self.config.num_chars
            )
            num_chars = config_num_chars if self.tokenizer is None else self.tokenizer.characters.num_chars
            if "characters" in config:
                self.config.num_chars = num_chars
                if self.config.model_args is not None:
                    config.model_args.num_chars = num_chars
                    self.args = self.config.model_args
            else:
                self.config = config
                self.args = config.model_args
        elif "Args" in config.__class__.__name__:
            self.args = config
        else:
            raise ValueError("config must be either a *Config or *Args")

    def init_multispeaker(self, config: Coqpit, data: list = None):
        """Set up for multi-speaker TTS.

        Initialize a speaker embedding layer if needed and define expected embedding
        channel size for defining `in_channels` size of the connected layers.

        This implementation yields 3 possible outcomes:

        1. If `config.use_speaker_embedding` and `config.use_d_vector_file` are False, do nothing.
        2. If `config.use_d_vector_file` is True, set expected embedding channel size to `config.d_vector_dim` or 512.
        3. If `config.use_speaker_embedding`, initialize a speaker embedding layer with channel size of
           `config.d_vector_dim` or 512.

        You can override this function for new models.

        Args:
            config (Coqpit): Model configuration.
        """
        # set number of speakers
        if self.speaker_manager is not None:
            self.num_speakers = self.speaker_manager.num_speakers
        elif hasattr(config, "num_speakers"):
            self.num_speakers = config.num_speakers

        # set ultimate speaker embedding size
        if config.use_speaker_embedding or config.use_d_vector_file:
            self.embedded_speaker_dim = (
                config.d_vector_dim if "d_vector_dim" in config and config.d_vector_dim is not None else 512
            )
        # init speaker embedding layer
        if config.use_speaker_embedding and not config.use_d_vector_file:
            logger.info("Init speaker_embedding layer.")
            self.speaker_embedding = nn.Embedding(self.num_speakers, self.embedded_speaker_dim)
            self.speaker_embedding.weight.data.normal_(0, 0.3)

    def get_aux_input_from_test_sentences(self, sentence_info: str | list[str]) -> dict[str, Any]:
        config = self.config.model_args if self.config.model_args is not None else self.config

        # extract speaker and language info
        text, speaker, style_wav, language = None, None, None, None

        if isinstance(sentence_info, list):
            if len(sentence_info) == 1:
                text = sentence_info[0]
            elif len(sentence_info) == 2:
                text, speaker = sentence_info
            elif len(sentence_info) == 3:
                text, speaker, style_wav = sentence_info
            elif len(sentence_info) == 4:
                text, speaker, style_wav, language = sentence_info
        else:
            text = sentence_info

        if speaker is None and self.speaker_manager is not None:
            speaker = random.sample(self.speaker_manager.speaker_names, 1)[0]

        return {
            "text": text,
            "speaker": speaker,
            "style_wav": style_wav,
            "language": language,
        }

    def format_batch(self, batch: dict) -> dict:
        """Generic batch formatting for `TTSDataset`.

        You must override this if you use a custom dataset.

        Args:
            batch (Dict): [description]

        Returns:
            Dict: [description]
        """
        # setup input batch
        text_input = batch["token_id"]
        text_lengths = batch["token_id_lengths"]
        speaker_names = batch["speaker_names"]
        linear_input = batch["linear"]
        mel_input = batch["mel"]
        mel_lengths = batch["mel_lengths"]
        stop_targets = batch["stop_targets"]
        item_idx = batch["item_idxs"]
        d_vectors = batch["d_vectors"]
        speaker_ids = batch["speaker_ids"]
        attn_mask = batch["attns"]
        waveform = batch["waveform"]
        pitch = batch["pitch"]
        energy = batch["energy"]
        language_ids = batch["language_ids"]
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
            "speaker_names": speaker_names,
            "mel_input": mel_input,
            "mel_lengths": mel_lengths,
            "linear_input": linear_input,
            "stop_targets": stop_targets,
            "stop_target_lengths": stop_target_lengths,
            "attn_mask": attn_mask,
            "durations": durations,
            "speaker_ids": speaker_ids,
            "d_vectors": d_vectors,
            "max_text_length": float(max_text_length),
            "max_spec_length": float(max_spec_length),
            "item_idx": item_idx,
            "waveform": waveform,
            "pitch": pitch,
            "energy": energy,
            "language_ids": language_ids,
            "audio_unique_names": batch["audio_unique_names"],
        }

    def get_sampler(self, config: Coqpit, dataset: TTSDataset, num_gpus=1):
        weights = None
        data_items = dataset.samples

        if getattr(config, "use_language_weighted_sampler", False):
            alpha = getattr(config, "language_weighted_sampler_alpha", 1.0)
            logger.info("Using Language weighted sampler with alpha: %.2f", alpha)
            weights = get_language_balancer_weights(data_items) * alpha

        if getattr(config, "use_speaker_weighted_sampler", False):
            alpha = getattr(config, "speaker_weighted_sampler_alpha", 1.0)
            logger.info("Using Speaker weighted sampler with alpha: %.2f", alpha)
            if weights is not None:
                weights += get_speaker_balancer_weights(data_items) * alpha
            else:
                weights = get_speaker_balancer_weights(data_items) * alpha

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
        if is_eval and not config.run_eval:
            loader = None
        else:
            # setup multi-speaker attributes
            if self.speaker_manager is not None:
                if config.model_args is not None:
                    speaker_id_mapping = (
                        self.speaker_manager.name_to_id if config.model_args.use_speaker_embedding else None
                    )
                    d_vector_mapping = self.speaker_manager.embeddings if config.model_args.use_d_vector_file else None
                    config.use_d_vector_file = config.model_args.use_d_vector_file
                else:
                    speaker_id_mapping = self.speaker_manager.name_to_id if config.use_speaker_embedding else None
                    d_vector_mapping = self.speaker_manager.embeddings if config.use_d_vector_file else None
            else:
                speaker_id_mapping = None
                d_vector_mapping = None

            # setup multi-lingual attributes
            if self.language_manager is not None:
                language_id_mapping = self.language_manager.name_to_id if self.args.use_language_embedding else None
            else:
                language_id_mapping = None

            # init dataloader
            dataset = TTSDataset(
                outputs_per_step=config.r if "r" in config else 1,
                compute_linear_spec=config.model.lower() == "tacotron" or config.compute_linear_spec,
                compute_f0=config.get("compute_f0", False),
                f0_cache_path=config.get("f0_cache_path", None),
                compute_energy=config.get("compute_energy", False),
                energy_cache_path=config.get("energy_cache_path", None),
                samples=samples,
                ap=self.ap,
                return_wav=config.return_wav if "return_wav" in config else False,
                batch_group_size=0 if is_eval else config.batch_group_size * config.batch_size,
                min_text_len=config.min_text_len,
                max_text_len=config.max_text_len,
                min_audio_len=config.min_audio_len,
                max_audio_len=config.max_audio_len,
                phoneme_cache_path=config.phoneme_cache_path,
                precompute_num_workers=config.precompute_num_workers,
                use_noise_augment=False if is_eval else config.use_noise_augment,
                speaker_id_mapping=speaker_id_mapping,
                d_vector_mapping=d_vector_mapping if config.use_d_vector_file else None,
                tokenizer=self.tokenizer,
                start_by_longest=config.start_by_longest,
                language_id_mapping=language_id_mapping,
            )

            # wait all the DDP process to be ready
            if num_gpus > 1:
                dist.barrier()

            # sort input sequences from short to long
            dataset.preprocess_samples()

            # get samplers
            sampler = self.get_sampler(config, dataset, num_gpus)

            loader = DataLoader(
                dataset,
                batch_size=config.eval_batch_size if is_eval else config.batch_size,
                shuffle=config.shuffle if sampler is None else False,  # if there is no other sampler
                collate_fn=dataset.collate_fn,
                drop_last=config.drop_last,  # setting this False might cause issues in AMP training.
                sampler=sampler,
                num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                pin_memory=False,
            )
        return loader

    def _create_logs(
        self, batch: dict[str, Any], outputs: dict[str, Any] | list[dict[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        raise NotImplementedError

    @torch.inference_mode()
    def train_log(
        self,
        batch: dict[str, Any],
        outputs: dict[str, Any] | list[dict[str, Any]],
        logger: BaseDashboardLogger,
        assets: dict[str, Any],
        steps: int,
    ) -> None:
        """Create visualizations and waveform examples.

        For example, here you can plot spectrograms and generate sample
        waveforms from these spectrograms to be projected onto Tensorboard.

        Args:
            batch: Model inputs used at the previous training step.
            outputs: Model outputs generated at the previous training step.
            logger: Logger instance.
            assets: Training assets.
        """
        figures, audios = self._create_logs(batch, outputs)
        logger.train_figures(steps, figures)
        logger.train_audios(steps, audios, self.ap.sample_rate)

    @torch.inference_mode()
    def eval_step(
        self, batch: dict[str, Any], criterion: nn.Module, optimizer_idx: int | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Perform a single evaluation step.

        Run the model forward ... and compute losses. In most cases, you can
        call `train_step()` with no changes.

        Args:
            batch: Input tensors.
            criterion: Loss layer designed for the model.
            optimizer_idx: Index of optimizer to use. 0 for the generator and 1 for the discriminator networks.

        Returns:
            Tuple[Dict, Dict]: Model outputs and computed losses.
        """
        if optimizer_idx is not None:
            return self.train_step(batch, criterion, optimizer_idx)
        return self.train_step(batch, criterion)

    @torch.inference_mode()
    def eval_log(
        self,
        batch: dict[str, Any],
        outputs: dict[str, Any] | list[dict[str, Any]],
        logger: BaseDashboardLogger,
        assets: dict[str, Any],
        steps: int,
    ) -> None:
        figures, audios = self._create_logs(batch, outputs)
        logger.eval_figures(steps, figures)
        logger.eval_audios(steps, audios, self.ap.sample_rate)

    @torch.inference_mode()
    def test_run(self, assets: dict) -> dict[str, Any]:
        """Generic test run for `tts` models used by `Trainer`.

        You can override this for a different behaviour.

        Args:
            assets (dict): A dict of training assets. For `tts` models, it must include `{'audio_processor': ap}`.

        Returns:
            Dictionary with test figures and audios to be projected to Tensorboard.
        """
        logger.info("Synthesizing test sentences.")
        test_audios = {}
        test_figures = {}
        test_sentences = self.config.test_sentences
        if len(test_sentences) == 0:
            logger.warning("No test sentences provided.")
        for idx, sen in enumerate(test_sentences):
            aux_inputs = self.get_aux_input_from_test_sentences(sen)
            # TODO: pass style_wav if needed
            outputs = self.synthesize(
                aux_inputs["text"],
                speaker=aux_inputs.get("speaker", None),
                language=aux_inputs.get("language", None),
                use_griffin_lim=True,
            )
            test_audios[f"{idx}-audio"] = outputs["wav"]
            test_figures[f"{idx}-prediction"] = plot_spectrogram(
                outputs["outputs"]["model_outputs"], self.ap, output_fig=False
            )
            test_figures[f"{idx}-alignment"] = plot_alignment(outputs["alignments"], output_fig=False)
        return {"figures": test_figures, "audios": test_audios}

    def test_log(
        self,
        outputs: dict[str, Any],
        logger: "Logger",
        assets: dict,
        steps: int,  # pylint: disable=unused-argument
    ) -> None:
        logger.test_audios(steps, outputs["audios"], self.ap.sample_rate)
        if "figures" in outputs:
            logger.test_figures(steps, outputs["figures"])

    def on_init_start(self, trainer):
        """Save the speaker.pth and language_ids.json at the beginning of the training. Also update both paths."""
        if self.speaker_manager is not None:
            output_path = os.path.join(trainer.output_path, "speakers.pth")
            self.speaker_manager.save_ids_to_file(output_path)
            trainer.config.speakers_file = output_path
            # some models don't have `model_args` set
            if getattr(trainer.config, "model_args", None) is not None:
                trainer.config.model_args.speakers_file = output_path
            trainer.config.save_json(os.path.join(trainer.output_path, "config.json"))
            logger.info("`speakers.pth` is saved to: %s", output_path)
            logger.info("`speakers_file` is updated in the config.json.")

        if self.language_manager is not None:
            output_path = os.path.join(trainer.output_path, "language_ids.json")
            self.language_manager.save_ids_to_file(output_path)
            trainer.config.language_ids_file = output_path
            if getattr(trainer.config, "model_args", None) is not None:
                trainer.config.model_args.language_ids_file = output_path
            trainer.config.save_json(os.path.join(trainer.output_path, "config.json"))
            logger.info("`language_ids.json` is saved to: %s", output_path)
            logger.info("`language_ids_file` is updated in the config.json.")

    def _get_language_id(self, language: str | None) -> int | None:
        if self.language_manager is not None:
            if len(self.language_manager.name_to_id) == 1:
                return list(self.language_manager.name_to_id.values())[0]
            if language is not None:
                try:
                    return self.language_manager.name_to_id[language]
                except KeyError as e:
                    msg = (
                        f"Looks like you use a multi-lingual model. "
                        f"Language {language} is not among the available languages: "
                        f"{self.language_manager.name_to_id.keys()}."
                    )
                    raise ValueError(msg) from e
            msg = "Looks like you use a multi-lingual model, but did not specify a language. "
            raise ValueError(msg)
        return None

    def _get_speaker_id_or_dvector(
        self,
        speaker: str | None,
        speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]] | None = None,
        voice_dir: str | os.PathLike[Any] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return speaker ID or d-vector, depending on the model.

        Considers the following cases:
        - Return speaker ID for embedding-based models.
        - Return d-vector for d-vector-based models with preset speakers.
        - Compute d-vector from `speaker_wav` if model has a speaker encoder.
          The result may be cached in `voice_dir` if a custom `speaker` name is specified.

        Returns:
            Tuple of (speaker id, d-vector), one of which is None, depending on the model.
        """
        if self.speaker_manager is None:
            return None, None

        if len(self.speaker_manager.name_to_id) == 1:
            speaker_id = list(self.speaker_manager.name_to_id.values())[0]
            return torch.tensor(speaker_id, device=self.device), None

        speaker_exists = True
        if get_from_config_or_model_args(self.config, "use_d_vector_file") and speaker is not None:
            if speaker in self.speaker_manager.embedding_names:
                d_vector = self.speaker_manager.get_mean_embedding(speaker, num_samples=None, randomize=False)
                d_vector = torch.tensor(d_vector, dtype=torch.float, device=self.device).unsqueeze(0)
                return None, d_vector  # [1 x embedding_dim]
            speaker_exists = False

        if get_from_config_or_model_args(self.config, "use_speaker_embedding") and speaker is not None:
            if speaker in self.speaker_manager.name_to_id:
                speaker_id = self.speaker_manager.name_to_id[speaker]
                return torch.tensor(speaker_id, device=self.device), None
            speaker_exists = False

        if self.speaker_manager.encoder is not None and (speaker is not None or speaker_wav is not None):
            d_vector = self.clone_voice(speaker_wav, speaker, voice_dir)["d_vector"]
            return None, torch.tensor(d_vector, dtype=torch.float, device=self.device).unsqueeze(0)

        if not speaker_exists:
            msg = f"{speaker} is not a valid speaker of the model."
            raise KeyError(msg)

        msg = (
            "Looks like you are using a multi-speaker model. "
            "You need to pass either a speaker name or a reference audio file."
        )
        raise ValueError(msg)

    def _clone_voice(
        self, speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]], **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        d_vector = self.speaker_manager.compute_embedding_from_clip(speaker_wav)
        voice = {"d_vector": d_vector}
        metadata = {"name": self.speaker_manager.encoder.__class__.__name__}
        return voice, metadata

    def synthesize(
        self,
        text: str,
        config: BaseTTSConfig | None = None,
        *,
        speaker: str | None = None,
        speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]] | None = None,
        voice_dir: str | os.PathLike[Any] | None = None,
        language: str | None = None,
        use_griffin_lim: bool = False,
        do_trim_silence: bool = False,
        extra_aux_input: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Synthesize speech for the given text.

        Use Griffin-Lim vocoder or just compute output features to be passed to
        the vocoder model.

        Args:
            text: Input text to convert to speech.
            config: DEPRECATED. Not used.
            speaker: Speaker name (for multi-speaker models).
            speaker_wav: Path(s) to reference audio (for models with voice cloning).
            voice_dir: Cache folder for cloned voices.
            language: Language name (for multilingual models).
            use_griffin_lim: Vocode with the Griffin-Lim algorithm.
            do_trim_silence: Trim silence after synthesis.
            extra_aux_input: Arguments added to aux_input in the inference() call.
            **kwargs: Arguments passed on to model-specific inference() methods.
        """
        if config is not None:
            warn_synthesize_config_deprecated()
        if (speaker_id := kwargs.pop("speaker_id", None)) is not None:
            speaker = speaker_id
            warn_synthesize_speaker_id_deprecated()
        text_inputs = self.tokenizer.text_to_ids(text, language=language)
        language_id = self._get_language_id(language)
        _speaker_id, d_vector = self._get_speaker_id_or_dvector(speaker, speaker_wav, voice_dir)
        text_inputs = torch.as_tensor(text_inputs, dtype=torch.long, device=self.device).unsqueeze(0)
        if language_id is not None:
            language_id = torch.tensor(language_id, device=self.device)

        if extra_aux_input is None:
            extra_aux_input = {}
        outputs = self.inference(
            text_inputs,
            aux_input={
                "x_lengths": torch.tensor(text_inputs.shape[1:2], device=self.device),
                "speaker_ids": _speaker_id,
                "d_vectors": d_vector,
                "language_ids": language_id,
                **extra_aux_input,
            },
        )
        model_outputs = outputs["model_outputs"]
        model_outputs = model_outputs[0].detach().cpu().numpy().squeeze()
        alignments = outputs["alignments"]

        wav = None
        if model_outputs.ndim == 2:  # [T, C_spec]
            if use_griffin_lim:
                wav = inv_spectrogram(model_outputs, self.ap, self.config)
                if do_trim_silence:
                    wav = wav[: self.ap.find_endpoint(wav)]
        else:  # [T,]
            wav = model_outputs
        return {
            "wav": wav,
            "alignments": alignments,
            "text_inputs": text_inputs,
            "outputs": outputs,
        }


class BaseTTSE2E(BaseTTS):
    def _set_model_args(self, config: Coqpit):
        self.config = config
        if "Config" in config.__class__.__name__:
            num_chars = (
                self.config.model_args.num_chars if self.tokenizer is None else self.tokenizer.characters.num_chars
            )
            self.config.model_args.num_chars = num_chars
            self.config.num_chars = num_chars
            self.args = config.model_args
            self.args.num_chars = num_chars
        elif "Args" in config.__class__.__name__:
            self.args = config
            self.args.num_chars = self.args.num_chars
        else:
            raise ValueError("config must be either a *Config or *Args")
