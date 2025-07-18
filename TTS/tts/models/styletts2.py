"""
StyleTTS 2 implementation.

Based on the paper: "StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion
and Adversarial Training with Large Speech Language Models"

Adapted from the original code: https://github.com/yl4579/StyleTTS2
"""

import logging
import os
from typing import Any

import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from coqpit import Coqpit
from torch.utils.data import DataLoader

from TTS.tts.configs.shared_configs import BaseTTSConfig, CharactersConfig
from TTS.tts.configs.styletts2_config import StyleTTS2Config
from TTS.tts.layers.styletts2.decoder import StyleTTSDecoder
from TTS.tts.layers.styletts2.diffusion import (
    ADPM2Sampler,
    AudioDiffusionConditional,
    DiffusionSampler,
    KarrasSchedule,
    KDiffusion,
    LogNormalDistribution,
    StyleTransformer1d,
    Transformer1d,
)
from TTS.tts.layers.styletts2.discriminator import SLMDiscriminator
from TTS.tts.layers.styletts2.duration_predictor import DifferentiableDurationPredictor
from TTS.tts.layers.styletts2.prosody_predictor import ProsodyPredictor
from TTS.tts.layers.styletts2.style_encoders import StyleEncoder
from TTS.tts.layers.styletts2.text_encoders import AcousticTextEncoder, ProsodicTextEncoder
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.text.characters import BaseCharacters
from TTS.utils.generic_utils import warn_synthesize_config_deprecated, warn_synthesize_speaker_id_deprecated

logger = logging.getLogger(__name__)


class StyleTTS2(BaseTTS):
    """StyleTTS 2 model implementation.

    This model implements the StyleTTS 2 architecture with style diffusion and SLM discriminators
    for human-level text-to-speech synthesis.

    Paper: "StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and
           Adversarial Training with Large Speech Language Models"
    """

    def __init__(self, config: StyleTTS2Config):
        super().__init__(config)
        self._init_model_components()

        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=self.config.audio.num_mels,
            n_fft=self.config.audio.fft_size,
            win_length=self.config.audio.win_length,
            hop_length=self.config.audio.hop_length,
        )

        self.training_stage = 1  # 1 for acoustic pre-training, 2 for joint training

    def _init_model_components(self):
        """Initialize all model components."""
        self.acoustic_text_encoder = AcousticTextEncoder(
            num_chars=self.args.num_chars,
            hidden_channels=self.args.hidden_channels,
            num_layers=self.args.acoustic_text_encoder_layers,
        )

        self.prosodic_text_encoder = ProsodicTextEncoder(
            plbert_config=self.args.plbert_config,
            hidden_channels=self.args.hidden_channels,
        )

        self.acoustic_style_encoder = StyleEncoder(
            style_dim=self.args.style_dim,
            max_conv_dim=self.args.hidden_channels,
        )

        self.prosodic_style_encoder = StyleEncoder(
            style_dim=self.args.style_dim,
            max_conv_dim=self.args.hidden_channels,
        )

        transformer_args = {
            "channels": self.args.style_dim * 2,
            "context_embedding_features": self.args.plbert_config.hidden_size,
            "num_layers": self.args.diffusion_transformer_layers,
            "num_heads": self.args.diffusion_transformer_heads,
            "head_features": self.args.diffusion_transformer_head_features,
            "multiplier": self.args.diffusion_transformer_multiplier,
        }
        if self.config._supports_cloning:
            transformer = StyleTransformer1d(context_features=self.args.style_dim * 2, **transformer_args)
        else:
            transformer = Transformer1d(**transformer_args)

        diffusion = KDiffusion(
            net=transformer,
            sigma_distribution=LogNormalDistribution(
                mean=self.args.diffusion_dist_mean, std=self.args.diffusion_dist_std
            ),
            sigma_data=self.args.diffusion_dist_sigma_data,
            dynamic_threshold=0.0,
        )

        self.style_diffusion = AudioDiffusionConditional(
            diffusion=diffusion,
            embedding_max_length=self.args.plbert_config.max_position_embeddings,
            embedding_features=self.args.plbert_config.hidden_size,
            embedding_mask_proba=self.args.diffusion_embedding_mask_prob,  # Conditional dropout of batch elements,
        )

        self.diffusion_sampler = DiffusionSampler(
            diffusion=diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(),
            clamp=False,
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

        self.slm_discriminator = SLMDiscriminator(
            slm_model_name=self.args.slm_model_name,
            discriminator_channels=self.args.slm_discriminator_channels,
            sample_rate=self.args.slm_sample_rate,
        )

        self.pitch_extractor = self._create_pitch_extractor()

    def _create_pitch_extractor(self) -> None:
        """Create pitch extractor for training."""
        # TODO: Implement pitch extractor

    def forward(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        aux_input: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Forward pass for training.

        Args:
            tokens: Input token sequences [B, T_text]
            token_lengths: Lengths of token sequences [B]
            audio: Input audio waveforms [B, T_audio]
            audio_lengths: Lengths of audio sequences [B]
            aux_input: Auxiliary inputs (speaker_ids, language_ids, etc.)

        Returns:
            Dictionary of model outputs
        """
        if aux_input is None:
            aux_input = {}

        outputs = {}

        # Text encoding
        acoustic_text_features = self.acoustic_text_encoder(tokens, token_lengths)

        prosodic_text_features = self.prosodic_text_encoder(tokens, token_lengths)

        # Style encoding from ground truth audio
        acoustic_style = self.acoustic_style_encoder(audio, audio_lengths)
        prosodic_style = self.prosodic_style_encoder(audio, audio_lengths)

        # Combine styles
        combined_style = torch.cat([acoustic_style, prosodic_style], dim=1)

        # Style diffusion training (original approach)
        batch_size = combined_style.shape[0]
        device = combined_style.device

        # Add channel dimension for diffusion
        style_target = combined_style.unsqueeze(1)  # [B, 1, style_dim*2]

        # Sample random timesteps/sigmas
        sigmas = self.style_diffusion.diffusion.sigma_distribution(batch_size, device)
        noise = torch.randn_like(style_target)

        # Add noise to target
        noisy_style = style_target + noise * sigmas.view(-1, 1, 1)

        # Predict denoised version
        pred_style = self.style_diffusion.diffusion(
            noisy_style,
            sigmas,
            context_embedding=prosodic_text_features.permute(0, 2, 1),  # [B, T, hidden_dim]
        )

        # Compute diffusion loss
        diffusion_loss = F.mse_loss(pred_style, style_target)
        outputs["diffusion_loss"] = diffusion_loss

        # Duration prediction
        duration_outputs = self.duration_predictor(prosodic_text_features, combined_style, training=True)
        outputs.update(duration_outputs)

        # Decoder
        decoder_outputs = self.decoder(
            acoustic_text_features,
            combined_style,
            duration_outputs["alignments"],
            pitch=aux_input.get("pitch"),
            energy=aux_input.get("energy"),
        )
        outputs.update(decoder_outputs)

        # SLM discriminator
        if self.training and self.training_stage == 2:
            slm_outputs = self.slm_discriminator(
                generated_audio=decoder_outputs["waveform"],
                real_audio=audio,
            )
            outputs.update(slm_outputs)

        return outputs

    @torch.inference_mode()
    def inference(
        self,
        tokens: torch.Tensor,
        aux_input: dict[str, Any] | None = None,
        *,
        alpha: float = 0.3,
        beta: float = 0.7,
    ) -> dict[str, Any]:
        """Inference forward pass.

        Args:
            tokens: Input token sequences [B, T_text]
            aux_input: Auxiliary inputs for inference

        Returns:
            Dictionary of inference outputs
        """
        if aux_input is None:
            aux_input = {}

        outputs = {}

        acoustic_text_features = self.acoustic_text_encoder.inference(tokens)
        prosodic_text_features, plbert_states = self.prosodic_text_encoder.inference(tokens)

        batch_size = prosodic_text_features.shape[0]
        device = prosodic_text_features.device
        noise = torch.randn(batch_size, 1, self.args.style_dim * 2, device=device)
        reference_style = aux_input.get("style", None)
        combined_style = self.diffusion_sampler(
            noise,
            embedding=plbert_states,
            num_steps=self.args.diffusion_inference_steps,
            embedding_scale=1.0,
            features=reference_style,
        ).squeeze(0)

        acoustic_style = combined_style[:, : self.args.style_dim]
        prosodic_style = combined_style[:, self.args.style_dim :]

        if reference_style is not None:
            acoustic_style = alpha * acoustic_style + (1 - alpha) * reference_style[:, :128]
            prosodic_style = beta * prosodic_style + (1 - beta) * reference_style[:, 128:]

        # The original inference recipes differ between the iSTFTNet (LJSpeech)
        # and HiFiGAN (LibriTTS) checkpoints.
        is_hifigan = self.args.decoder_config.generator_type == "hifigan"

        duration_outputs = self.duration_predictor(
            prosodic_text_features, prosodic_style, training=False, extra_final_frames=0 if is_hifigan else 5
        )
        alignments = duration_outputs["alignments"]

        en = duration_outputs["encoder_output"].transpose(-1, -2) @ alignments.unsqueeze(0).to(self.device)
        aligned_text_features = acoustic_text_features @ alignments.unsqueeze(0)
        if is_hifigan:
            # Shift by one frame, repeating the first one
            en = torch.cat([en[..., :1], en[..., :-1]], dim=-1)
            aligned_text_features = torch.cat([aligned_text_features[..., :1], aligned_text_features[..., :-1]], dim=-1)

        f0, energy = self.prosody_predictor(en, prosodic_style)
        model_outputs = self.decoder(aligned_text_features, acoustic_style, f0, energy)
        if is_hifigan:
            # Remove pulse artifact at the end
            model_outputs = model_outputs[..., :-50]

        return {"model_outputs": model_outputs, "alignments": alignments, **outputs}

    def _clone_voice(
        self, speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]], **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compute style vector for the given audio."""
        if isinstance(speaker_wav, list):
            # TODO
            logger.warning("Multiple files not supported yet in speaker_wav, only using first one")
            speaker_wav = speaker_wav[0]
        wave, _ = librosa.load(speaker_wav, sr=self.config.audio.sample_rate)
        audio, _ = librosa.effects.trim(wave, top_db=30)

        wave_tensor = torch.from_numpy(audio).float()
        mel_tensor = self.to_mel(wave_tensor).to(self.device)
        mean, std = -4, 4
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std

        with torch.no_grad():
            acoustic_style = self.acoustic_style_encoder(mel_tensor.unsqueeze(1))
            prosodic_style = self.prosodic_style_encoder(mel_tensor.unsqueeze(1))

        voice = {"style": torch.cat([acoustic_style, prosodic_style], dim=1)}
        metadata = {"name": self.config["model"]}
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
            **kwargs: Arguments passed on to model-specific inference() methods.
        """
        if config is not None:
            warn_synthesize_config_deprecated()
        if (speaker_id := kwargs.pop("speaker_id", None)) is not None:
            speaker = speaker_id
            warn_synthesize_speaker_id_deprecated()
        for key in ("use_griffin_lim", "do_trim_silence", "extra_aux_input"):
            kwargs.pop(key, None)
        # The original model only adds a pad token at the start, no EOS
        text_inputs = [self.tokenizer.characters.pad_id, *self.tokenizer.text_to_ids(text, language=language)]
        text_inputs = torch.as_tensor(text_inputs, dtype=torch.long, device=self.device).unsqueeze(0)

        style = None
        if speaker_wav is not None or speaker:
            style = self.clone_voice(speaker_wav, speaker or None, voice_dir)["style"]
        elif self.config.supports_cloning:
            msg = "This model requires a reference voice: pass `speaker_wav`, or `speaker` for a saved voice"
            raise ValueError(msg)

        outputs = self.inference(
            text_inputs,
            aux_input={
                "x_lengths": torch.tensor(text_inputs.shape[1:2], device=self.device),
                "style": style,
            },
            **kwargs,
        )
        model_outputs = outputs["model_outputs"]
        model_outputs = model_outputs[0].detach().cpu().numpy().squeeze()

        return {
            "wav": model_outputs,
            "alignments": outputs["alignments"],
            "text_inputs": text_inputs,
            "outputs": outputs,
        }

    def train_step(
        self, batch: dict[str, Any], criterion: nn.Module, optimizer_idx: int
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Training step for StyleTTS 2.

        Args:
            batch: Training batch
            criterion: Loss criterion
            optimizer_idx: Optimizer index (0 for discriminator, 1 for generator)

        Returns:
            Tuple of (outputs, losses)
        """
        if optimizer_idx == 0:
            # Discriminator training
            outputs = self.forward(
                batch["tokens"],
                batch["token_lengths"],
                batch["audio"],
                batch["audio_lengths"],
                aux_input=batch.get("aux_input", {}),
            )

            # Cache outputs for generator step
            self._cached_outputs = outputs

            # Compute discriminator losses
            loss_dict = criterion[0](outputs)
            return outputs, loss_dict

        elif optimizer_idx == 1:
            # Generator training
            outputs = self._cached_outputs

            # Compute generator losses
            loss_dict = criterion[1](outputs, batch)
            return outputs, loss_dict

        else:
            raise ValueError(f"Invalid optimizer_idx: {optimizer_idx}")

    def eval_step(self, batch: dict[str, Any], criterion: nn.Module) -> tuple[dict[str, Any], dict[str, Any]]:
        """Evaluation step."""
        outputs = self.forward(
            batch["tokens"],
            batch["token_lengths"],
            batch["audio"],
            batch["audio_lengths"],
            aux_input=batch.get("aux_input", {}),
        )

        # Compute all losses for evaluation
        loss_dict = {}
        if isinstance(criterion, list):
            for i, crit in enumerate(criterion):
                losses = crit(outputs, batch if i == 1 else None)
                loss_dict.update({f"{k}_stage{i}": v for k, v in losses.items()})
        else:
            loss_dict = criterion(outputs, batch)

        return outputs, loss_dict

    def get_optimizer(self) -> list[torch.optim.Optimizer]:
        """Get optimizers for discriminator and generator."""
        # Discriminator optimizer
        disc_optimizer = torch.optim.AdamW(
            self.slm_discriminator.parameters(),
            lr=self.config.lr_disc,
            betas=(0.8, 0.99),
            weight_decay=0.01,
        )

        # Generator optimizer (all other parameters)
        gen_params = []
        for name, param in self.named_parameters():
            if not name.startswith("slm_discriminator"):
                gen_params.append(param)

        gen_optimizer = torch.optim.AdamW(
            gen_params,
            lr=self.config.lr_gen,
            betas=(0.8, 0.99),
            weight_decay=0.01,
        )

        return [disc_optimizer, gen_optimizer]

    def get_criterion(self) -> list[nn.Module]:
        """Get loss criteria for discriminator and generator."""
        from TTS.tts.layers.losses.styletts2_losses import StyleTTS2DiscriminatorLoss, StyleTTS2GeneratorLoss

        disc_criterion = StyleTTS2DiscriminatorLoss(self.config)
        gen_criterion = StyleTTS2GeneratorLoss(self.config)

        return [disc_criterion, gen_criterion]

    def get_lr(self) -> list[float]:
        """Get learning rates for discriminator and generator."""
        return [self.config.lr_disc, self.config.lr_gen]

    def get_scheduler(self, optimizer: list[torch.optim.Optimizer]) -> list[torch.optim.lr_scheduler._LRScheduler]:
        """Get learning rate schedulers."""
        schedulers = []
        for opt in optimizer:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.999)
            schedulers.append(scheduler)
        return schedulers

    def load_checkpoint(
        self,
        config: StyleTTS2Config,
        checkpoint_path: str,
        eval: bool = False,
        strict: bool = True,
        cache: bool = False,
    ):
        """Load model checkpoint."""
        # TODO: Implement checkpoint loading

    def get_data_loader(
        self,
        config: Coqpit,
        assets: dict[str, Any],
        is_eval: bool,
        samples: list[list] | list[dict],
        verbose: bool,
        num_gpus: int,
        rank: int | None = None,
    ) -> DataLoader:
        """Get data loader for StyleTTS 2."""
        # TODO: Implement StyleTTS 2 specific data loader
