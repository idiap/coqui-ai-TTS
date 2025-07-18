"""SLM discriminator for StyleTTS 2."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

logger = logging.getLogger(__name__)


class SLMDiscriminator(nn.Module):
    """Speech Language Model discriminator for StyleTTS 2.

    This discriminator uses a pre-trained WavLM model as a feature extractor
    and adds a CNN head for discrimination.
    """

    def __init__(
        self,
        slm_model_name: str = "microsoft/wavlm-base-plus",
        discriminator_channels: int = 256,
        sample_rate: int = 16000,
        slm_hidden: int = 768,
        slm_layers: int = 13,
        initial_channel: int = 64,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.slm_model_name = slm_model_name
        self.sample_rate = sample_rate
        self.discriminator_channels = discriminator_channels

        # Load pre-trained WavLM model
        self._load_slm_model()

        # Discriminator head
        norm_f = weight_norm if not use_spectral_norm else nn.utils.spectral_norm

        # Feature pooling layer
        self.pre = norm_f(nn.Conv1d(slm_hidden * slm_layers, initial_channel, 1, 1, padding=0))

        # CNN layers
        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv1d(initial_channel, initial_channel * 2, kernel_size=5, padding=2)),
                norm_f(nn.Conv1d(initial_channel * 2, initial_channel * 4, kernel_size=5, padding=2)),
                norm_f(nn.Conv1d(initial_channel * 4, initial_channel * 4, 5, 1, padding=2)),
            ]
        )

        # Output layer
        self.conv_post = norm_f(nn.Conv1d(initial_channel * 4, 1, 3, 1, padding=1))

        # Freeze WavLM parameters
        for param in self.slm_model.parameters():
            param.requires_grad = False

    def _load_slm_model(self):
        """Load pre-trained SLM model."""
        try:
            from transformers import AutoModel

            self.slm_model = AutoModel.from_pretrained(self.slm_model_name)
            self.slm_model.eval()
        except ImportError:
            raise ImportError("transformers library required for SLM discriminator")
        except OSError:
            logger.warning("Could not load %s, using dummy model", self.slm_model_name)
            # Create dummy model for development
            self.slm_model = self._create_dummy_slm_model()

    def _create_dummy_slm_model(self):
        """Create a dummy SLM model for development."""

        class DummySLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(1, 768 * 13, 3, 1, 1)

            def forward(self, input_values, **kwargs):
                # Simple processing for dummy model
                x = input_values.unsqueeze(1) if input_values.dim() == 2 else input_values
                x = self.conv(x)

                # Simulate WavLM output format
                class DummyOutput:
                    def __init__(self, hidden_states):
                        self.hidden_states = hidden_states

                return DummyOutput(x.transpose(1, 2))

        return DummySLM()

    def _resample_audio(self, audio: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Resample audio to SLM sample rate."""
        if orig_sr == self.sample_rate:
            return audio

        # Simple resampling (in practice, you'd use torchaudio.transforms.Resample)
        if orig_sr > self.sample_rate:
            # Downsample
            step = orig_sr // self.sample_rate
            return audio[..., ::step]
        else:
            # Upsample (simple repetition)
            repeat_factor = self.sample_rate // orig_sr
            return audio.repeat_interleave(repeat_factor, dim=-1)

    def _extract_slm_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract features from SLM model."""
        # Ensure audio is in correct sample rate
        if hasattr(self, "target_sample_rate"):
            audio = self._resample_audio(audio, self.target_sample_rate)

        # Extract features from all layers
        with torch.no_grad():
            if hasattr(self.slm_model, "forward"):
                outputs = self.slm_model(audio, output_hidden_states=True)
                if hasattr(outputs, "hidden_states"):
                    # Real WavLM model
                    hidden_states = outputs.hidden_states  # List of [B, T, hidden_size]
                    hidden_states = torch.cat(hidden_states, dim=-1)  # [B, T, hidden_size * num_layers]
                else:
                    # Dummy model
                    hidden_states = outputs.hidden_states  # [B, T, hidden_size * num_layers]
            else:
                # Fallback
                hidden_states = self.slm_model(audio).hidden_states

        return hidden_states.transpose(1, 2)  # [B, hidden_size * num_layers, T]

    def forward(
        self,
        generated_audio: torch.Tensor,
        real_audio: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            generated_audio: Generated audio waveform [B, T]
            real_audio: Real audio waveform [B, T]

        Returns:
            Dictionary containing discriminator outputs and features
        """
        # Extract SLM features
        gen_features = self._extract_slm_features(generated_audio)
        real_features = self._extract_slm_features(real_audio)

        # Process through discriminator head
        gen_scores, gen_fmaps = self._discriminate(gen_features)
        real_scores, real_fmaps = self._discriminate(real_features)

        return {
            "gen_scores": gen_scores,
            "real_scores": real_scores,
            "gen_fmaps": gen_fmaps,
            "real_fmaps": real_fmaps,
        }

    def _discriminate(self, features: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run discriminator on features."""
        x = self.pre(features)

        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class WavLMDiscriminator(nn.Module):
    """Simplified WavLM discriminator (from original codebase)."""

    def __init__(
        self,
        slm_hidden: int = 768,
        slm_layers: int = 13,
        initial_channel: int = 64,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        norm_f = weight_norm if not use_spectral_norm else nn.utils.spectral_norm

        self.pre = norm_f(nn.Conv1d(slm_hidden * slm_layers, initial_channel, 1, 1, padding=0))

        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv1d(initial_channel, initial_channel * 2, kernel_size=5, padding=2)),
                norm_f(nn.Conv1d(initial_channel * 2, initial_channel * 4, kernel_size=5, padding=2)),
                norm_f(nn.Conv1d(initial_channel * 4, initial_channel * 4, 5, 1, padding=2)),
            ]
        )

        self.conv_post = norm_f(nn.Conv1d(initial_channel * 4, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features [B, C, T]

        Returns:
            Discriminator scores [B, 1]
        """
        x = self.pre(x)

        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)

        return x
