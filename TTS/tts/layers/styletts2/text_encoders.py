"""Text encoders for StyleTTS 2."""

import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.nn.utils import weight_norm
from transformers import AlbertConfig, AlbertModel

from TTS.tts.configs.styletts2_config import PLBertConfig
from TTS.tts.layers.styletts2.utils import _length_to_mask


class LayerNorm(nn.Module):
    """Layer normalization for StyleTTS 2."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class AcousticTextEncoder(nn.Module):
    """Acoustic text encoder for StyleTTS 2.

    Processes text input and produces acoustic text features.
    """

    def __init__(
        self,
        num_chars: int,
        hidden_channels: int = 512,
        num_layers: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.2,
        actv: nn.Module | None = None,
    ):
        super().__init__()
        if actv is None:
            actv = nn.LeakyReLU(0.2)
        self.embedding = nn.Embedding(num_chars, hidden_channels)

        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(num_layers):
            self.cnn.append(
                nn.Sequential(
                    weight_norm(nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding)),
                    LayerNorm(hidden_channels),
                    actv,
                    nn.Dropout(dropout),
                )
            )

        self.lstm = nn.LSTM(hidden_channels, hidden_channels // 2, 1, batch_first=True, bidirectional=True)

    def forward(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            tokens: Input token sequences [B, T]
            token_lengths: Lengths of token sequences [B]

        Returns:
            Encoded text features [B, hidden_channels, T]
        """
        x = self.embedding(tokens)  # [B, T, hidden_channels]
        x = x.transpose(1, 2)  # [B, hidden_channels, T]

        mask = _length_to_mask(token_lengths).to(x.device).unsqueeze(1)
        x.masked_fill_(mask, 0.0)

        for cnn_layer in self.cnn:
            x = cnn_layer(x)
            x.masked_fill_(mask, 0.0)

        x = x.transpose(1, 2)  # [B, T, hidden_channels]

        x = nn.utils.rnn.pack_padded_sequence(x, token_lengths.cpu(), batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x = x.transpose(-1, -2)  # [B, hidden_channels, T]
        x_pad = torch.zeros([x.shape[0], x.shape[1], mask.shape[-1]])

        x_pad[:, :, : x.shape[-1]] = x
        x = x_pad.to(x.device)

        x.masked_fill_(mask, 0.0)

        return x

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.transpose(1, 2)
        for cnn_layer in self.cnn:
            x = cnn_layer(x)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = x.transpose(-1, -2)
        return x


class CustomAlbert(AlbertModel):
    """Custom Albert model for PLBERT."""

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Call the original forward method
        outputs = super().forward(*args, **kwargs)

        # Only return the last_hidden_state
        return outputs.last_hidden_state


class ProsodicTextEncoder(nn.Module):
    """Prosodic text encoder for StyleTTS 2.

    This encoder uses the specific PLBERT model from StyleTTS 2 for prosodic text encoding.
    """

    def __init__(
        self,
        *,
        plbert_config: PLBertConfig | None = None,
        hidden_channels: int = 512,
    ):
        super().__init__()
        if plbert_config is None:
            plbert_config = PLBertConfig()
        self.plbert = CustomAlbert(AlbertConfig(**plbert_config.to_dict()))
        self.plbert_projection = nn.Linear(plbert_config.hidden_size, hidden_channels)

    def forward(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            tokens: Input token sequences [B, T]
            token_lengths: Lengths of token sequences [B]

        Returns:
            Encoded prosodic features [B, hidden_channels, T]
        """
        if token_lengths is not None:
            # Match original StyleTTS2 attention mask exactly: (~text_mask).int()
            attention_mask = (~_length_to_mask(token_lengths)).int()
        else:
            attention_mask = None

        with torch.no_grad():
            # [B, T, plbert_hidden_size]
            hidden_states = self.plbert(input_ids=tokens, attention_mask=attention_mask)

        x = self.plbert_projection(hidden_states)  # [B, T, hidden_channels]
        x = x.transpose(-1, -2)  # [B, hidden_channels, T]

        return x

    def inference(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            tokens: Input token sequences [B, T]

        Returns:
            Tuple of
              - encoded prosodic features [B, hidden_channels, T]
              - BERT hidden states [B, T, plbert_hidden_size]
        """
        with torch.no_grad():
            # [B, T, plbert_hidden_size]
            hidden_states = self.plbert(input_ids=tokens)

        x = self.plbert_projection(hidden_states)  # [B, T, hidden_channels]
        x = x.transpose(-1, -2)  # [B, hidden_channels, T]

        return x, hidden_states

    @staticmethod
    def load_plbert(model_dir: str | os.PathLike[Any]) -> CustomAlbert:
        """Load PLBERT model from StyleTTS 2.

        Args:
            model_dir: Directory containing PLBERT checkpoint and config

        Returns:
            Loaded PLBERT model
        """
        model_dir = Path(model_dir)
        config_path = model_dir / "config.yml"
        if not config_path.is_file():
            msg = f"PLBERT config not found at {config_path}"
            raise FileNotFoundError(msg)

        with open(config_path) as f:
            plbert_config = yaml.safe_load(f)

        # Create Albert configuration
        albert_base_configuration = AlbertConfig(**plbert_config["model_params"])
        bert = CustomAlbert(albert_base_configuration)

        # Find latest checkpoint
        files = os.listdir(model_dir)
        ckpts = list(model_dir.glob("step_*"))

        if not ckpts:
            raise FileNotFoundError(f"No PLBERT checkpoint found in {model_dir}")

        iters = [int(f.stem.split("_")[1]) for f in ckpts]
        max_iters = max(iters)

        checkpoint_path = model_dir / f"step_{max_iters}.t7"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["net"]

        # Process state dict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            if name.startswith("encoder."):
                name = name[8:]  # remove `encoder.`
                new_state_dict[name] = v

        del new_state_dict["embeddings.position_ids"]
        bert.load_state_dict(new_state_dict, strict=False)

        return bert
