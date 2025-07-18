"""Differentiable duration predictor for StyleTTS 2."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNorm(nn.Module):
    """Linear layer with Xavier initialization."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, w_init_gain: str = "linear"):
        super().__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization (matches original StyleTTS2)."""

    def __init__(self, style_dim: int, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)


class DurationEncoder(nn.Module):
    """Encodes text for duration prediction."""

    def __init__(self, style_dim: int, hidden_channels: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(num_layers):
            self.lstms.append(
                nn.LSTM(
                    hidden_channels + style_dim,
                    hidden_channels // 2,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout,
                )
            )
            self.lstms.append(AdaLayerNorm(style_dim, hidden_channels))
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        self.style_dim = style_dim

    def forward(
        self, x: torch.Tensor, style: torch.Tensor, text_lengths: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass (matches original exactly).

        Args:
            x: Text features [B, hidden_dim, T]
            style: Style vector [B, style_dim]
            text_lengths: Text lengths [B]
            masks: Text masks [B, T]

        Returns:
            Encoded features [B, T, hidden_dim]
        """
        masks = masks.to(text_lengths.device)
        x = x.permute(2, 0, 1)  # [T, B, hidden_dim]
        s = style.expand(x.shape[0], x.shape[1], -1)  # [T, B, style_dim]
        x = torch.cat([x, s], axis=-1)  # [T, B, hidden_dim + style_dim]
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)  # [B, T, hidden_dim + style_dim]
        input_lengths = text_lengths.cpu().numpy()
        x = x.transpose(-1, -2)  # [B, hidden_dim + style_dim, T]

        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, -1, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                x = x.transpose(-1, -2)  # [B, T, hidden_dim + style_dim]
                x_packed = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True, enforce_sorted=False)
                block.flatten_parameters()
                x_packed, _ = block(x_packed)
                x, _ = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)
                x = F.dropout(x, self.dropout, training=self.training)
                x = x.transpose(-1, -2)

                x_pad = torch.zeros([x.shape[0], x.shape[1], masks.shape[-1]], device=x.device)
                x_pad[:, :, : x.shape[-1]] = x
                x = x_pad

        return x.transpose(-1, -2)


class DifferentiableDurationPredictor(nn.Module):
    """Differentiable duration predictor for StyleTTS 2.

    This predictor matches the original StyleTTS2 ProsodyPredictor architecture exactly.
    """

    def __init__(
        self,
        style_dim: int = 128,
        hidden_channels: int = 512,
        num_layers: int = 3,
        max_duration: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.max_duration = max_duration
        self.style_dim = style_dim

        self.text_encoder = DurationEncoder(
            style_dim=style_dim,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.lstm = nn.LSTM(
            hidden_channels + style_dim,  # 512 + 128 = 640
            hidden_channels // 2,  # 256
            1,
            batch_first=True,
            bidirectional=True,
        )

        self.duration_proj = LinearNorm(hidden_channels, max_duration)

    def forward(
        self,
        text_features: torch.Tensor,
        style: torch.Tensor,
        training: bool = True,
        target_durations: torch.Tensor | None = None,
        extra_final_frames: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            text_features: Text features [B, hidden_channels, T]
            style: Style vector [B, style_dim]
            training: Whether in training mode
            target_durations: Target durations for training [B, T]
            extra_final_frames: Frames added to the last token's predicted duration at inference

        Returns:
            Dictionary containing duration predictions and alignments
        """
        batch_size, _, text_len = text_features.shape

        # Create text lengths (assume no padding for now)
        text_lengths = torch.full((batch_size,), text_len, device=text_features.device)

        # TODO
        # In original: mask = torch.gt(mask+1, lengths.unsqueeze(1)) where mask = torch.arange(lengths.max())
        # This creates all False for full-length sequences
        masks = torch.zeros(batch_size, text_len, device=text_features.device, dtype=torch.bool)

        d = self.text_encoder(text_features, style, text_lengths, masks)  # [B, T, 640]

        # Step 2: Main LSTM processing (matches original predictor.lstm)
        # No need to concatenate style again - DurationEncoder already includes it
        packed_input = nn.utils.rnn.pack_padded_sequence(d, text_lengths.cpu(), batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()
        packed_output, _ = self.lstm(packed_input)

        # Unpack sequences
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        x_dropout = F.dropout(x, 0.5, training=self.training)
        duration_logits = self.duration_proj(x_dropout)

        if training and target_durations is not None:
            # Training mode: use target durations
            durations = target_durations.float()
            alignments = self._generate_alignment(durations)
        else:
            # Inference mode: predict durations
            duration_raw = torch.sigmoid(duration_logits).sum(dim=-1)  # [B, T]
            durations = torch.round(duration_raw.squeeze()).clamp(min=1)

            durations[-1] += extra_final_frames
            alignments = self._generate_alignment(durations)

        return {
            "duration_logits": duration_logits,
            "predicted_durations": durations,
            "alignments": alignments,
            "encoder_output": d,
        }

    def _generate_alignment(self, durations: torch.Tensor) -> torch.Tensor:
        """Generate alignment matrix from durations (matching original StyleTTS2).

        Args:
            durations: Duration predictions [T]

        Returns:
            Alignment matrix [T_out, T_in]
        """
        text_len = durations.shape[0]

        # Compute total duration for each batch
        total_durations = durations.sum()
        max_duration = int(total_durations.max().item())

        # Create alignment matrix (binary assignment like original)
        alignment = torch.zeros(text_len, int(total_durations))
        # alignment = torch.zeros(batch_size, max_duration, text_len, device=device)

        c_frame = 0
        for t in range(text_len):
            alignment[t, c_frame : c_frame + int(durations[t])] = 1
            c_frame += int(durations[t])

        return alignment
