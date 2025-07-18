import torch
import torch.nn as nn

from TTS.tts.layers.styletts2.common import AdaINResBlk1d


class ProsodyPredictor(nn.Module):
    """F0 and Energy predictor.

    NB: In the original implementation this is part of the duration predictor.
    """

    def __init__(self, hidden_dim: int = 512, style_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.shared = nn.LSTM(hidden_dim + style_dim, hidden_dim // 2, 1, batch_first=True, bidirectional=True)

        self.F0 = nn.ModuleList(
            [
                AdaINResBlk1d(hidden_dim, hidden_dim, style_dim, dropout_p=dropout),
                AdaINResBlk1d(hidden_dim, hidden_dim // 2, style_dim, upsample=True, dropout_p=dropout),
                AdaINResBlk1d(hidden_dim // 2, hidden_dim // 2, style_dim, dropout_p=dropout),
            ]
        )

        self.energy = nn.ModuleList(
            [
                AdaINResBlk1d(hidden_dim, hidden_dim, style_dim, dropout_p=dropout),
                AdaINResBlk1d(hidden_dim, hidden_dim // 2, style_dim, upsample=True, dropout_p=dropout),
                AdaINResBlk1d(hidden_dim // 2, hidden_dim // 2, style_dim, dropout_p=dropout),
            ]
        )

        # Output projections
        self.F0_proj = nn.Conv1d(hidden_dim // 2, 1, 1, 1, 0)
        self.energy_proj = nn.Conv1d(hidden_dim // 2, 1, 1, 1, 0)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Alignment-upsampled text features [B, alignment_frames, hidden_dim]
            style: Prosodic style vector [B, style_dim]

        Returns:
            f0_pred: F0 predictions [B, alignment_frames * 2]
            energy_pred: Energy predictions [B, alignment_frames * 2]
        """
        x, _ = self.shared(x.transpose(-1, -2))

        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)

        energy = x.transpose(-1, -2)
        for block in self.energy:
            energy = block(energy, s)
        energy = self.energy_proj(energy)

        return F0.squeeze(1), energy.squeeze(1)
