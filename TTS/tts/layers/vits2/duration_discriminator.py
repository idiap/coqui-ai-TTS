import torch
from torch import nn
from TTS.tts.layers.generic.normalization import LayerNorm2


class DurationDiscriminatorV2(nn.Module):  # vits2
    """
    Duration predictor discriminator used in the VITS model.

    This class is a PyTorch nn.Module that takes in a sequence of embeddings and
    outputs a probability that the sequence is real or fake.

    The architecture is a two-layer convolutional neural network with ReLU
    activations and LayerNorm.
    """
    def __init__(
        self, in_channels, hidden_channels, kernel_size, p_dropout, cond_channels=0, language_emb_dim=0
    ):
        """
        Args:
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of output channels.
        kernel_size (int): Kernel size of the convolutional layers.
        p_dropout (float): Dropout probability.
        gin_channels (int): Number of channels in the global information (gin).
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.cond_channels = cond_channels
        
        if language_emb_dim:
            hidden_channels += language_emb_dim

        self.conv_1 = nn.Conv1d(
            hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = LayerNorm2(hidden_channels)
        self.conv_2 = nn.Conv1d(
            hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = LayerNorm2(hidden_channels)
        self.dur_proj = nn.Conv1d(1, hidden_channels, 1)

        self.pre_out_conv_1 = nn.Conv1d(
            2 * hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_norm_1 = LayerNorm2(hidden_channels)
        self.pre_out_conv_2 = nn.Conv1d(
            hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_norm_2 = LayerNorm2(hidden_channels)

        self.output_layer = nn.Sequential(nn.Linear(hidden_channels, 1), nn.Sigmoid())
        
        if cond_channels != 0:
           self.cond = nn.Conv1d(cond_channels, in_channels, 1)
           
        if language_emb_dim != 0 and language_emb_dim is not None:
            self.cond_lang = nn.Conv1d(language_emb_dim, hidden_channels, 1)

    def forward_probability(self, x, x_mask, dur, g=None):
        """
        Forward pass of the duration predictor discriminator.

        Args:
        x (torch.Tensor): Input sequence of embeddings.
        x_mask (torch.Tensor): Mask of the input sequence.
        dur (torch.Tensor): Duration of the input sequence.
        g (torch.Tensor): Global information (gin).

        Returns:
        output_prob (torch.Tensor): Probability that the sequence is real or fake.
        """
        dur = self.dur_proj(dur)
        x = torch.cat([x, dur], dim=1)
        x = self.pre_out_conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.pre_out_norm_1(x)
        x = self.pre_out_conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.pre_out_norm_2(x)
        x = x * x_mask
        x = x.transpose(1, 2)
        output_prob = self.output_layer(x)
        return output_prob

    def forward(self, x, x_mask, dur_r, dur_hat, g=None):
        """
        Forward pass of the duration predictor discriminator.

        Args:
        x (torch.Tensor): Input sequence of embeddings.
        x_mask (torch.Tensor): Mask of the input sequence.
        dur_r (torch.Tensor): Real duration of the input sequence.
        dur_hat (torch.Tensor): Predicted duration of the input sequence.
        g (torch.Tensor): Global information (gin).

        Returns:
        output_probs (list): List of probabilities that the sequences are real or fake.
        """
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)

        output_probs = []
        for dur in [dur_r, dur_hat]:
            output_prob = self.forward_probability(x, x_mask, dur, g)
            output_probs.append([output_prob])

        return output_probs