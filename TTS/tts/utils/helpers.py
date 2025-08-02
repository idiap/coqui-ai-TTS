import numpy as np
import torch
from scipy.stats import betabinom
from torch.nn import functional as F


class StandardScaler:
    """StandardScaler for mean-scale normalization with the given mean and scale values."""

    def __init__(self, mean: np.ndarray = None, scale: np.ndarray = None) -> None:
        self.mean_ = mean
        self.scale_ = scale

    def set_stats(self, mean, scale):
        self.mean_ = mean
        self.scale_ = scale

    def reset_stats(self):
        delattr(self, "mean_")
        delattr(self, "scale_")

    def transform(self, X):
        X = np.asarray(X)
        X -= self.mean_
        X /= self.scale_
        return X

    def inverse_transform(self, X):
        X = np.asarray(X)
        X *= self.scale_
        X += self.mean_
        return X


# from https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
def sequence_mask(sequence_length: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """Create a sequence mask for filtering padding in a sequence tensor.

    Args:
        sequence_length (torch.tensor): Sequence lengths.
        max_len (int, Optional): Maximum sequence length. Defaults to None.

    Shapes:
        - mask: :math:`[B, T_max]`
    """
    if max_len is None:
        max_len = sequence_length.max()
    seq_range = torch.arange(max_len, dtype=sequence_length.dtype, device=sequence_length.device)
    # B x T_max
    return seq_range.unsqueeze(0) < sequence_length.unsqueeze(1)


def segment(x: torch.tensor, segment_indices: torch.tensor, segment_size=4, pad_short=False):
    """Segment each sample in a batch based on the provided segment indices

    Args:
        x (torch.tensor): Input tensor.
        segment_indices (torch.tensor): Segment indices.
        segment_size (int): Expected output segment size.
        pad_short (bool): Pad the end of input tensor with zeros if shorter than the segment size.
    """
    # pad the input tensor if it is shorter than the segment size
    if pad_short and x.shape[-1] < segment_size:
        x = torch.nn.functional.pad(x, (0, segment_size - x.size(2)))

    segments = torch.zeros_like(x[:, :, :segment_size])

    for i in range(x.size(0)):
        index_start = segment_indices[i]
        index_end = index_start + segment_size
        x_i = x[i]
        if pad_short and index_end >= x.size(2):
            # pad the sample if it is shorter than the segment size
            x_i = torch.nn.functional.pad(x_i, (0, (index_end + 1) - x.size(2)))
        segments[i] = x_i[:, index_start:index_end]
    return segments


def rand_segments(
    x: torch.tensor, x_lengths: torch.tensor = None, segment_size=4, let_short_samples=False, pad_short=False
):
    """Create random segments based on the input lengths.

    Args:
        x (torch.tensor): Input tensor.
        x_lengths (torch.tensor): Input lengths.
        segment_size (int): Expected output segment size.
        let_short_samples (bool): Allow shorter samples than the segment size.
        pad_short (bool): Pad the end of input tensor with zeros if shorter than the segment size.

    Shapes:
        - x: :math:`[B, C, T]`
        - x_lengths: :math:`[B]`
    """
    _x_lenghts = x_lengths.clone()
    B, _, T = x.size()
    if pad_short:
        if T < segment_size:
            x = torch.nn.functional.pad(x, (0, segment_size - T))
            T = segment_size
    if _x_lenghts is None:
        _x_lenghts = T
    len_diff = _x_lenghts - segment_size
    if let_short_samples:
        _x_lenghts[len_diff < 0] = segment_size
        len_diff = _x_lenghts - segment_size
    else:
        assert all(len_diff > 0), (
            f" [!] At least one sample is shorter than the segment size ({segment_size}). \n {_x_lenghts}"
        )
    segment_indices = (torch.rand([B]).type_as(x) * (len_diff + 1)).long()
    ret = segment(x, segment_indices, segment_size, pad_short=pad_short)
    return ret, segment_indices


def average_over_durations(values, durs):
    """Average values over durations.

    Shapes:
        - values: :math:`[B, 1, T_de]`
        - durs: :math:`[B, T_en]`
        - avg: :math:`[B, 1, T_en]`
    """
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = torch.nn.functional.pad(durs_cums_ends[:, :-1], (1, 0))
    values_nonzero_cums = torch.nn.functional.pad(torch.cumsum(values != 0.0, dim=2), (1, 0))
    values_cums = torch.nn.functional.pad(torch.cumsum(values, dim=2), (1, 0))

    bs, l = durs_cums_ends.size()
    n_formants = values.size(1)
    dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, l)
    dce = durs_cums_ends[:, None, :].expand(bs, n_formants, l)

    values_sums = (torch.gather(values_cums, 2, dce) - torch.gather(values_cums, 2, dcs)).float()
    values_nelems = (torch.gather(values_nonzero_cums, 2, dce) - torch.gather(values_nonzero_cums, 2, dcs)).float()

    avg = torch.where(values_nelems == 0.0, values_nelems, values_sums / values_nelems)
    return avg


def convert_pad_shape(pad_shape: list[list]) -> list:
    l = pad_shape[::-1]
    return [item for sublist in l for item in sublist]


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Generate alignment path based on the given segment durations.

    Shapes:
        - duration: :math:`[B, T_en]`
        - mask: :math:'[B, T_en, T_de]`
        - path: :math:`[B, T_en, T_de]`
    """
    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, dim=1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    return path * mask


def generate_attention(
    duration: torch.Tensor, x_mask: torch.Tensor, y_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Generate an attention map from the linear scale durations.

    Args:
        duration (Tensor): Linear scale durations.
        x_mask (Tensor): Mask for the input (character) sequence.
        y_mask (Tensor): Mask for the output (spectrogram) sequence. Compute it from the predicted durations
            if None. Defaults to None.

    Shapes
       - duration: :math:`(B, T_{en})`
       - x_mask: :math:`(B, T_{en})`
       - y_mask: :math:`(B, T_{de})`
    """
    # compute decode mask from the durations
    if y_mask is None:
        y_lengths = duration.sum(dim=1).long()
        y_lengths[y_lengths < 1] = 1
        y_mask = sequence_mask(y_lengths).unsqueeze(1).to(duration.dtype)
    attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
    return generate_path(duration, attn_mask.squeeze(1)).to(duration.dtype)


def expand_encoder_outputs(
    x: torch.Tensor, duration: torch.Tensor, x_mask: torch.Tensor, y_lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate attention alignment map from durations and expand encoder outputs.

    Shapes:
        - x: Encoder output :math:`(B, D_{en}, T_{en})`
        - duration: :math:`(B, T_{en})`
        - x_mask: :math:`(B, T_{en})`
        - y_lengths: :math:`(B)`

    Examples::

        encoder output: [a,b,c,d]
        durations: [1, 3, 2, 1]

        expanded: [a, b, b, b, c, c, d]
        attention map: [[0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 1, 1, 0],
                        [0, 1, 1, 1, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0]]
    """
    y_mask = sequence_mask(y_lengths).unsqueeze(1).to(x.dtype)
    attn = generate_attention(duration, x_mask, y_mask)
    x_expanded = torch.einsum("kmn, kjm -> kjn", [attn.float(), x])
    return x_expanded, attn, y_mask


def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling_factor=1.0):
    P, M = phoneme_count, mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M + 1):
        a, b = scaling_factor * i, scaling_factor * (M + 1 - i)
        rv = betabinom(P, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return np.array(mel_text_probs)


def compute_attn_prior(x_len, y_len, scaling_factor=1.0):
    """Compute attention priors for the alignment network."""
    attn_prior = beta_binomial_prior_distribution(
        x_len,
        y_len,
        scaling_factor,
    )
    return attn_prior  # [y_len, x_len]
