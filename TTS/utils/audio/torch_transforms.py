import logging

import librosa
import torch
from torch import nn

logger = logging.getLogger(__name__)


hann_window = {}
mel_basis = {}


def amp_to_db(x: torch.Tensor, *, spec_gain: float = 1.0, clip_val: float = 1e-5) -> torch.Tensor:
    """Spectral normalization / dynamic range compression."""
    return torch.log(torch.clamp(x, min=clip_val) * spec_gain)


def db_to_amp(x: torch.Tensor, *, spec_gain: float = 1.0) -> torch.Tensor:
    """Spectral denormalization / dynamic range decompression."""
    return torch.exp(x) / spec_gain


def wav_to_spec(y: torch.Tensor, n_fft: int, hop_length: int, win_length: int, *, center: bool = False) -> torch.Tensor:
    """
    Args Shapes:
        - y : :math:`[B, 1, T]`

    Return Shapes:
        - spec : :math:`[B,C,T]`
    """
    y = y.squeeze(1)

    if torch.min(y) < -1.0:
        logger.info("min value is %.3f", torch.min(y))
    if torch.max(y) > 1.0:
        logger.info("max value is %.3f", torch.max(y))

    global hann_window
    wnsize_dtype_device = f"{win_length}_{y.dtype}_{y.device}"
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_length).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=hann_window[wnsize_dtype_device].clone(),
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    return torch.sqrt(spec.pow(2).sum(-1) + 1e-6)


def spec_to_mel(
    spec: torch.Tensor, n_fft: int, num_mels: int, sample_rate: int, fmin: float, fmax: float
) -> torch.Tensor:
    """
    Args Shapes:
        - spec : :math:`[B,C,T]`

    Return Shapes:
        - mel : :math:`[B,C,T]`
    """
    global mel_basis
    fmax_dtype_device = f"{n_fft}_{fmax}_{spec.dtype}_{spec.device}"
    if fmax_dtype_device not in mel_basis:
        # TODO: switch librosa to torchaudio
        mel = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    mel = torch.matmul(mel_basis[fmax_dtype_device], spec)
    return amp_to_db(mel)


def wav_to_mel(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sample_rate: int,
    hop_length: int,
    win_length: int,
    fmin: float,
    fmax: float,
    *,
    center: bool = False,
) -> torch.Tensor:
    """
    Args Shapes:
        - y : :math:`[B, 1, T]`

    Return Shapes:
        - spec : :math:`[B,C,T]`
    """
    spec = wav_to_spec(y, n_fft, hop_length, win_length, center=center)
    return spec_to_mel(spec, n_fft, num_mels, sample_rate, fmin, fmax)


class TorchSTFT(nn.Module):  # pylint: disable=abstract-method
    """Some of the audio processing funtions using Torch for faster batch processing.

    Args:

        n_fft (int):
            FFT window size for STFT.

        hop_length (int):
            number of frames between STFT columns.

        win_length (int, optional):
            STFT window length.

        pad_wav (bool, optional):
            If True pad the audio with (n_fft - hop_length) / 2). Defaults to False.

        window (str, optional):
            The name of a function to create a window tensor that is applied/multiplied to each frame/window. Defaults to "hann_window"

        sample_rate (int, optional):
            target audio sampling rate. Defaults to None.

        mel_fmin (int, optional):
            minimum filter frequency for computing melspectrograms. Defaults to None.

        mel_fmax (int, optional):
            maximum filter frequency for computing melspectrograms. Defaults to None.

        n_mels (int, optional):
            number of melspectrogram dimensions. Defaults to None.

        use_mel (bool, optional):
            If True compute the melspectrograms otherwise. Defaults to False.

        do_amp_to_db_linear (bool, optional):
            enable/disable amplitude to dB conversion of linear spectrograms. Defaults to False.

        spec_gain (float, optional):
            gain applied when converting amplitude to DB. Defaults to 1.0.

        power (float, optional):
            Exponent for the magnitude spectrogram, e.g., 1 for energy, 2 for power, etc.  Defaults to None.

        use_htk (bool, optional):
            Use HTK formula in mel filter instead of Slaney.

        mel_norm (None, 'slaney', or number, optional):
            If 'slaney', divide the triangular mel weights by the width of the mel band
            (area normalization).

            If numeric, use `librosa.util.normalize` to normalize each filter by to unit l_p norm.
            See `librosa.util.normalize` for a full description of supported norm values
            (including `+-np.inf`).

            Otherwise, leave all the triangles aiming for a peak value of 1.0. Defaults to "slaney".
    """

    def __init__(
        self,
        n_fft,
        hop_length,
        win_length,
        pad_wav=False,
        window="hann_window",
        sample_rate=None,
        mel_fmin=0,
        mel_fmax=None,
        n_mels=80,
        use_mel=False,
        do_amp_to_db=False,
        spec_gain=1.0,
        power=None,
        use_htk=False,
        mel_norm="slaney",
        normalized=False,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.pad_wav = pad_wav
        self.sample_rate = sample_rate
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.n_mels = n_mels
        self.use_mel = use_mel
        self.do_amp_to_db = do_amp_to_db
        self.spec_gain = spec_gain
        self.power = power
        self.use_htk = use_htk
        self.mel_norm = mel_norm
        self.window = nn.Parameter(getattr(torch, window)(win_length), requires_grad=False)
        self.mel_basis = None
        self.normalized = normalized
        if use_mel:
            self._build_mel_basis()

    def __call__(self, x):
        """Compute spectrogram frames by torch based stft.

        Args:
            x (Tensor): input waveform

        Returns:
            Tensor: spectrogram frames.

        Shapes:
            x: [B x T] or [:math:`[B, 1, T]`]
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if self.pad_wav:
            padding = int((self.n_fft - self.hop_length) / 2)
            x = torch.nn.functional.pad(x, (padding, padding), mode="reflect")
        # B x D x T x 2
        o = torch.view_as_real(
            torch.stft(
                x.squeeze(1),
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
                center=True,
                pad_mode="reflect",  # compatible with audio.py
                normalized=self.normalized,
                onesided=True,
                return_complex=True,
            )
        )
        M = o[:, :, :, 0]
        P = o[:, :, :, 1]
        S = torch.sqrt(torch.clamp(M**2 + P**2, min=1e-8))

        if self.power is not None:
            S = S**self.power

        if self.use_mel:
            S = torch.matmul(self.mel_basis.to(x), S)
        if self.do_amp_to_db:
            S = self._amp_to_db(S, spec_gain=self.spec_gain)
        return S

    def _build_mel_basis(self):
        mel_basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax,
            htk=self.use_htk,
            norm=self.mel_norm,
        )
        self.mel_basis = torch.from_numpy(mel_basis).float()
