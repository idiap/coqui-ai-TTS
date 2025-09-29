import os

import pytest

from tests import get_tests_input_path
from TTS.config import BaseAudioConfig
from TTS.utils.audio.processor import AudioProcessor

WAV_FILE = os.path.join(get_tests_input_path(), "example_1.wav")

conf = BaseAudioConfig(mel_fmax=8000, pitch_fmax=640, pitch_fmin=1)


@pytest.fixture
def ap():
    """Set up audio processor."""
    return AudioProcessor(**conf)


norms = [
    # maxnorm = 1.0
    (1.0, False, False, False),
    (1.0, True, False, False),
    (1.0, True, True, False),
    (1.0, True, False, True),
    (1.0, True, True, True),
    # maxnorm = 4.0
    (4.0, False, False, False),
    (4.0, True, False, False),
    (4.0, True, True, False),
    (4.0, True, False, True),
    (4.0, True, True, True),
]


@pytest.mark.parametrize("norms", norms)
def test_audio_synthesis(tmp_path, ap, norms):
    """1. load wav
    2. set normalization parameters
    3. extract mel-spec
    4. invert to wav and save the output
    """
    print(" > Sanity check for the process wav -> mel -> wav")
    max_norm, signal_norm, symmetric_norm, clip_norm = norms
    ap.max_norm = max_norm
    ap.signal_norm = signal_norm
    ap.symmetric_norm = symmetric_norm
    ap.clip_norm = clip_norm
    wav = ap.load_wav(WAV_FILE)
    mel = ap.melspectrogram(wav)
    wav_ = ap.inv_melspectrogram(mel)
    file_name = (
        f"audio_test-melspec_max_norm_{max_norm}-signal_norm_{signal_norm}-"
        f"symmetric_{symmetric_norm}-clip_norm_{clip_norm}.wav"
    )
    print(" | > Creating wav file at : ", file_name)
    ap.save_wav(wav_, tmp_path / file_name)


def test_normalize(ap):
    """Check normalization and denormalization for range values and consistency"""
    print(" > Testing normalization and denormalization.")
    wav = ap.load_wav(WAV_FILE)
    wav = ap.sound_norm(wav)  # normalize audio to get abetter normalization range below.
    ap.signal_norm = False
    x = ap.melspectrogram(wav)
    x_old = x

    ap.signal_norm = True
    ap.symmetric_norm = False
    ap.clip_norm = False
    ap.max_norm = 4.0
    x_norm = ap.normalize(x)
    print(
        f" > MaxNorm: {ap.max_norm}, ClipNorm:{ap.clip_norm}, SymmetricNorm:{ap.symmetric_norm}, SignalNorm:{ap.signal_norm} Range-> {x_norm.max()} --  {x_norm.min()}"
    )
    assert (x_old - x).sum() == 0
    # check value range
    assert x_norm.max() <= ap.max_norm + 1, x_norm.max()
    assert x_norm.min() >= 0 - 1, x_norm.min()
    # check denorm.
    x_ = ap.denormalize(x_norm)
    assert (x - x_).sum() < 1e-3, (x - x_).mean()

    ap.signal_norm = True
    ap.symmetric_norm = False
    ap.clip_norm = True
    ap.max_norm = 4.0
    x_norm = ap.normalize(x)
    print(
        f" > MaxNorm: {ap.max_norm}, ClipNorm:{ap.clip_norm}, SymmetricNorm:{ap.symmetric_norm}, SignalNorm:{ap.signal_norm} Range-> {x_norm.max()} --  {x_norm.min()}"
    )

    assert (x_old - x).sum() == 0
    # check value range
    assert x_norm.max() <= ap.max_norm, x_norm.max()
    assert x_norm.min() >= 0, x_norm.min()
    # check denorm.
    x_ = ap.denormalize(x_norm)
    assert (x - x_).sum() < 1e-3, (x - x_).mean()

    ap.signal_norm = True
    ap.symmetric_norm = True
    ap.clip_norm = False
    ap.max_norm = 4.0
    x_norm = ap.normalize(x)
    print(
        f" > MaxNorm: {ap.max_norm}, ClipNorm:{ap.clip_norm}, SymmetricNorm:{ap.symmetric_norm}, SignalNorm:{ap.signal_norm} Range-> {x_norm.max()} --  {x_norm.min()}"
    )

    assert (x_old - x).sum() == 0
    # check value range
    assert x_norm.max() <= ap.max_norm + 1, x_norm.max()
    assert x_norm.min() >= -ap.max_norm - 2, x_norm.min()  # pylint: disable=invalid-unary-operand-type
    assert x_norm.min() <= 0, x_norm.min()
    # check denorm.
    x_ = ap.denormalize(x_norm)
    assert (x - x_).sum() < 1e-3, (x - x_).mean()

    ap.signal_norm = True
    ap.symmetric_norm = True
    ap.clip_norm = True
    ap.max_norm = 4.0
    x_norm = ap.normalize(x)
    print(
        f" > MaxNorm: {ap.max_norm}, ClipNorm:{ap.clip_norm}, SymmetricNorm:{ap.symmetric_norm}, SignalNorm:{ap.signal_norm} Range-> {x_norm.max()} --  {x_norm.min()}"
    )

    assert (x_old - x).sum() == 0
    # check value range
    assert x_norm.max() <= ap.max_norm, x_norm.max()
    assert x_norm.min() >= -ap.max_norm, x_norm.min()  # pylint: disable=invalid-unary-operand-type
    assert x_norm.min() <= 0, x_norm.min()
    # check denorm.
    x_ = ap.denormalize(x_norm)
    assert (x - x_).sum() < 1e-3, (x - x_).mean()

    ap.signal_norm = True
    ap.symmetric_norm = False
    ap.max_norm = 1.0
    x_norm = ap.normalize(x)
    print(
        f" > MaxNorm: {ap.max_norm}, ClipNorm:{ap.clip_norm}, SymmetricNorm:{ap.symmetric_norm}, SignalNorm:{ap.signal_norm} Range-> {x_norm.max()} --  {x_norm.min()}"
    )

    assert (x_old - x).sum() == 0
    assert x_norm.max() <= ap.max_norm, x_norm.max()
    assert x_norm.min() >= 0, x_norm.min()
    x_ = ap.denormalize(x_norm)
    assert (x - x_).sum() < 1e-3

    ap.signal_norm = True
    ap.symmetric_norm = True
    ap.max_norm = 1.0
    x_norm = ap.normalize(x)
    print(
        f" > MaxNorm: {ap.max_norm}, ClipNorm:{ap.clip_norm}, SymmetricNorm:{ap.symmetric_norm}, SignalNorm:{ap.signal_norm} Range-> {x_norm.max()} --  {x_norm.min()}"
    )

    assert (x_old - x).sum() == 0
    assert x_norm.max() <= ap.max_norm, x_norm.max()
    assert x_norm.min() >= -ap.max_norm, x_norm.min()  # pylint: disable=invalid-unary-operand-type
    assert x_norm.min() < 0, x_norm.min()
    x_ = ap.denormalize(x_norm)
    assert (x - x_).sum() < 1e-3


def test_scaler(ap):
    scaler_stats_path = os.path.join(get_tests_input_path(), "scale_stats.npy")
    conf.stats_path = scaler_stats_path
    conf.preemphasis = 0.0
    conf.do_trim_silence = True
    conf.signal_norm = True

    ap = AudioProcessor(**conf)
    mel_mean, mel_std, linear_mean, linear_std, _ = ap.load_stats(scaler_stats_path)
    ap.setup_scaler(mel_mean, mel_std, linear_mean, linear_std)

    ap.signal_norm = False
    ap.preemphasis = 0.0

    # test scaler forward and backward transforms
    wav = ap.load_wav(WAV_FILE)
    mel_reference = ap.melspectrogram(wav)
    mel_norm = ap.melspectrogram(wav)
    mel_denorm = ap.denormalize(mel_norm)
    assert abs(mel_reference - mel_denorm).max() < 1e-4


def test_compute_f0(ap):
    wav = ap.load_wav(WAV_FILE)
    pitch = ap.compute_f0(wav)
    mel = ap.melspectrogram(wav)
    assert pitch.shape[0] == mel.shape[1]
