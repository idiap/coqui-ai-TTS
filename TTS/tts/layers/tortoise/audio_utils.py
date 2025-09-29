import logging

import librosa
import numpy as np
import torch
import torchaudio
from scipy.io.wavfile import read

from TTS.utils.audio.torch_transforms import TorchSTFT, amp_to_db

logger = logging.getLogger(__name__)


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    if data.dtype == np.int32:
        norm_fix = 2**31
    elif data.dtype == np.int16:
        norm_fix = 2**15
    elif data.dtype == np.float16 or data.dtype == np.float32:
        norm_fix = 1.0
    else:
        raise NotImplementedError(f"Provided data dtype not supported: {data.dtype}")
    return (torch.FloatTensor(data.astype(np.float32)) / norm_fix, sampling_rate)


def check_audio(audio, audiopath: str):
    # Check some assumptions about audio range. This should be automatically fixed in load_wav_to_torch, but might not be in some edge cases, where we should squawk.
    # '2' is arbitrarily chosen since it seems like audio will often "overdrive" the [-1,1] bounds.
    if torch.any(audio > 2) or not torch.any(audio < 0):
        logger.error("Error with %s. Max=%.2f min=%.2f", audiopath, audio.max(), audio.min())
    audio.clip_(-1, 1)


def read_audio_file(audiopath: str):
    if audiopath[-4:] == ".wav":
        audio, lsr = load_wav_to_torch(audiopath)
    elif audiopath[-4:] == ".mp3":
        audio, lsr = librosa.load(audiopath, sr=None)
        audio = torch.FloatTensor(audio)
    else:
        assert False, f"Unsupported audio format provided: {audiopath[-4:]}"

    # Remove any channel data.
    if len(audio.shape) > 1:
        if audio.shape[0] < 5:
            audio = audio[0]
        else:
            assert audio.shape[1] < 5
            audio = audio[:, 0]

    return audio, lsr


def load_required_audio(audiopath: str):
    audio, lsr = read_audio_file(audiopath)

    audios = [torchaudio.functional.resample(audio, lsr, sampling_rate) for sampling_rate in (22050, 24000)]
    for audio in audios:
        check_audio(audio, audiopath)

    return [audio.unsqueeze(0) for audio in audios]


def load_audio(audiopath, sampling_rate):
    audio, lsr = read_audio_file(audiopath)

    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)
    check_audio(audio, audiopath)

    return audio.unsqueeze(0)


TACOTRON_MEL_MAX = 2.3143386840820312
TACOTRON_MEL_MIN = -11.512925148010254


def denormalize_tacotron_mel(norm_mel):
    return ((norm_mel + 1) / 2) * (TACOTRON_MEL_MAX - TACOTRON_MEL_MIN) + TACOTRON_MEL_MIN


def normalize_tacotron_mel(mel):
    return 2 * ((mel - TACOTRON_MEL_MIN) / (TACOTRON_MEL_MAX - TACOTRON_MEL_MIN)) - 1


def wav_to_univnet_mel(wav, do_normalization=False, device="cuda"):
    stft = TorchSTFT(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        use_mel=True,
        n_mels=100,
        sample_rate=24000,
        mel_fmin=0,
        mel_fmax=12000,
    )
    stft = stft.to(device)
    mel = stft(wav)
    mel = amp_to_db(mel)
    if do_normalization:
        mel = normalize_tacotron_mel(mel)
    return mel
