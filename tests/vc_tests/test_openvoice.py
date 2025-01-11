import os
import unittest

import torch

from tests import get_tests_input_path
from TTS.vc.models.openvoice import OpenVoice, OpenVoiceConfig

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

c = OpenVoiceConfig()

WAV_FILE = os.path.join(get_tests_input_path(), "example_1.wav")


class TestOpenVoice(unittest.TestCase):
    @staticmethod
    def _create_inputs_inference():
        source_wav = torch.rand(16100)
        target_wav = torch.rand(16000)
        return source_wav, target_wav

    def test_load_audio(self):
        config = OpenVoiceConfig()
        model = OpenVoice(config).to(device)
        wav = model.load_audio(WAV_FILE)
        wav2 = model.load_audio(wav)
        assert all(torch.isclose(wav, wav2))

    def test_voice_conversion(self):
        config = OpenVoiceConfig()
        model = OpenVoice(config).to(device)
        model.eval()

        source_wav, target_wav = self._create_inputs_inference()
        output_wav = model.voice_conversion(source_wav, target_wav)
        assert output_wav.shape[0] == source_wav.shape[0] - source_wav.shape[0] % config.audio.hop_length, (
            f"{output_wav.shape} != {source_wav.shape}"
        )
