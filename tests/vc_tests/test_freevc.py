import os
import unittest

import torch
from trainer.generic_utils import count_parameters

from tests import get_tests_input_path
from TTS.vc.models.freevc import FreeVC, FreeVCConfig

# pylint: disable=unused-variable
# pylint: disable=no-self-use

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

c = FreeVCConfig()

WAV_FILE = os.path.join(get_tests_input_path(), "example_1.wav")
BATCH_SIZE = 3


class TestFreeVC(unittest.TestCase):
    def _create_inputs(self, config, batch_size=2):
        spec = torch.rand(batch_size, 30, config.audio["filter_length"] // 2 + 1).to(device)
        mel = torch.rand(batch_size, 30, config.audio["n_mel_channels"]).to(device)
        spec_lengths = torch.randint(20, 30, (batch_size,)).long().to(device)
        spec_lengths[-1] = spec.size(2)
        waveform = torch.rand(batch_size, spec.size(2) * config.audio["hop_length"]).to(device)
        return mel, spec, spec_lengths, waveform

    @staticmethod
    def _create_inputs_inference():
        source_wav = torch.rand(15999)
        target_wav = torch.rand(16000)
        return source_wav, target_wav

    def test_methods(self):
        config = FreeVCConfig()
        model = FreeVC(config).to(device)
        model.load_pretrained_speaker_encoder()
        model.init_multispeaker(config)
        wavlm_feats = model.extract_wavlm_features(torch.rand(1, 16000))
        assert wavlm_feats.shape == (1, 1024, 49), wavlm_feats.shape

    def test_load_audio(self):
        config = FreeVCConfig()
        model = FreeVC(config).to(device)
        wav = model.load_audio(WAV_FILE)
        wav2 = model.load_audio(wav)
        assert all(torch.isclose(wav, wav2))

    def _test_forward(self, batch_size):
        # create model
        config = FreeVCConfig()
        model = FreeVC(config).to(device)
        model.train()
        print(f" > Num parameters for FreeVC model:{count_parameters(model)}")

        mel, spec, spec_lengths, waveform = self._create_inputs(config, batch_size)

        wavlm_vec = model.extract_wavlm_features(waveform)
        wavlm_vec_lengths = torch.ones(batch_size, dtype=torch.long)

        y = model.forward(wavlm_vec, spec, None, mel, spec_lengths, wavlm_vec_lengths)
        # TODO: assert with training implementation

    def test_forward(self):
        self._test_forward(1)
        self._test_forward(3)

    def _test_inference(self, batch_size):
        config = FreeVCConfig()
        model = FreeVC(config).to(device)
        model.eval()

        mel, _, _, waveform = self._create_inputs(config, batch_size)

        wavlm_vec = model.extract_wavlm_features(waveform)
        wavlm_vec_lengths = torch.ones(batch_size, dtype=torch.long)

        output_wav = model.inference(wavlm_vec, None, mel, wavlm_vec_lengths)
        assert output_wav.shape[-1] // config.audio.hop_length == wavlm_vec.shape[-1], (
            f"{output_wav.shape[-1] // config.audio.hop_length} != {wavlm_vec.shape}"
        )

    def test_inference(self):
        self._test_inference(1)
        self._test_inference(3)

    def test_voice_conversion(self):
        config = FreeVCConfig()
        model = FreeVC(config).to(device)
        model.eval()

        source_wav, target_wav = self._create_inputs_inference()
        output_wav = model.voice_conversion(source_wav, target_wav)
        assert output_wav.shape[0] == source_wav.shape[0] - source_wav.shape[0] % config.audio.hop_length, (
            f"{output_wav.shape} != {source_wav.shape}, {config.audio.hop_length}"
        )

    def test_train_step(self): ...

    def test_train_eval_log(self): ...

    def test_test_run(self): ...

    def test_load_checkpoint(self): ...

    def test_get_criterion(self): ...

    def test_init_from_config(self): ...
