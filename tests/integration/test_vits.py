import copy

import pytest
import torch

from tests import assert_parameters_change, assert_parameters_equal
from tests.tts_tests.test_vits import _create_inputs
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _create_batch(config, batch_size):
    input_dummy, input_lengths, mel, spec, mel_lengths, _ = _create_inputs(config, batch_size)
    batch = {}
    batch["tokens"] = input_dummy
    batch["token_lens"] = input_lengths
    batch["spec_lens"] = mel_lengths
    batch["mel_lens"] = mel_lengths
    batch["spec"] = spec
    batch["mel"] = mel
    batch["waveform"] = torch.rand(batch_size, 1, config.audio["sample_rate"] * 10).to(device)
    batch["d_vectors"] = None
    batch["speaker_ids"] = None
    batch["language_ids"] = None
    return batch


def _train_and_check_updates(config, num_iterations=5):
    """Train model and verify parameters are updated."""
    with torch.autograd.set_detect_anomaly(True):
        model = Vits(config).to(device)
        model.train()
        # model to train
        optimizers = model.get_optimizer()
        criterions = model.get_criterion()
        criterions = [criterions[0].to(device), criterions[1].to(device)]
        # reference model to compare model weights
        model_ref = Vits(config).to(device)
        # pass the state to ref model
        model_ref.load_state_dict(copy.deepcopy(model.state_dict()))
        assert_parameters_equal(model, model_ref)
        for _ in range(num_iterations):
            batch = _create_batch(config, 2)
            for idx in [0, 1]:
                outputs, loss_dict = model.train_step(batch, criterions, idx)
                assert outputs
                assert loss_dict
                loss_dict["loss"].backward()
                optimizers[idx].step()
                optimizers[idx].zero_grad()

    assert_parameters_change(model, model_ref)


@pytest.mark.parametrize(
    "encoder_sample_rate,interpolate_z,upsample_rates,sample_rate,test_id",
    [
        (None, None, None, None, "base"),
        (11025, False, [8, 8, 4, 2], 22050, "upsampling"),
        (11025, True, [8, 8, 2, 2], 22050, "upsampling_interpolation"),
    ],
)
def test_train_step(encoder_sample_rate, interpolate_z, upsample_rates, sample_rate, test_id):
    """Test VITS training with different upsampling configurations.

    Tests:
    - base: Standard VITS training
    - upsampling: Upsampling by the decoder upsampling layers
    - upsampling_interpolation: Upsampling by interpolation
    """
    model_args = VitsArgs(num_chars=32, spec_segment_size=10)

    if encoder_sample_rate is not None:
        model_args.encoder_sample_rate = encoder_sample_rate
        model_args.interpolate_z = interpolate_z
        model_args.upsample_rates_decoder = upsample_rates
        audio_config = VitsAudioConfig(sample_rate=sample_rate)
        config = VitsConfig(model_args=model_args, audio=audio_config)
    else:
        config = VitsConfig(model_args=model_args)

    _train_and_check_updates(config)
