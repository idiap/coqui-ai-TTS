import copy

import torch

from tests import check_parameter_changes
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


def test_train_step():
    # setup the model
    with torch.autograd.set_detect_anomaly(True):
        config = VitsConfig(model_args=VitsArgs(num_chars=32, spec_segment_size=10))
        model = Vits(config).to(device)
        model.train()
        # model to train
        optimizers = model.get_optimizer()
        criterions = model.get_criterion()
        criterions = [criterions[0].to(device), criterions[1].to(device)]
        # reference model to compare model weights
        model_ref = Vits(config).to(device)
        # # pass the state to ref model
        model_ref.load_state_dict(copy.deepcopy(model.state_dict()))
        count = 0
        for param, param_ref in zip(model.parameters(), model_ref.parameters()):
            assert (param - param_ref).sum() == 0, param
            count = count + 1
        for _ in range(5):
            batch = _create_batch(config, 2)
            for idx in [0, 1]:
                outputs, loss_dict = model.train_step(batch, criterions, idx)
                assert outputs
                assert loss_dict
                loss_dict["loss"].backward()
                optimizers[idx].step()
                optimizers[idx].zero_grad()

    check_parameter_changes(model, model_ref)


def test_train_step_upsampling():
    """Upsampling by the decoder upsampling layers"""
    # setup the model
    with torch.autograd.set_detect_anomaly(True):
        audio_config = VitsAudioConfig(sample_rate=22050)
        model_args = VitsArgs(
            num_chars=32,
            spec_segment_size=10,
            encoder_sample_rate=11025,
            interpolate_z=False,
            upsample_rates_decoder=[8, 8, 4, 2],
        )
        config = VitsConfig(model_args=model_args, audio=audio_config)
        model = Vits(config).to(device)
        model.train()
        # model to train
        optimizers = model.get_optimizer()
        criterions = model.get_criterion()
        criterions = [criterions[0].to(device), criterions[1].to(device)]
        # reference model to compare model weights
        model_ref = Vits(config).to(device)
        # # pass the state to ref model
        model_ref.load_state_dict(copy.deepcopy(model.state_dict()))
        count = 0
        for param, param_ref in zip(model.parameters(), model_ref.parameters()):
            assert (param - param_ref).sum() == 0, param
            count = count + 1
        for _ in range(5):
            batch = _create_batch(config, 2)
            for idx in [0, 1]:
                outputs, loss_dict = model.train_step(batch, criterions, idx)
                assert outputs
                assert loss_dict
                loss_dict["loss"].backward()
                optimizers[idx].step()
                optimizers[idx].zero_grad()

    check_parameter_changes(model, model_ref)


def test_train_step_upsampling_interpolation():
    """Upsampling by interpolation"""
    # setup the model
    with torch.autograd.set_detect_anomaly(True):
        audio_config = VitsAudioConfig(sample_rate=22050)
        model_args = VitsArgs(
            num_chars=32,
            spec_segment_size=10,
            encoder_sample_rate=11025,
            interpolate_z=True,
            upsample_rates_decoder=[8, 8, 2, 2],
        )
        config = VitsConfig(model_args=model_args, audio=audio_config)
        model = Vits(config).to(device)
        model.train()
        # model to train
        optimizers = model.get_optimizer()
        criterions = model.get_criterion()
        criterions = [criterions[0].to(device), criterions[1].to(device)]
        # reference model to compare model weights
        model_ref = Vits(config).to(device)
        # # pass the state to ref model
        model_ref.load_state_dict(copy.deepcopy(model.state_dict()))
        count = 0
        for param, param_ref in zip(model.parameters(), model_ref.parameters()):
            assert (param - param_ref).sum() == 0, param
            count = count + 1
        for _ in range(5):
            batch = _create_batch(config, 2)
            for idx in [0, 1]:
                outputs, loss_dict = model.train_step(batch, criterions, idx)
                assert outputs
                assert loss_dict
                loss_dict["loss"].backward()
                optimizers[idx].step()
                optimizers[idx].zero_grad()

    check_parameter_changes(model, model_ref)
