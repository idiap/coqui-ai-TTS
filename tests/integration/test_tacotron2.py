import copy
import os

import pytest
import torch
from torch import nn, optim

from tests import assert_parameters_change, assert_parameters_equal, get_tests_input_path
from tests.integration.test_tacotron import create_stop_targets
from TTS.tts.configs.shared_configs import CapacitronVAEConfig, GSTConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.layers.losses import MSELossMasked
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.utils.audio import AudioProcessor

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

WAV_FILE = os.path.join(get_tests_input_path(), "example_1.wav")


@pytest.fixture
def base_config():
    """Base Tacotron2 config for testing."""
    return Tacotron2Config(num_chars=32, num_speakers=5, out_channels=80, decoder_output_dim=80)


def create_tacotron2_inputs(config, batch_size=8, max_seq_len=128, max_mel_len=30):
    """Create input tensors for Tacotron2 training."""
    input_dummy = torch.randint(0, 24, (batch_size, max_seq_len)).long().to(device)
    input_lengths = torch.randint(100, max_seq_len, (batch_size,)).long().to(device)
    input_lengths = torch.sort(input_lengths, descending=True)[0]
    mel_spec = torch.rand(batch_size, max_mel_len, config.audio["num_mels"]).to(device)
    mel_postnet_spec = torch.rand(batch_size, max_mel_len, config.audio["num_mels"]).to(device)
    mel_lengths = torch.randint(20, max_mel_len, (batch_size,)).long().to(device)
    mel_lengths[0] = max_mel_len
    return input_dummy, input_lengths, mel_spec, mel_postnet_spec, mel_lengths


def train_and_verify_updates(config, aux_input=None, num_iterations=5, ignore_params=None):
    """Train Tacotron2 model and verify parameters are updated."""
    criterion = MSELossMasked(seq_len_norm=False).to(device)
    criterion_st = nn.BCEWithLogitsLoss().to(device)
    model = Tacotron2(config).to(device)
    model.train()
    model_ref = copy.deepcopy(model)
    assert_parameters_equal(model, model_ref)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    for _ in range(num_iterations):
        input_dummy, input_lengths, mel_spec, mel_postnet_spec, mel_lengths = create_tacotron2_inputs(config)
        stop_targets = create_stop_targets(mel_lengths, 8, 30, config.r)
        outputs = model.forward(input_dummy, input_lengths, mel_spec, mel_lengths, aux_input=aux_input)
        assert torch.sigmoid(outputs["stop_tokens"]).data.max() <= 1.0
        assert torch.sigmoid(outputs["stop_tokens"]).data.min() >= 0.0
        optimizer.zero_grad()
        loss = criterion(outputs["decoder_outputs"], mel_spec, mel_lengths)
        stop_loss = criterion_st(outputs["stop_tokens"], stop_targets)
        loss = loss + criterion(outputs["model_outputs"], mel_postnet_spec, mel_lengths) + stop_loss
        loss.backward()
        optimizer.step()
    assert_parameters_change(model, model_ref, ignore=ignore_params)


def test_train_step(base_config):
    """Test vanilla Tacotron2 model."""
    config = base_config.copy()
    config.use_speaker_embedding = False
    config.num_speakers = 1
    train_and_verify_updates(config)


def test_multispeaker_train_step(base_config):
    """Test multi-speaker Tacotron2 with speaker embedding layer."""
    config = base_config.copy()
    config.use_speaker_embedding = True
    config.num_speakers = 5
    config.d_vector_dim = 55

    speaker_ids = torch.randint(0, 5, (8,)).long().to(device)
    train_and_verify_updates(config, aux_input={"speaker_ids": speaker_ids})


def test_gst_train_step_random(base_config):
    """Test multi-speaker Tacotron2 with Global Style Token and Speaker Embedding using random mel."""
    config = base_config.copy()
    config.use_speaker_embedding = True
    config.num_speakers = 10
    config.use_gst = True
    config.gst = GSTConfig()

    speaker_ids = torch.randint(0, 5, (8,)).long().to(device)
    train_and_verify_updates(
        config,
        aux_input={"speaker_ids": speaker_ids},
        num_iterations=10,
        ignore_params=["gst_layer.encoder.recurrence.weight_hh_l0"],
    )


def test_gst_train_step_file(base_config):
    """Test multi-speaker Tacotron2 with Global Style Token using mel from file."""
    config = base_config.copy()
    config.use_speaker_embedding = True
    config.num_speakers = 10
    config.use_gst = True
    config.gst = GSTConfig()

    # Load mel from file
    ap = AudioProcessor(**base_config.audio)
    mel_spec = (
        torch.FloatTensor(ap.melspectrogram(ap.load_wav(WAV_FILE)))[:, :30].unsqueeze(0).transpose(1, 2).to(device)
    )
    mel_spec = mel_spec.repeat(8, 1, 1)

    criterion = MSELossMasked(seq_len_norm=False).to(device)
    criterion_st = nn.BCEWithLogitsLoss().to(device)
    model = Tacotron2(config).to(device)
    model.train()
    model_ref = copy.deepcopy(model)
    assert_parameters_equal(model, model_ref)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    for _ in range(10):
        input_dummy = torch.randint(0, 24, (8, 128)).long().to(device)
        input_lengths = torch.randint(100, 128, (8,)).long().to(device)
        input_lengths = torch.sort(input_lengths, descending=True)[0]
        mel_postnet_spec = torch.rand(8, 30, config.audio["num_mels"]).to(device)
        mel_lengths = torch.randint(20, 30, (8,)).long().to(device)
        mel_lengths[0] = 30
        stop_targets = create_stop_targets(mel_lengths, 8, 30, config.r)
        speaker_ids = torch.randint(0, 5, (8,)).long().to(device)

        outputs = model.forward(
            input_dummy, input_lengths, mel_spec, mel_lengths, aux_input={"speaker_ids": speaker_ids}
        )
        assert torch.sigmoid(outputs["stop_tokens"]).data.max() <= 1.0
        assert torch.sigmoid(outputs["stop_tokens"]).data.min() >= 0.0
        optimizer.zero_grad()
        loss = criterion(outputs["decoder_outputs"], mel_spec, mel_lengths)
        stop_loss = criterion_st(outputs["stop_tokens"], stop_targets)
        loss = loss + criterion(outputs["model_outputs"], mel_postnet_spec, mel_lengths) + stop_loss
        loss.backward()
        optimizer.step()
    assert_parameters_change(model, model_ref, ignore=["gst_layer.encoder.recurrence.weight_hh_l0"])


def test_capacitron_train_step():
    """Test Tacotron2 with Capacitron VAE."""
    config = Tacotron2Config(
        num_chars=32,
        num_speakers=10,
        use_speaker_embedding=True,
        out_channels=80,
        decoder_output_dim=80,
        use_capacitron_vae=True,
        capacitron_vae=CapacitronVAEConfig(),
        optimizer="CapacitronOptimizer",
        optimizer_params={
            "RAdam": {"betas": [0.9, 0.998], "weight_decay": 1e-6},
            "SGD": {"lr": 1e-5, "momentum": 0.9},
        },
    )

    batch = {}
    batch["text_input"] = torch.randint(0, 24, (8, 128)).long().to(device)
    batch["text_lengths"] = torch.randint(100, 129, (8,)).long().to(device)
    batch["text_lengths"] = torch.sort(batch["text_lengths"], descending=True)[0]
    batch["text_lengths"][0] = 128
    batch["mel_input"] = torch.rand(8, 120, config.audio["num_mels"]).to(device)
    batch["mel_lengths"] = torch.randint(20, 120, (8,)).long().to(device)
    batch["mel_lengths"] = torch.sort(batch["mel_lengths"], descending=True)[0]
    batch["mel_lengths"][0] = 120
    batch["stop_targets"] = torch.zeros(8, 120, 1).float().to(device)
    batch["stop_target_lengths"] = torch.randint(0, 120, (8,)).to(device)
    batch["speaker_ids"] = torch.randint(0, 5, (8,)).long().to(device)
    batch["d_vectors"] = None

    for idx in batch["mel_lengths"]:
        batch["stop_targets"][:, int(idx.item()) :, 0] = 1.0

    batch["stop_targets"] = batch["stop_targets"].view(
        batch["text_input"].shape[0], batch["stop_targets"].size(1) // config.r, -1
    )
    batch["stop_targets"] = (batch["stop_targets"].sum(2) > 0.0).unsqueeze(2).float().squeeze()

    model = Tacotron2(config).to(device)
    criterion = model.get_criterion().to(device)
    optimizer = model.get_optimizer()

    model.train()
    model_ref = copy.deepcopy(model)
    assert_parameters_equal(model, model_ref)
    for _ in range(10):
        _, loss_dict = model.train_step(batch, criterion)
        optimizer.zero_grad()
        loss_dict["capacitron_vae_beta_loss"].backward()
        optimizer.first_step()
        loss_dict["loss"].backward()
        optimizer.step()
    assert_parameters_change(model, model_ref)


def test_scgst_multispeaker_train_step(base_config):
    """Test multi-speaker Tacotron2 with Global Style Tokens and d-vector inputs."""
    config = base_config.copy()
    config.use_d_vector_file = True
    config.use_gst = True
    config.gst = GSTConfig()
    config.d_vector_dim = 55

    speaker_embeddings = torch.rand(8, 55).to(device)
    train_and_verify_updates(
        config,
        aux_input={"d_vectors": speaker_embeddings},
        ignore_params=["gst_layer.encoder.recurrence.weight_hh_l0"],
    )
