import random
from pathlib import Path

import pytest
import torch

from tests import get_tests_output_path
from TTS.tts.configs.overflow_config import OverflowConfig
from TTS.tts.layers.overflow.common_layers import Encoder, Outputnet, OverflowUtils
from TTS.tts.layers.overflow.decoder import Decoder
from TTS.tts.layers.overflow.neural_hmm import EmissionModel, NeuralHMM, TransitionModel
from TTS.tts.models.overflow import Overflow
from TTS.tts.utils.helpers import sequence_mask


@pytest.fixture(scope="session")
def parameter_path() -> Path:
    """Create and return path to test parameters."""
    path = Path(get_tests_output_path()) / "lj_parameters.pt"
    torch.save({"mean": -5.5138, "std": 2.0636, "init_transition_prob": 0.3212}, path)
    return path


@pytest.fixture
def config():
    """Create a fresh config for each test."""
    return OverflowConfig(num_chars=24)


@pytest.fixture
def create_inputs(device, config):
    """Factory fixture to create test inputs."""

    def _create_inputs(batch_size=8):
        max_len_t, max_len_m = random.randint(25, 50), random.randint(50, 80)
        input_dummy = torch.randint(0, 24, (batch_size, max_len_t)).long().to(device)
        input_lengths = torch.randint(20, max_len_t, (batch_size,)).long().to(device).sort(descending=True)[0]
        input_lengths[0] = max_len_t
        input_dummy = input_dummy * sequence_mask(input_lengths)
        mel_spec = torch.randn(batch_size, max_len_m, config.audio["num_mels"]).to(device)
        mel_lengths = torch.randint(40, max_len_m, (batch_size,)).long().to(device).sort(descending=True)[0]
        mel_lengths[0] = max_len_m
        mel_spec = mel_spec * sequence_mask(mel_lengths).unsqueeze(2)
        return input_dummy, input_lengths, mel_spec, mel_lengths

    return _create_inputs


@pytest.fixture
def get_overflow_model(config, parameter_path, device):
    """Factory fixture to create Overflow model."""

    def _get_model(custom_config=None):
        cfg = custom_config if custom_config is not None else config
        cfg.mel_statistics_parameter_path = parameter_path
        model = Overflow(cfg)
        return model.to(device)

    return _get_model


# Tests for Overflow model


def test_overflow_forward(get_overflow_model, create_inputs):
    model = get_overflow_model()
    input_dummy, input_lengths, mel_spec, mel_lengths = create_inputs()
    outputs = model(input_dummy, input_lengths, mel_spec, mel_lengths)
    assert outputs["log_probs"].shape == (input_dummy.shape[0],)
    assert outputs["alignments"].shape[2] == model.state_per_phone * max(input_lengths)


def test_overflow_inference(get_overflow_model, create_inputs, config):
    model = get_overflow_model()
    input_dummy, input_lengths, mel_spec, mel_lengths = create_inputs()
    output_dict = model.inference(input_dummy)
    assert output_dict["model_outputs"].shape[2] == config.out_channels


def test_overflow_init_from_config(config, parameter_path):
    config.mel_statistics_parameter_path = parameter_path
    config.prenet_dim = 256
    model = Overflow(config)
    assert model.prenet_dim == config.prenet_dim


# Tests for Overflow Encoder


@pytest.fixture
def get_encoder(config, device):
    """Factory fixture to create Encoder."""

    def _get_encoder(state_per_phone):
        return Encoder(config.num_chars, state_per_phone, config.prenet_dim, config.encoder_n_convolutions).to(device)

    return _get_encoder


@pytest.mark.parametrize("state_per_phone", [1, 2, 3])
def test_encoder_forward_with_state_per_phone_multiplication(get_encoder, create_inputs, state_per_phone):
    input_dummy, input_lengths, _, _ = create_inputs()
    model = get_encoder(state_per_phone)
    x, x_len = model(input_dummy, input_lengths)
    assert x.shape[1] == input_dummy.shape[1] * state_per_phone


@pytest.mark.parametrize("state_per_phone", [1, 2, 3])
def test_encoder_inference_with_state_per_phone_multiplication(get_encoder, create_inputs, state_per_phone):
    input_dummy, input_lengths, _, _ = create_inputs()
    model = get_encoder(state_per_phone)
    x, x_len = model.inference(input_dummy, input_lengths)
    assert x.shape[1] == input_dummy.shape[1] * state_per_phone


# Tests for Overflow Utils


@pytest.mark.parametrize(
    "tensor",
    [torch.randn(10), torch.zeros(10), torch.ones(10)],
    ids=["random", "zeros", "ones"],
)
def test_overflow_utils_logsumexp(tensor):
    assert torch.eq(torch.logsumexp(tensor, dim=0), OverflowUtils.logsumexp(tensor, dim=0)).all()


# Tests for Overflow Decoder


@pytest.fixture
def get_decoder(config, device):
    """Factory fixture to create Decoder."""

    def _get_decoder(num_flow_blocks_dec=None, hidden_channels_dec=None):
        num_blocks = num_flow_blocks_dec if num_flow_blocks_dec is not None else config.num_flow_blocks_dec
        hidden_channels = hidden_channels_dec if hidden_channels_dec is not None else config.hidden_channels_dec
        dropout_p_dec = 0.0  # turn off dropout to check invertibility

        return Decoder(
            config.out_channels,
            hidden_channels,
            config.kernel_size_dec,
            config.dilation_rate,
            num_blocks,
            config.num_block_layers,
            dropout_p_dec,
            config.num_splits,
            config.num_squeeze,
            config.sigmoid_scale,
            config.c_in_channels,
        ).to(device)

    return _get_decoder


@pytest.mark.parametrize("num_flow_blocks_dec", [8, None])
@pytest.mark.parametrize("hidden_channels_dec", [100, None])
def test_decoder_forward_backward(get_decoder, create_inputs, num_flow_blocks_dec, hidden_channels_dec):
    decoder = get_decoder(num_flow_blocks_dec, hidden_channels_dec)
    _, _, mel_spec, mel_lengths = create_inputs()
    z, z_len, _ = decoder(mel_spec.transpose(1, 2), mel_lengths)
    mel_spec_, mel_lengths_, _ = decoder(z, z_len, reverse=True)
    mask = sequence_mask(z_len).unsqueeze(1)
    mel_spec = mel_spec[:, : z.shape[2], :].transpose(1, 2) * mask
    z = z * mask
    assert torch.isclose(mel_spec, mel_spec_, atol=1e-2).all(), (
        f"num_flow_blocks_dec={num_flow_blocks_dec}, hidden_channels_dec={hidden_channels_dec}"
    )


# Tests for Neural HMM


@pytest.fixture
def get_neural_hmm(config, device):
    """Factory fixture to create NeuralHMM."""

    def _get_neural_hmm(deterministic_transition=None):
        det_trans = config.deterministic_transition if deterministic_transition is None else deterministic_transition
        return NeuralHMM(
            config.out_channels,
            config.ar_order,
            det_trans,
            config.encoder_in_out_features,
            config.prenet_type,
            config.prenet_dim,
            config.prenet_n_layers,
            config.prenet_dropout,
            config.prenet_dropout_at_inference,
            config.memory_rnn_dim,
            config.outputnet_size,
            config.flat_start_params,
            config.std_floor,
        ).to(device)

    return _get_neural_hmm


@pytest.fixture
def get_embedded_input(create_inputs, config, device):
    """Factory fixture to create embedded inputs."""

    def _get_embedded_input():
        input_dummy, input_lengths, mel_spec, mel_lengths = create_inputs()
        input_dummy = torch.nn.Embedding(config.num_chars, config.encoder_in_out_features).to(device)(input_dummy)
        return input_dummy, input_lengths, mel_spec, mel_lengths

    return _get_embedded_input


def test_neural_hmm_forward(get_neural_hmm, get_embedded_input):
    input_dummy, input_lengths, mel_spec, mel_lengths = get_embedded_input()
    neural_hmm = get_neural_hmm()
    log_prob, log_alpha_scaled, transition_matrix, means = neural_hmm(
        input_dummy, input_lengths, mel_spec.transpose(1, 2), mel_lengths
    )
    assert log_prob.shape == (input_dummy.shape[0],)
    assert log_alpha_scaled.shape == transition_matrix.shape


def test_neural_hmm_mask_lengths(get_neural_hmm, get_embedded_input, device):
    input_dummy, input_lengths, mel_spec, mel_lengths = get_embedded_input()
    neural_hmm = get_neural_hmm()
    log_prob, log_alpha_scaled, transition_matrix, means = neural_hmm(
        input_dummy, input_lengths, mel_spec.transpose(1, 2), mel_lengths
    )
    log_c = torch.randn(mel_spec.shape[0], mel_spec.shape[1], device=device)
    log_c, log_alpha_scaled = neural_hmm._mask_lengths(  # pylint: disable=protected-access
        mel_lengths, log_c, log_alpha_scaled
    )
    assertions = []
    for i in range(mel_spec.shape[0]):
        assertions.append(log_c[i, mel_lengths[i] :].sum() == 0.0)
    assert all(assertions), "Incorrect masking"
    assertions = []
    for i in range(mel_spec.shape[0]):
        assertions.append(log_alpha_scaled[i, mel_lengths[i] :, : input_lengths[i]].sum() == 0.0)
    assert all(assertions), "Incorrect masking"


def test_neural_hmm_process_ar_timestep(get_neural_hmm, get_embedded_input, config):
    model = get_neural_hmm()
    input_dummy, input_lengths, mel_spec, mel_lengths = get_embedded_input()

    h_post_prenet, c_post_prenet = model._init_lstm_states(  # pylint: disable=protected-access
        input_dummy.shape[0], config.memory_rnn_dim, mel_spec
    )
    h_post_prenet, c_post_prenet = model._process_ar_timestep(  # pylint: disable=protected-access
        1,
        mel_spec,
        h_post_prenet,
        c_post_prenet,
    )

    assert h_post_prenet.shape == (input_dummy.shape[0], config.memory_rnn_dim)
    assert c_post_prenet.shape == (input_dummy.shape[0], config.memory_rnn_dim)


def test_neural_hmm_add_go_token(get_neural_hmm, get_embedded_input):
    model = get_neural_hmm()
    input_dummy, input_lengths, mel_spec, mel_lengths = get_embedded_input()

    out = model._add_go_token(mel_spec)  # pylint: disable=protected-access
    assert out.shape == mel_spec.shape
    assert (out[:, 1:] == mel_spec[:, :-1]).all(), "Go token not appended properly"


def test_neural_hmm_forward_algorithm_variables(get_neural_hmm, get_embedded_input, config):
    model = get_neural_hmm()
    input_dummy, input_lengths, mel_spec, mel_lengths = get_embedded_input()

    (
        log_c,
        log_alpha_scaled,
        transition_matrix,
        _,
    ) = model._initialize_forward_algorithm_variables(  # pylint: disable=protected-access
        mel_spec, input_dummy.shape[1] * config.state_per_phone
    )

    assert log_c.shape == (mel_spec.shape[0], mel_spec.shape[1])
    assert log_alpha_scaled.shape == (
        mel_spec.shape[0],
        mel_spec.shape[1],
        input_dummy.shape[1] * config.state_per_phone,
    )
    assert transition_matrix.shape == (
        mel_spec.shape[0],
        mel_spec.shape[1],
        input_dummy.shape[1] * config.state_per_phone,
    )


def test_neural_hmm_get_absorption_state_scaling_factor(get_neural_hmm, get_embedded_input, config):
    model = get_neural_hmm()
    input_dummy, input_lengths, mel_spec, mel_lengths = get_embedded_input()
    input_lengths = input_lengths * config.state_per_phone
    (
        log_c,
        log_alpha_scaled,
        transition_matrix,
        _,
    ) = model._initialize_forward_algorithm_variables(  # pylint: disable=protected-access
        mel_spec, input_dummy.shape[1] * config.state_per_phone
    )
    log_alpha_scaled = torch.rand_like(log_alpha_scaled).clamp(1e-3)
    transition_matrix = torch.randn_like(transition_matrix).sigmoid().log()
    sum_final_log_c = model.get_absorption_state_scaling_factor(
        mel_lengths, log_alpha_scaled, input_lengths, transition_matrix
    )

    text_mask = ~sequence_mask(input_lengths)
    transition_prob_mask = ~model.get_mask_for_last_item(input_lengths, device=input_lengths.device)

    outputs = []

    for i in range(input_dummy.shape[0]):
        last_log_alpha_scaled = log_alpha_scaled[i, mel_lengths[i] - 1].masked_fill(text_mask[i], -float("inf"))
        log_last_transition_probability = OverflowUtils.log_clamped(
            torch.sigmoid(transition_matrix[i, mel_lengths[i] - 1])
        ).masked_fill(transition_prob_mask[i], -float("inf"))
        outputs.append(last_log_alpha_scaled + log_last_transition_probability)

    sum_final_log_c_computed = torch.logsumexp(torch.stack(outputs), dim=1)

    assert torch.isclose(sum_final_log_c_computed, sum_final_log_c).all()


def test_neural_hmm_inference(get_neural_hmm, get_embedded_input, config):
    model = get_neural_hmm()
    input_dummy, input_lengths, mel_spec, mel_lengths = get_embedded_input()
    for temperature in [0.334, 0.667, 1.0]:
        outputs = model.inference(
            input_dummy, input_lengths, temperature, config.max_sampling_time, config.duration_threshold
        )
        assert outputs["hmm_outputs"].shape[-1] == outputs["input_parameters"][0][0][0].shape[-1]
        assert outputs["output_parameters"][0][0][0].shape[-1] == outputs["input_parameters"][0][0][0].shape[-1]
        assert len(outputs["alignments"]) == input_dummy.shape[0]


def test_emission_model(get_embedded_input, config, device):
    model = EmissionModel().to(device)
    input_dummy, input_lengths, mel_spec, mel_lengths = get_embedded_input()
    x_t = torch.randn(input_dummy.shape[0], config.out_channels).to(device)
    means = torch.randn(input_dummy.shape[0], input_dummy.shape[1], config.out_channels).to(device)
    std = torch.rand_like(means).to(device).clamp_(1e-3)  # std should be positive
    out = model(x_t, means, std, input_lengths)
    assert out.shape == (input_dummy.shape[0], input_dummy.shape[1])

    # testing sampling
    for temperature in [0, 0.334, 0.667]:
        out = model.sample(means, std, temperature)
        assert out.shape == means.shape
        if temperature == 0:
            assert torch.isclose(out, means).all()


def test_transition_model(get_embedded_input, device):
    model = TransitionModel().to(device)
    input_dummy, input_lengths, mel_spec, mel_lengths = get_embedded_input()
    prev_t_log_scaled_alph = torch.randn(input_dummy.shape[0], input_lengths.max()).to(input_dummy.device)
    transition_vector = torch.randn(input_lengths.max()).to(input_dummy.device)
    out = model(prev_t_log_scaled_alph, transition_vector, input_lengths)
    assert out.shape == (input_dummy.shape[0], input_lengths.max())


# Tests for Overflow OutputNet


def test_outputnet_forward_with_flat_start(create_inputs, config, device):
    model = Outputnet(
        config.encoder_in_out_features,
        config.memory_rnn_dim,
        config.out_channels,
        config.outputnet_size,
        config.flat_start_params,
        config.std_floor,
    ).to(device)

    input_dummy, input_lengths, mel_spec, mel_lengths = create_inputs()
    input_dummy = torch.nn.Embedding(config.num_chars, config.encoder_in_out_features).to(device)(input_dummy)
    one_timestep_frame = torch.randn(input_dummy.shape[0], config.memory_rnn_dim).to(device)
    mean, std, transition_vector = model(one_timestep_frame, input_dummy)
    assert torch.isclose(mean, torch.tensor(model.flat_start_params["mean"] * 1.0)).all()
    assert torch.isclose(std, torch.tensor(model.flat_start_params["std"] * 1.0)).all()
    assert torch.isclose(transition_vector.sigmoid(), torch.tensor(model.flat_start_params["transition_p"] * 1.0)).all()
