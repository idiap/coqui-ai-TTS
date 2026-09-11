import json
from pathlib import Path

import pytest
import torch
from trainer.logging.tensorboard_logger import TensorboardLogger

from tests import get_test_speakers, get_tests_data_path, get_tests_input_path, get_tests_output_path
from TTS.tts.configs.vits_config import VitsArgs, VitsConfig
from TTS.tts.models.vits import Vits, load_audio
from TTS.utils.audio.torch_transforms import amp_to_db, db_to_amp, spec_to_mel, wav_to_mel, wav_to_spec

LANGUAGES = ["en", "fr-fr", "pt-br"]
WAV_FILE = Path(get_tests_input_path()) / "example_1.wav"


@pytest.fixture
def language_ids_file(tmp_path: Path) -> Path:
    """Create a temporary language IDs JSON file."""
    lang_ids = {name: i for i, name in enumerate(LANGUAGES)}
    language_ids_file = tmp_path / "language_ids.json"
    language_ids_file.write_text(json.dumps(lang_ids))
    return language_ids_file


def _create_inputs(config, device, batch_size=2):
    input_dummy = torch.randint(0, 24, (batch_size, 128)).long().to(device)
    input_lengths = torch.randint(100, 129, (batch_size,)).long().to(device)
    input_lengths[-1] = 128
    spec = torch.rand(batch_size, config.audio["fft_size"] // 2 + 1, 30).to(device)
    mel = torch.rand(batch_size, config.audio["num_mels"], 30).to(device)
    spec_lengths = torch.randint(20, 30, (batch_size,)).long().to(device)
    spec_lengths[-1] = spec.size(2)
    waveform = torch.rand(batch_size, 1, spec.size(2) * config.audio["hop_length"]).to(device)
    return input_dummy, input_lengths, mel, spec, spec_lengths, waveform


def _check_forward_outputs(config, output_dict, encoder_config=None, batch_size=2):
    assert output_dict["model_outputs"].shape[2] == config.model_args.spec_segment_size * config.audio["hop_length"]
    assert output_dict["alignments"].shape == (batch_size, 128, 30)
    assert output_dict["alignments"].max() == 1
    assert output_dict["alignments"].min() == 0
    assert output_dict["z"].shape == (batch_size, config.model_args.hidden_channels, 30)
    assert output_dict["z_p"].shape == (batch_size, config.model_args.hidden_channels, 30)
    assert output_dict["m_p"].shape == (batch_size, config.model_args.hidden_channels, 30)
    assert output_dict["logs_p"].shape == (batch_size, config.model_args.hidden_channels, 30)
    assert output_dict["m_q"].shape == (batch_size, config.model_args.hidden_channels, 30)
    assert output_dict["logs_q"].shape == (batch_size, config.model_args.hidden_channels, 30)
    assert output_dict["waveform_seg"].shape[2] == config.model_args.spec_segment_size * config.audio["hop_length"]
    if encoder_config:
        assert output_dict["gt_spk_emb"].shape == (batch_size, encoder_config.model_params["proj_dim"])
        assert output_dict["syn_spk_emb"].shape == (batch_size, encoder_config.model_params["proj_dim"])
    else:
        assert output_dict["gt_spk_emb"] is None
        assert output_dict["syn_spk_emb"] is None


def _check_inference_outputs(config, outputs, input_dummy, batch_size=1):
    feat_len = outputs["z"].shape[2]
    assert outputs["model_outputs"].shape[:2] == (batch_size, 1)  # we don't know the channel dimension
    assert outputs["alignments"].shape == (batch_size, input_dummy.shape[1], feat_len)
    assert outputs["z"].shape == (batch_size, config.model_args.hidden_channels, feat_len)
    assert outputs["z_p"].shape == (batch_size, config.model_args.hidden_channels, feat_len)
    assert outputs["m_p"].shape == (batch_size, config.model_args.hidden_channels, feat_len)
    assert outputs["logs_p"].shape == (batch_size, config.model_args.hidden_channels, feat_len)


def _create_batch(config, device, batch_size):
    input_dummy, input_lengths, mel, spec, mel_lengths, _ = _create_inputs(config, device, batch_size)
    return {
        "tokens": input_dummy,
        "token_lens": input_lengths,
        "spec_lens": mel_lengths,
        "mel_lens": mel_lengths,
        "spec": spec,
        "mel": mel,
        "waveform": torch.rand(batch_size, 1, config.audio["sample_rate"] * 10).to(device),
        "d_vectors": None,
        "speaker_ids": None,
        "language_ids": None,
    }


def test_load_audio():
    wav, sr = load_audio(WAV_FILE)
    assert wav.shape == (1, 41885)
    assert sr == 22050

    spec = wav_to_spec(wav, n_fft=1024, hop_length=512, win_length=1024, center=False)
    mel = wav_to_mel(
        wav,
        n_fft=1024,
        num_mels=80,
        sample_rate=sr,
        hop_length=512,
        win_length=1024,
        fmin=0,
        fmax=8000,
        center=False,
    )
    mel2 = spec_to_mel(spec, n_fft=1024, num_mels=80, sample_rate=sr, fmin=0, fmax=8000)

    assert (mel - mel2).abs().max() == 0
    assert spec.shape[0] == mel.shape[0]
    assert spec.shape[2] == mel.shape[2]

    spec_db = amp_to_db(spec)
    spec_amp = db_to_amp(spec_db)

    assert (spec - spec_amp).abs().max() == pytest.approx(0, abs=1e-4)


def test_init_multispeaker():
    args = VitsArgs(use_speaker_embedding=True)
    model = Vits(VitsConfig(model_args=args, speakers=get_test_speakers(10)))
    assert hasattr(model, "emb_g")

    args = VitsArgs(use_speaker_embedding=True)
    with pytest.raises(ValueError, match="No SpeakerManager provided"):
        model = Vits(VitsConfig(model_args=args))

    args = VitsArgs(use_speaker_embedding=False)
    model = Vits(VitsConfig(model_args=args, speakers=get_test_speakers(10)))
    assert not hasattr(model, "emb_g")

    args = VitsArgs(d_vector_dim=101, use_d_vector_file=True)
    model = Vits(VitsConfig(model_args=args))
    assert model.embedded_speaker_dim == 101


def test_init_multilingual(language_ids_file):
    args = VitsArgs(language_ids_file=None, use_language_embedding=False)
    config = VitsConfig(model_args=args)
    model = Vits(config)
    assert model.language_manager.num_languages == 1
    assert model.language_manager.language_names == ["en"]
    assert config.languages == []
    assert model.embedded_language_dim == 0
    assert not hasattr(model, "emb_l")

    args = VitsArgs(language_ids_file=language_ids_file)
    config = VitsConfig(model_args=args)
    model = Vits(config)
    assert model.language_manager.num_languages == 1
    assert model.language_manager.language_names == ["en"]
    assert config.languages == []
    assert model.embedded_language_dim == 0
    assert not hasattr(model, "emb_l")

    args = VitsArgs(language_ids_file=language_ids_file, use_language_embedding=True)
    model = Vits(VitsConfig(model_args=args))
    assert model.language_manager.num_languages == 3
    assert model.language_manager.language_names == LANGUAGES
    assert model.embedded_language_dim == args.embedded_language_dim
    assert hasattr(model, "emb_l")

    args = VitsArgs(language_ids_file=language_ids_file, use_language_embedding=True, embedded_language_dim=102)
    model = Vits(VitsConfig(model_args=args))
    assert model.language_manager.num_languages == 3
    assert model.language_manager.language_names == LANGUAGES
    assert model.embedded_language_dim == args.embedded_language_dim
    assert hasattr(model, "emb_l")


def test_voice_conversion():
    num_speakers = 10
    spec_len = 101
    spec_effective_len = 50

    args = VitsArgs(use_speaker_embedding=True)
    model = Vits(VitsConfig(model_args=args, speakers=get_test_speakers(num_speakers)))

    ref_inp = torch.randn(1, 513, spec_len)
    ref_inp_len = torch.randint(1, spec_effective_len, (1,))
    ref_spk_id = torch.randint(1, num_speakers, (1,)).item()
    tgt_spk_id = torch.randint(1, num_speakers, (1,)).item()
    o_hat, y_mask, (z, z_p, z_hat) = model.inference_voice_conversion(ref_inp, ref_inp_len, ref_spk_id, tgt_spk_id)

    assert o_hat.shape == (1, 1, spec_len * 256)
    assert y_mask.shape == (1, 1, spec_len)
    assert y_mask.sum() == ref_inp_len[0]
    assert z.shape == (1, args.hidden_channels, spec_len)
    assert z_p.shape == (1, args.hidden_channels, spec_len)
    assert z_hat.shape == (1, args.hidden_channels, spec_len)


def test_forward(device):
    config = VitsConfig(use_speaker_embedding=True)
    config.model_args.spec_segment_size = 10
    input_dummy, input_lengths, _, spec, spec_lengths, waveform = _create_inputs(config, device)
    model = Vits(config).to(device)
    output_dict = model.forward(input_dummy, input_lengths, spec, spec_lengths, waveform)
    _check_forward_outputs(config, output_dict)


def test_multispeaker_forward(device):
    num_speakers = 10

    config = VitsConfig(speakers=get_test_speakers(num_speakers), use_speaker_embedding=True)
    config.model_args.spec_segment_size = 10

    input_dummy, input_lengths, _, spec, spec_lengths, waveform = _create_inputs(config, device)
    speaker_ids = torch.randint(0, num_speakers, (8,)).long().to(device)

    model = Vits(config).to(device)
    output_dict = model.forward(
        input_dummy, input_lengths, spec, spec_lengths, waveform, aux_input={"speaker_ids": speaker_ids}
    )
    _check_forward_outputs(config, output_dict)


def test_d_vector_forward(device):
    batch_size = 2
    args = VitsArgs(
        spec_segment_size=10,
        num_chars=32,
        use_d_vector_file=True,
        d_vector_dim=256,
        d_vector_file=[str(Path(get_tests_data_path()) / "dummy_speakers.json")],
    )
    config = VitsConfig(model_args=args)
    model = Vits(config).to(device)
    model.train()
    input_dummy, input_lengths, _, spec, spec_lengths, waveform = _create_inputs(config, device, batch_size=batch_size)
    d_vectors = torch.randn(batch_size, 256).to(device)
    output_dict = model.forward(
        input_dummy, input_lengths, spec, spec_lengths, waveform, aux_input={"d_vectors": d_vectors}
    )
    _check_forward_outputs(config, output_dict)


def test_multilingual_forward(device, language_ids_file):
    num_speakers = 10
    num_langs = 3
    batch_size = 2

    args = VitsArgs(language_ids_file=language_ids_file, use_language_embedding=True, spec_segment_size=10)
    config = VitsConfig(speakers=get_test_speakers(num_speakers), use_speaker_embedding=True, model_args=args)

    input_dummy, input_lengths, _, spec, spec_lengths, waveform = _create_inputs(config, device, batch_size=batch_size)
    speaker_ids = torch.randint(0, num_speakers, (batch_size,)).long().to(device)
    lang_ids = torch.randint(0, num_langs, (batch_size,)).long().to(device)

    model = Vits(config).to(device)
    output_dict = model.forward(
        input_dummy,
        input_lengths,
        spec,
        spec_lengths,
        waveform,
        aux_input={"speaker_ids": speaker_ids, "language_ids": lang_ids},
    )
    _check_forward_outputs(config, output_dict)


@pytest.mark.parametrize("encoder_config_path", [{"use_torch_spec": True}], indirect=True)
def test_secl_forward(device, language_ids_file, encoder_model_path, encoder_config_path):
    num_speakers = 10
    num_langs = 3
    batch_size = 2

    args = VitsArgs(
        language_ids_file=language_ids_file,
        use_language_embedding=True,
        spec_segment_size=10,
        use_speaker_encoder_as_loss=True,
        speaker_encoder_model_path=encoder_model_path,
        speaker_encoder_config_path=encoder_config_path,
    )
    config = VitsConfig(speakers=get_test_speakers(num_speakers), use_speaker_embedding=True, model_args=args)
    config.audio.sample_rate = 16000

    input_dummy, input_lengths, _, spec, spec_lengths, waveform = _create_inputs(config, device, batch_size=batch_size)
    speaker_ids = torch.randint(0, num_speakers, (batch_size,)).long().to(device)
    lang_ids = torch.randint(0, num_langs, (batch_size,)).long().to(device)

    model = Vits(config).to(device)
    output_dict = model.forward(
        input_dummy,
        input_lengths,
        spec,
        spec_lengths,
        waveform,
        aux_input={"speaker_ids": speaker_ids, "language_ids": lang_ids},
    )
    _check_forward_outputs(config, output_dict, model.speaker_manager.encoder_config)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_inference(device, batch_size):
    config = VitsConfig(use_speaker_embedding=True)
    model = Vits(config).to(device)

    input_dummy, input_lengths, *_ = _create_inputs(config, device, batch_size=batch_size)
    aux_input = {"x_lengths": input_lengths} if batch_size > 1 else None
    outputs = model.inference(input_dummy, aux_input=aux_input)
    _check_inference_outputs(config, outputs, input_dummy, batch_size=batch_size)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_multispeaker_inference(device, batch_size):
    num_speakers = 10
    config = VitsConfig(speakers=get_test_speakers(num_speakers), use_speaker_embedding=True)
    model = Vits(config).to(device)

    input_dummy, input_lengths, *_ = _create_inputs(config, device, batch_size=batch_size)
    speaker_ids = torch.randint(0, num_speakers, (batch_size,)).long().to(device)
    aux_input = {"speaker_ids": speaker_ids}
    if batch_size > 1:
        aux_input["x_lengths"] = input_lengths
    outputs = model.inference(input_dummy, aux_input)
    _check_inference_outputs(config, outputs, input_dummy, batch_size=batch_size)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_multilingual_inference(device, language_ids_file, batch_size):
    num_speakers = 10
    num_langs = 3
    args = VitsArgs(language_ids_file=language_ids_file, use_language_embedding=True, spec_segment_size=10)
    config = VitsConfig(speakers=get_test_speakers(num_speakers), use_speaker_embedding=True, model_args=args)
    model = Vits(config).to(device)

    input_dummy, input_lengths, *_ = _create_inputs(config, device, batch_size=batch_size)
    speaker_ids = torch.randint(0, num_speakers, (batch_size,)).long().to(device)
    lang_ids = torch.randint(0, num_langs, (batch_size,)).long().to(device)
    aux_input = {"speaker_ids": speaker_ids, "language_ids": lang_ids}
    if batch_size > 1:
        aux_input["x_lengths"] = input_lengths
    outputs = model.inference(input_dummy, aux_input)
    _check_inference_outputs(config, outputs, input_dummy, batch_size=batch_size)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_d_vector_inference(device, batch_size):
    args = VitsArgs(
        spec_segment_size=10,
        num_chars=32,
        use_d_vector_file=True,
        d_vector_dim=256,
        d_vector_file=[str(Path(get_tests_data_path()) / "dummy_speakers.json")],
    )
    config = VitsConfig(model_args=args)
    model = Vits(config).to(device)
    model.eval()

    input_dummy, input_lengths, *_ = _create_inputs(config, device, batch_size=batch_size)
    d_vectors = torch.randn(batch_size, 256).to(device)
    aux_input = {"d_vectors": d_vectors}
    if batch_size > 1:
        aux_input["x_lengths"] = input_lengths
    outputs = model.inference(input_dummy, aux_input=aux_input)
    _check_inference_outputs(config, outputs, input_dummy, batch_size=batch_size)


def test_train_eval_log(device):
    batch_size = 2
    config = VitsConfig(model_args=VitsArgs(num_chars=32, spec_segment_size=10))
    model = Vits(config).to(device)
    model.run_data_dep_init = False
    model.train()
    batch = _create_batch(config, device, batch_size)
    logger = TensorboardLogger(
        log_dir=str(Path(get_tests_output_path()) / "dummy_vits_logs"), model_name="vits_test_train_log"
    )
    criterion = model.get_criterion()
    criterion = [criterion[0].to(device), criterion[1].to(device)]
    outputs = [None] * 2
    outputs[0], _ = model.train_step(batch, criterion, 0)
    outputs[1], _ = model.train_step(batch, criterion, 1)
    model.train_log(batch, outputs, logger, None, 1)

    model.eval_log(batch, outputs, logger, None, 1)
    logger.finish()


def test_test_run(device):
    config = VitsConfig(model_args=VitsArgs(num_chars=32))
    model = Vits(config).to(device)
    model.run_data_dep_init = False
    model.eval()
    test_figures, test_audios = model.test_run(None)
    assert test_figures is not None
    assert test_audios is not None


def test_load_checkpoint(device):
    chkp_path = str(Path(get_tests_output_path()) / "dummy_glow_tts_checkpoint.pth")
    config = VitsConfig(VitsArgs(num_chars=32))
    model = Vits(config).to(device)
    chkp = {}
    chkp["model"] = model.state_dict()
    torch.save(chkp, chkp_path)
    model.load_checkpoint(config, chkp_path)
    assert model.training
    model.load_checkpoint(config, chkp_path, eval=True)
    assert not model.training


def test_get_criterion(device):
    config = VitsConfig(VitsArgs(num_chars=32))
    model = Vits(config).to(device)
    criterion = model.get_criterion()
    assert criterion is not None


def test_init_from_config(device):
    config = VitsConfig(model_args=VitsArgs(num_chars=32))
    model = Vits(config).to(device)

    config = VitsConfig(model_args=VitsArgs(num_chars=32), speakers=get_test_speakers(2))
    model = Vits(config).to(device)
    assert not hasattr(model, "emb_g")

    config = VitsConfig(model_args=VitsArgs(num_chars=32, use_speaker_embedding=True), speakers=get_test_speakers(2))
    model = Vits(config).to(device)
    assert model.num_speakers == 2
    assert hasattr(model, "emb_g")

    config = VitsConfig(
        model_args=VitsArgs(
            num_chars=32,
            use_speaker_embedding=True,
            speakers_file=str(Path(get_tests_data_path()) / "ljspeech" / "speakers.json"),
        ),
    )
    model = Vits(config).to(device)
    assert model.num_speakers == 10
    assert hasattr(model, "emb_g")

    config = VitsConfig(
        model_args=VitsArgs(
            num_chars=32,
            use_d_vector_file=True,
            d_vector_dim=256,
            d_vector_file=[str(Path(get_tests_data_path()) / "dummy_speakers.json")],
        )
    )
    model = Vits(config).to(device)
    assert model.num_speakers == 1
    assert not hasattr(model, "emb_g")
    assert model.embedded_speaker_dim == config.d_vector_dim


def test_model_creates_speaker_manager_from_config():
    """Test that model creates SpeakerManager from config.speakers."""
    speakers = ["speaker_a", "speaker_b"]
    config = VitsConfig(speakers=speakers, model_args=VitsArgs(use_speaker_embedding=True))
    model = Vits(config)

    assert model.speaker_manager is not None
    assert model.speaker_manager.num_speakers == 2
    assert model.speaker_manager.speaker_names == speakers
