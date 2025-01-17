import pytest

from tests.integration import create_config, run_tts_train
from TTS.tts.configs.align_tts_config import AlignTTSConfig
from TTS.tts.configs.delightful_tts_config import DelightfulTTSConfig
from TTS.tts.configs.fast_pitch_config import FastPitchConfig
from TTS.tts.configs.fastspeech2_config import Fastspeech2Config
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.neuralhmm_tts_config import NeuralhmmTTSConfig
from TTS.tts.configs.overflow_config import OverflowConfig
from TTS.tts.configs.speedy_speech_config import SpeedySpeechConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.configs.tacotron_config import TacotronConfig
from TTS.tts.configs.vits_config import VitsConfig

SPEAKER_ARGS = (
    {},
    {
        "use_d_vector_file": True,
        "d_vector_file": "tests/data/ljspeech/speakers.json",
        "d_vector_dim": 256,
    },
    {
        "use_speaker_embedding": True,
        "num_speakers": 4,
    },
)
SPEAKER_ARG_IDS = ["single", "dvector", "speaker_emb"]


def test_train_align_tts(tmp_path):
    config = create_config(AlignTTSConfig, use_phonemes=False)
    run_tts_train(tmp_path, config)


@pytest.mark.parametrize("speaker_args", SPEAKER_ARGS, ids=SPEAKER_ARG_IDS)
def test_train_delightful_tts(tmp_path, speaker_args):
    config = create_config(
        DelightfulTTSConfig,
        batch_size=2,
        f0_cache_path=tmp_path / "f0_cache",  # delightful f0 cache is incompatible with other models
        binary_align_loss_alpha=0.0,
        use_attn_priors=False,
        **speaker_args,
    )
    run_tts_train(tmp_path, config)


@pytest.mark.parametrize("speaker_args", SPEAKER_ARGS, ids=SPEAKER_ARG_IDS)
def test_train_fast_pitch(tmp_path, speaker_args):
    config = create_config(FastPitchConfig, f0_cache_path="tests/data/ljspeech/f0_cache", **speaker_args)
    config.audio.signal_norm = False
    config.audio.mel_fmax = 8000
    config.audio.spec_gain = 1
    config.audio.log_func = "np.log"
    run_tts_train(tmp_path, config)


@pytest.mark.parametrize("speaker_args", SPEAKER_ARGS, ids=SPEAKER_ARG_IDS)
def test_train_fast_speech2(tmp_path, speaker_args):
    config = create_config(
        Fastspeech2Config,
        f0_cache_path="tests/data/ljspeech/f0_cache",
        energy_cache_path=tmp_path / "energy_cache",
        **speaker_args,
    )
    config.audio.signal_norm = False
    config.audio.mel_fmax = 8000
    config.audio.spec_gain = 1
    config.audio.log_func = "np.log"
    run_tts_train(tmp_path, config)


@pytest.mark.parametrize("speaker_args", SPEAKER_ARGS, ids=SPEAKER_ARG_IDS)
def test_train_glow_tts(tmp_path, speaker_args):
    config = create_config(GlowTTSConfig, batch_size=2, data_dep_init_steps=1, **speaker_args)
    run_tts_train(tmp_path, config)


def test_train_neuralhmm(tmp_path):
    config = create_config(NeuralhmmTTSConfig, batch_size=3, eval_batch_size=3, max_sampling_time=50)
    run_tts_train(tmp_path, config)


def test_train_overflow(tmp_path):
    config = create_config(OverflowConfig, batch_size=3, eval_batch_size=3, max_sampling_time=50)
    run_tts_train(tmp_path, config)


def test_train_speedy_speech(tmp_path):
    config = create_config(SpeedySpeechConfig)
    run_tts_train(tmp_path, config)


def test_train_tacotron(tmp_path):
    config = create_config(TacotronConfig, use_phonemes=False, r=5, max_decoder_steps=50)
    run_tts_train(tmp_path, config)


@pytest.mark.parametrize("speaker_args", SPEAKER_ARGS, ids=SPEAKER_ARG_IDS)
def test_train_tacotron2(tmp_path, speaker_args):
    config = create_config(Tacotron2Config, use_phonemes=False, r=5, max_decoder_steps=50, **speaker_args)
    run_tts_train(tmp_path, config)


@pytest.mark.parametrize("speaker_args", SPEAKER_ARGS, ids=SPEAKER_ARG_IDS)
def test_train_vits(tmp_path, speaker_args):
    config = create_config(VitsConfig, batch_size=2, eval_batch_size=2, **speaker_args)
    run_tts_train(tmp_path, config)
