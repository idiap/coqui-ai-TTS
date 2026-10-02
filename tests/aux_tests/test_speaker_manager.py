"""Test initialization and behaviour of SpeakerManager."""

import os

import numpy as np
import torch

from tests import get_tests_input_path
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.utils.speakers import SpeakerManager

sample_wav_path = os.path.join(get_tests_input_path(), "../data/ljspeech/wavs/LJ001-0001.wav")
sample_wav_path2 = os.path.join(get_tests_input_path(), "../data/ljspeech/wavs/LJ001-0002.wav")
d_vectors_file_path = os.path.join(get_tests_input_path(), "../data/dummy_speakers.json")
d_vectors_file_pth_path = os.path.join(get_tests_input_path(), "../data/dummy_speakers.pth")


def test_speaker_embedding(encoder_model_path, encoder_config_path):
    """Test SpeakerManager for computing d_vectors from waveforms"""
    # load speaker encoder
    manager = SpeakerManager(encoder_model_path=encoder_model_path, encoder_config_path=encoder_config_path)

    # load a sample audio and compute embedding
    waveform = manager.encoder_ap.load_wav(sample_wav_path)
    mel = manager.encoder_ap.melspectrogram(waveform)
    d_vector = manager.compute_embeddings(mel)
    assert d_vector.shape[1] == 256

    # compute d_vector directly from an input file
    d_vector = manager.compute_embedding_from_clip(sample_wav_path)
    d_vector2 = manager.compute_embedding_from_clip(sample_wav_path)
    d_vector = torch.FloatTensor(d_vector)
    d_vector2 = torch.FloatTensor(d_vector2)
    assert d_vector.shape[0] == 256
    assert (d_vector - d_vector2).sum() == 0.0

    # compute d_vector from a list of wav files.
    d_vector3 = manager.compute_embedding_from_clip([sample_wav_path, sample_wav_path2])
    d_vector3 = torch.FloatTensor(d_vector3)
    assert d_vector3.shape[0] == 256
    assert (d_vector - d_vector3).sum() != 0.0


def test_dvector_file_processing():
    """Test SpeakerManager for loading embedding files"""
    manager = SpeakerManager(d_vectors_file_path=d_vectors_file_path)
    assert manager.num_speakers == 1
    assert manager.embedding_dim == 256
    manager = SpeakerManager(d_vectors_file_path=d_vectors_file_pth_path)
    assert manager.num_speakers == 1
    assert manager.embedding_dim == 256
    d_vector = manager.get_embedding_by_clip(manager.clip_ids[0])
    assert len(d_vector) == 256
    d_vectors = manager.get_embeddings_by_name(manager.speaker_names[0])
    assert len(d_vectors[0]) == 256
    d_vector1 = manager.get_mean_embedding(manager.speaker_names[0], num_samples=2, randomize=True)
    assert len(d_vector1) == 256
    d_vector2 = manager.get_mean_embedding(manager.speaker_names[0], num_samples=2, randomize=False)
    assert len(d_vector2) == 256
    assert np.sum(np.array(d_vector1) - np.array(d_vector2)) != 0


def test_init_from_config_empty_speakers():
    """Test with no speakers configured."""
    config = GlowTTSConfig(speakers=[], use_speaker_embedding=True)
    sm = SpeakerManager.init_from_config(config)

    assert sm is None


def test_init_from_config_speakers():
    """Test that speaker IDs are read from config.speakers."""
    config = GlowTTSConfig(speakers=["alice", "bob"], use_speaker_embedding=True)
    sm = SpeakerManager.init_from_config(config)

    assert sm.num_speakers == 2
    assert sm.name_to_id == {"alice": 0, "bob": 1}
