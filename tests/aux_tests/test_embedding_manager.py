from pathlib import Path

import numpy as np
import pytest
import torch

from tests import get_tests_input_path
from TTS.tts.utils.managers import EmbeddingManager

sample_wav_path = Path(get_tests_input_path()) / "../data/ljspeech/wavs/LJ001-0001.wav"
sample_wav_path2 = Path(get_tests_input_path()) / "../data/ljspeech/wavs/LJ001-0002.wav"
embedding_file_path = Path(get_tests_input_path()) / "../data/dummy_speakers.json"
embeddings_file_path2 = Path(get_tests_input_path()) / "../data/dummy_speakers2.json"
embeddings_file_pth_path = Path(get_tests_input_path()) / "../data/dummy_speakers.pth"


def test_speaker_embedding(encoder_model_path: Path, encoder_config_path: Path):
    """Test EmbeddingManager for loading embedding files and computing embeddings from waveforms"""
    # load speaker encoder
    manager = EmbeddingManager(encoder_model_path=encoder_model_path, encoder_config_path=encoder_config_path)

    # load a sample audio and compute embedding
    waveform = manager.encoder_ap.load_wav(sample_wav_path)
    mel = manager.encoder_ap.melspectrogram(waveform)
    embedding = manager.compute_embeddings(mel)
    assert embedding.shape[1] == 256

    # compute embedding directly from an input file
    embedding = manager.compute_embedding_from_clip(sample_wav_path)
    embedding2 = manager.compute_embedding_from_clip(sample_wav_path)
    embedding = torch.FloatTensor(embedding)
    embedding2 = torch.FloatTensor(embedding2)
    assert embedding.shape[0] == 256
    assert (embedding - embedding2).sum() == 0.0

    # compute embedding from a list of wav files.
    embedding3 = manager.compute_embedding_from_clip([sample_wav_path, sample_wav_path2])
    embedding3 = torch.FloatTensor(embedding3)
    assert embedding3.shape[0] == 256
    assert (embedding - embedding3).sum() != 0.0


def test_embedding_file_processing():
    """Test EmbeddingManager for querying embeddings"""
    manager = EmbeddingManager(embedding_file_path=embeddings_file_pth_path)
    # test embedding querying
    embedding = manager.get_embedding_by_clip(manager.clip_ids[0])
    assert len(embedding) == 256
    embeddings = manager.get_embeddings_by_name(manager.embedding_names[0])
    assert len(embeddings[0]) == 256
    embedding1 = manager.get_mean_embedding(manager.embedding_names[0], num_samples=2, randomize=True)
    assert len(embedding1) == 256
    embedding2 = manager.get_mean_embedding(manager.embedding_names[0], num_samples=2, randomize=False)
    assert len(embedding2) == 256
    assert np.sum(np.array(embedding1) - np.array(embedding2)) != 0


def test_embedding_file_loading():
    """Test EmbeddingManager for loading different file formats"""
    # test loading a json file
    manager = EmbeddingManager(embedding_file_path=embedding_file_path)
    assert manager.num_embeddings == 384
    assert manager.embedding_dim == 256
    # test loading a pth file
    manager = EmbeddingManager(embedding_file_path=embeddings_file_pth_path)
    assert manager.num_embeddings == 384
    assert manager.embedding_dim == 256


def test_embedding_file_loading_duplicates():
    """Test EmbeddingManager raises on duplicate embedding keys"""
    with pytest.raises(Exception, match="Duplicate embedding names"):
        EmbeddingManager(embedding_file_path=[embeddings_file_pth_path, embeddings_file_pth_path])


def test_embedding_file_loading_multiple():
    """Test EmbeddingManager for loading multiple embedding files"""
    manager = EmbeddingManager(embedding_file_path=[embeddings_file_pth_path, embeddings_file_path2])
    assert manager.embedding_dim == 256
    assert manager.num_embeddings == 384 * 2
