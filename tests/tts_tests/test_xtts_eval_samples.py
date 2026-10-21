import logging
from unittest.mock import MagicMock

import pytest
import torch

from TTS.tts.layers.xtts.trainer.dataset import XTTSDataset


def test_xtts_dataset_eval_with_none_samples():
    """Verify XTTSDataset handles samples=None gracefully during evaluation."""
    config = MagicMock()
    config.model_args.min_conditioning_length = 66150
    config.model_args.max_wav_length = 255995
    config.model_args.max_text_length = 200
    config.model_args.gpt_use_masking_gt_prompt_approach = True
    tokenizer = MagicMock()

    # Should not raise TypeError: 'NoneType' object is not iterable
    dataset = XTTSDataset(
        config=config,
        samples=None,
        tokenizer=tokenizer,
        sample_rate=22050,
        is_eval=True,
    )
    assert dataset.samples == []
    assert len(dataset) == 0


def test_xtts_dataset_train_with_none_samples_raises():
    """Verify XTTSDataset raises clear ValueError when training samples are empty or None."""
    config = MagicMock()
    config.model_args.min_conditioning_length = 66150
    config.model_args.max_wav_length = 255995
    config.model_args.max_text_length = 200
    config.model_args.gpt_use_masking_gt_prompt_approach = True
    tokenizer = MagicMock()

    with pytest.raises(ValueError, match="Training dataset cannot be empty"):
        XTTSDataset(
            config=config,
            samples=None,
            tokenizer=tokenizer,
            sample_rate=22050,
            is_eval=False,
        )


def test_xtts_dataset_check_eval_samples_filtering_logs(caplog):
    """Verify XTTSDataset.check_eval_samples logs informative warnings when samples are filtered out."""
    config = MagicMock()
    config.model_args.min_conditioning_length = 100
    config.model_args.max_wav_length = 1000
    config.model_args.max_text_length = 50
    config.model_args.gpt_use_masking_gt_prompt_approach = True
    tokenizer = MagicMock()

    dataset = XTTSDataset.__new__(XTTSDataset)
    dataset.config = config
    dataset.max_wav_len = 1000
    dataset.max_text_len = 50
    dataset.samples = [
        {"audio_file": "missing.wav", "text": "sample 1", "language": "en"},
        {"audio_file": "too_long_audio.wav", "text": "sample 2", "language": "en"},
        {"audio_file": "too_long_text.wav", "text": "sample 3", "language": "en"},
        {"audio_file": "valid.wav", "text": "sample 4", "language": "en"},
    ]

    def mock_load_item(sample):
        name = sample["audio_file"]
        if name == "missing.wav":
            return torch.zeros(10), None, None, None, None, None
        if name == "too_long_audio.wav":
            return torch.zeros(10), None, torch.zeros(2000), None, None, None
        if name == "too_long_text.wav":
            return torch.zeros(100), None, torch.zeros(500), None, None, None
        return torch.zeros(10), None, torch.zeros(500), None, None, None

    dataset.load_item = mock_load_item

    with caplog.at_level(logging.WARNING):
        dataset.check_eval_samples()

    assert len(dataset.samples) == 1
    assert dataset.samples[0]["audio_file"] == "valid.wav"
    assert "audio waveform is None" in caplog.text
    assert "exceeds max_wav_len" in caplog.text
    assert "exceeds max_text_len" in caplog.text


def test_xtts_dataset_check_eval_samples_all_filtered_logs_warning(caplog):
    """Verify XTTSDataset logs a warning when all evaluation samples are filtered out."""
    dataset = XTTSDataset.__new__(XTTSDataset)
    dataset.max_wav_len = 100
    dataset.max_text_len = 50
    dataset.samples = [{"audio_file": "empty.wav", "text": "sample", "language": "en"}]
    dataset.load_item = lambda s: (torch.zeros(10), None, None, None, None, None)

    with caplog.at_level(logging.WARNING):
        dataset.check_eval_samples()

    assert len(dataset.samples) == 0
    assert "All evaluation samples were filtered out" in caplog.text


def test_xtts_dataset_check_eval_samples_exception_handling(caplog):
    """Verify XTTSDataset catches and logs exception if a sample raises on load_item."""
    dataset = XTTSDataset.__new__(XTTSDataset)
    dataset.max_wav_len = 1000
    dataset.max_text_len = 50
    dataset.samples = [
        {"audio_file": "corrupt.wav", "text": "bad", "language": "en"},
        {"audio_file": "good.wav", "text": "good", "language": "en"},
    ]

    def mock_load_item(sample):
        if sample["audio_file"] == "corrupt.wav":
            raise RuntimeError("Corrupted audio header")
        return torch.zeros(10), None, torch.zeros(500), None, None, None

    dataset.load_item = mock_load_item

    with caplog.at_level(logging.ERROR):
        dataset.check_eval_samples()

    assert len(dataset.samples) == 1
    assert dataset.samples[0]["audio_file"] == "good.wav"
    assert "Failed to load eval sample corrupt.wav" in caplog.text
