from unittest.mock import MagicMock

import pytest

from TTS.tts.layers.xtts.trainer.dataset import XTTSDataset
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTTrainer


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


@pytest.mark.parametrize("empty_samples", [None, []])
def test_gpt_trainer_get_data_loader_eval_empty_samples_raises_value_error(empty_samples):
    """Verify get_data_loader(is_eval=True) raises informative ValueError when evaluation samples are missing."""
    trainer = GPTTrainer.__new__(GPTTrainer)
    trainer.config = MagicMock()

    with pytest.raises(ValueError, match="No evaluation samples provided to `get_data_loader\\(is_eval=True\\)`"):
        trainer.get_data_loader(
            config=trainer.config,
            is_eval=True,
            samples=empty_samples,
        )
