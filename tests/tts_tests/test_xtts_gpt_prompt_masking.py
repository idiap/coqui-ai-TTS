from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from TTS.tts.layers.xtts.gpt import GPT


def _build_tiny_gpt(use_perceiver: bool = True) -> GPT:
    torch.manual_seed(0)
    return GPT(
        layers=1,
        model_dim=64,
        heads=2,
        max_text_tokens=30,
        max_mel_tokens=100,
        max_prompt_tokens=10,
        number_text_tokens=30,
        start_text_token=28,
        stop_text_token=0,
        num_audio_tokens=1026,
        start_audio_token=1024,
        stop_audio_token=1025,
        use_perceiver_resampler=use_perceiver,
    ).eval()


def _build_synthetic_batch(gpt: GPT, batch_size: int = 3):
    stride = gpt.code_stride_len
    torch.manual_seed(42)
    return {
        "text_inputs": torch.randint(3, 25, (batch_size, 6)),
        "text_lengths": torch.tensor([6, 5, 4][:batch_size]),
        "audio_codes": torch.randint(0, 1024, (batch_size, 40)),
        "wav_lengths": torch.tensor([40, 30, 36][:batch_size]) * stride,
        "cond_mels": torch.randn(batch_size, 1, 80, 100),
        "cond_idxs": torch.tensor(
            [
                [2 * stride, 8 * stride],
                [stride, 6 * stride],
                [3 * stride, 11 * stride],
            ][:batch_size]
        ),
    }


@pytest.mark.parametrize("use_perceiver", [True, False])
def test_gpt_prompt_loss_masking_applies_to_all_batch_samples(use_perceiver: bool):
    """Verify that cond_idxs prompt masking is applied to EVERY sample in a batch."""
    gpt = _build_tiny_gpt(use_perceiver=use_perceiver)
    batch = _build_synthetic_batch(gpt, batch_size=3)
    stride = gpt.code_stride_len

    captured_targets = []
    original_ce = F.cross_entropy

    def spy_cross_entropy(logits, targets, *args, **kwargs):
        captured_targets.append(targets.clone())
        return original_ce(logits, targets, *args, **kwargs)

    with patch("torch.nn.functional.cross_entropy", side_effect=spy_cross_entropy):
        gpt(
            batch["text_inputs"],
            batch["text_lengths"],
            batch["audio_codes"],
            batch["wav_lengths"],
            cond_mels=batch["cond_mels"],
            cond_idxs=batch["cond_idxs"].clone(),
        )

    # Cross entropy is called twice: once for text, once for mel targets
    assert len(captured_targets) == 2
    mel_targets = captured_targets[1]

    for i in range(batch["cond_idxs"].shape[0]):
        lo = (batch["cond_idxs"][i, 0] // stride).item()
        hi = (batch["cond_idxs"][i, 1] // stride).item()

        # Prompt tokens must be masked with -1
        assert (mel_targets[i, lo:hi] == -1).all(), (
            f"Sample {i} did not have prompt codes [{lo}:{hi}] masked in batch size 3"
        )

        # Pre-prompt tokens must remain unmasked
        if lo > 0:
            assert (mel_targets[i, :lo] != -1).all(), f"Sample {i} had unexpected masking before prompt index {lo}"


def test_gpt_prompt_loss_masking_none():
    """Verify that when cond_idxs is None, no prompt tokens are masked."""
    gpt = _build_tiny_gpt()
    batch = _build_synthetic_batch(gpt, batch_size=2)

    captured_targets = []
    original_ce = F.cross_entropy

    def spy_cross_entropy(logits, targets, *args, **kwargs):
        captured_targets.append(targets.clone())
        return original_ce(logits, targets, *args, **kwargs)

    with patch("torch.nn.functional.cross_entropy", side_effect=spy_cross_entropy):
        gpt(
            batch["text_inputs"],
            batch["text_lengths"],
            batch["audio_codes"],
            batch["wav_lengths"],
            cond_mels=batch["cond_mels"],
            cond_idxs=None,
        )

    mel_targets = captured_targets[1]
    # Check that sample 0 retains normal code targets up to its length
    code_len = 40
    assert (mel_targets[0, :code_len] != -1).all()


def test_gpt_forward_backward_with_prompt_masking():
    """Verify forward and backward gradients flow correctly with per-sample prompt masking."""
    gpt = _build_tiny_gpt().train()
    batch = _build_synthetic_batch(gpt, batch_size=2)

    loss_text, loss_mel, _ = gpt(
        batch["text_inputs"],
        batch["text_lengths"],
        batch["audio_codes"],
        batch["wav_lengths"],
        cond_mels=batch["cond_mels"],
        cond_idxs=batch["cond_idxs"].clone(),
    )
    total_loss = loss_text + loss_mel
    assert not torch.isnan(total_loss)
    total_loss.backward()

    # Check that embeddings received gradients
    assert gpt.mel_embedding.weight.grad is not None
    assert not torch.isnan(gpt.mel_embedding.weight.grad).any()
