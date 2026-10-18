"""Run cached autoregressive generation for Tortoise and XTTS with tiny random models.

These rely on private transformers generation internals and catch breaking changes in new releases, e.g.:
- transformers>=5.3 only slices the inputs to the uncached tokens if `next_sequence_length` is passed
- transformers>=5.18 drops an all-ones attention_mask in generate()
"""

import functools

import pytest
import torch

from TTS.tts.layers.tortoise.autoregressive import UnifiedVoice
from TTS.tts.layers.xtts.gpt import GPT
from TTS.tts.layers.xtts.stream_generator import init_stream_support

MAX_NEW_TOKENS = 8
MODEL_DIM = 64
SAMPLING = {"do_sample": True, "top_p": 0.8, "temperature": 0.8, "repetition_penalty": 2.0}


def record_steps(monkeypatch, inference_model) -> list[tuple[int, int]]:
    """Record (input length, cache length) for every forward step of the inference model."""
    steps = []
    forward = inference_model.forward

    # wraps() keeps the signature, which transformers inspects to decide which inputs to pass
    @functools.wraps(forward)
    def recording_forward(*args, **kwargs):
        past_key_values = kwargs.get("past_key_values")
        past_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        steps.append((kwargs["input_ids"].shape[1], past_length))
        return forward(*args, **kwargs)

    monkeypatch.setattr(inference_model, "forward", recording_forward)
    return steps


def assert_one_new_token_per_step(steps: list[tuple[int, int]]) -> None:
    prompt_length = steps[0][0]
    assert steps == [(prompt_length, 0)] + [(1, prompt_length + i) for i in range(len(steps) - 1)]


@pytest.fixture
def xtts_gpt():
    torch.manual_seed(0)
    model = GPT(
        layers=2,
        model_dim=MODEL_DIM,
        heads=2,
        max_text_tokens=40,
        max_mel_tokens=MAX_NEW_TOKENS + 3,  # generates max_mel_tokens - max_conditioning_inputs - 2 tokens
        max_prompt_tokens=10,
        number_text_tokens=300,
    )
    model.init_gpt_for_inference(kv_cache=True)
    model.eval()
    with torch.no_grad():
        # Never stop early, so that the number of generated tokens is deterministic
        model.mel_head.bias[model.stop_audio_token] = -1e4
    return model


@pytest.mark.parametrize("num_return_sequences", [1, 2])
def test_tortoise_inference_speech_with_kv_cache(monkeypatch, num_return_sequences):
    torch.manual_seed(0)
    model = UnifiedVoice(
        layers=2,
        model_dim=MODEL_DIM,
        heads=2,
        max_text_tokens=40,
        max_mel_tokens=60,
        max_conditioning_inputs=1,
        number_text_tokens=255,
        start_text_token=255,
        checkpointing=False,
    )
    model.post_init_gpt2_config(kv_cache=True)
    model.eval()
    with torch.no_grad():
        model.mel_head.bias[model.stop_mel_token] = -1e4
    steps = record_steps(monkeypatch, model.inference_model)

    with torch.inference_mode():
        codes = model.inference_speech(
            torch.randn(1, MODEL_DIM),
            torch.randint(0, 255, (1, 12)),
            num_return_sequences=num_return_sequences,
            max_generate_length=MAX_NEW_TOKENS,
            **SAMPLING,
        )

    assert codes.shape == (num_return_sequences, MAX_NEW_TOKENS)
    assert len(steps) == MAX_NEW_TOKENS
    assert_one_new_token_per_step(steps)


def test_xtts_gpt_generate_with_kv_cache(monkeypatch, xtts_gpt):
    steps = record_steps(monkeypatch, xtts_gpt.gpt_inference)
    with torch.inference_mode():
        codes = xtts_gpt.generate(torch.randn(1, 3, MODEL_DIM), torch.randint(0, 255, (1, 12)), **SAMPLING)

    assert codes.shape == (1, MAX_NEW_TOKENS)
    assert len(steps) == MAX_NEW_TOKENS
    assert_one_new_token_per_step(steps)


def test_xtts_gpt_stream_matches_generate(monkeypatch, xtts_gpt):
    init_stream_support()
    cond_latents = torch.randn(1, 3, MODEL_DIM)
    text_tokens = torch.randint(0, 255, (1, 12))

    torch.manual_seed(1)
    with torch.inference_mode():
        codes = xtts_gpt.generate(cond_latents, text_tokens, **SAMPLING)

    steps = record_steps(monkeypatch, xtts_gpt.gpt_inference)
    torch.manual_seed(1)
    with torch.inference_mode():
        fake_inputs = xtts_gpt.compute_embeddings(cond_latents, text_tokens)
        generator = xtts_gpt.get_generator(
            fake_inputs=fake_inputs,
            num_beams=1,
            num_return_sequences=1,
            output_attentions=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
            **SAMPLING,
        )
        tokens, latents = zip(*generator, strict=True)

    assert torch.cat(tokens).tolist() == codes[0].tolist()
    assert all(latent.shape == (1, MODEL_DIM) for latent in latents)
    assert len(steps) == MAX_NEW_TOKENS
    assert_one_new_token_per_step(steps)
