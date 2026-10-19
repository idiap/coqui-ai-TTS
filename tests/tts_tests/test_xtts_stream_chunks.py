"""Check streaming finalization with synthetic waveforms, without model weights."""

from types import MethodType, SimpleNamespace

import pytest
import torch

from TTS.tts.models.xtts import Xtts


@pytest.mark.parametrize("lengths", [[40000, 50000], [40000, 40000], [40000, 40500], [500], [50000]])
def test_final_chunk_preserves_all_samples(lengths):
    previous = overlap = None
    chunks = []
    for index, length in enumerate(lengths):
        chunk, previous, overlap = Xtts.handle_chunks(
            None,
            torch.arange(length, dtype=torch.float32),
            previous,
            overlap,
            1024,
            is_final=index == len(lengths) - 1,
        )
        chunks.append(chunk)
    torch.testing.assert_close(torch.cat(chunks), torch.arange(lengths[-1], dtype=torch.float32))
    assert overlap is None


def test_final_chunk_crossfades_with_previous_overlap():
    _, previous, overlap = Xtts.handle_chunks(None, torch.zeros(100), None, None, 10)
    chunk, _, overlap = Xtts.handle_chunks(None, torch.ones(125), previous, overlap, 10, is_final=True)
    torch.testing.assert_close(chunk[:10], torch.linspace(0, 1, 10))
    torch.testing.assert_close(chunk[10:], torch.ones(25))
    assert overlap is None


@pytest.mark.parametrize("token_count", [1, 2, 19, 20, 21, 22, 23, 40, 45])
def test_inference_stream_flushes_final_audio(token_count):
    gpt = SimpleNamespace(
        compute_embeddings=lambda *args: torch.zeros(1, 1),
        get_generator=lambda **kwargs: iter((torch.tensor([i]), torch.zeros(1, 4)) for i in range(token_count)),
    )
    model = SimpleNamespace(
        device=torch.device("cpu"),
        gpt=gpt,
        args=SimpleNamespace(gpt_max_text_tokens=100),
        tokenizer=SimpleNamespace(encode=lambda text, lang: [1, 2]),
        hifigan_decoder=lambda latents, **kwargs: torch.arange(latents.shape[1] * 16, dtype=torch.float32)[
            None, None, :
        ],
    )
    model.handle_chunks = MethodType(Xtts.handle_chunks, model)
    chunks = list(
        Xtts.inference_stream(
            model,
            "Hello",
            "en",
            torch.zeros(1),
            torch.zeros(1),
            stream_chunk_size=20,
            overlap_wav_len=32,
        )
    )
    torch.testing.assert_close(torch.cat(chunks), torch.arange(token_count * 16, dtype=torch.float32))
