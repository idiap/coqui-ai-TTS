import logging
import re

import numpy as np
import torch
import tqdm
from scipy.special import softmax
from torch.nn import functional as F

from TTS.tts.layers.bark.load_model import clear_cuda_cache

logger = logging.getLogger(__name__)


def _tokenize(tokenizer: "BertTokenizer", text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _detokenize(tokenizer: "BertTokenizer", enc_text: list[int]) -> str:
    return tokenizer.decode(enc_text)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@torch.inference_mode()
def generate_text_semantic(
    text: str,
    model: "Bark",
    history_prompt: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
    temp: float = 0.7,
    top_k: int | None = None,
    top_p: float | None = None,
    silent: bool = False,
    min_eos_p: float = 0.2,
    max_gen_duration_s: float | None = None,
    allow_early_stop: bool = True,
    base: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    use_kv_caching: bool = True,
    **kwargs,  # pylint: disable=unused-argument
) -> torch.Tensor:
    """Generate semantic tokens from text.

    Args:
        text: The text to generate semantic tokens from.
        model: The BarkModel to use for generating the semantic tokens.
        history_prompt: A tuple of (semantic_history, coarse_history, fine_history) to use as a prompt for the generation.
        temp: The temperature to use for the generation.
        top_k: The number of top tokens to consider for the generation.
        top_p: The cumulative probability to consider for the generation.
        silent: Whether to silence the tqdm progress bar.
        min_eos_p: The minimum probability to consider for the end of sentence token.
        max_gen_duration_s: The maximum duration in seconds to generate for.
        allow_early_stop: Whether to allow the generation to stop early.
        base: A tuple of (semantic_history, coarse_history, fine_history) to use as a base for the generation.
        use_kv_caching: Whether to use key-value caching for the generation.
        **kwargs: Additional keyword arguments. They are ignored.

    Returns:
        The generated semantic tokens.
    """
    assert isinstance(text, str)
    text = _normalize_whitespace(text)
    assert len(text.strip()) > 0
    if all(v is not None for v in history_prompt) or base is not None:
        semantic_history = history_prompt[0]
        if base is not None:
            semantic_history = base[0]
        assert (
            isinstance(semantic_history, torch.Tensor)
            and len(semantic_history.shape) == 1
            and len(semantic_history) > 0
            and semantic_history.min() >= 0
            and semantic_history.max() <= model.config.SEMANTIC_VOCAB_SIZE - 1
        )
    else:
        semantic_history = None
    encoded_text = (
        torch.tensor(_tokenize(model.tokenizer, text), device=model.device, dtype=torch.long)
        + model.config.TEXT_ENCODING_OFFSET
    )
    if len(encoded_text) > 256:
        p = (len(encoded_text) - 256) / len(encoded_text) * 100
        logger.warning("warning, text too long, lopping of last %.1f%%", p)
        encoded_text = encoded_text[:256]
    encoded_text = F.pad(
        encoded_text,
        (0, 256 - len(encoded_text)),
        mode="constant",
        value=model.config.TEXT_PAD_TOKEN,
    )
    if semantic_history is not None:
        semantic_history = semantic_history.to(dtype=torch.int64)
        # lop off if history is too long, pad if needed
        semantic_history = semantic_history[-256:]
        semantic_history = F.pad(
            semantic_history,
            (0, 256 - len(semantic_history)),
            mode="constant",
            value=model.config.SEMANTIC_PAD_TOKEN,
        )
    else:
        semantic_history = torch.full((256,), model.config.SEMANTIC_PAD_TOKEN, dtype=torch.int64)
    x = torch.cat(
        [encoded_text, semantic_history, torch.tensor([model.config.SEMANTIC_INFER_TOKEN], device=model.device)]
    ).unsqueeze(0)
    assert x.shape[1] == 256 + 256 + 1

    n_tot_steps = 768
    # custom tqdm updates since we don't know when eos will occur
    pbar = tqdm.tqdm(disable=silent, total=100)
    pbar_state = 0
    tot_generated_duration_s = 0
    kv_cache = None
    for n in range(n_tot_steps):
        if use_kv_caching and kv_cache is not None:
            x_input = x[:, [-1]]
        else:
            x_input = x
        logits, kv_cache = model.semantic_model(x_input, merge_context=True, use_cache=use_kv_caching, past_kv=kv_cache)
        relevant_logits = logits[0, 0, : model.config.SEMANTIC_VOCAB_SIZE]
        if allow_early_stop:
            relevant_logits = torch.hstack((relevant_logits, logits[0, 0, [model.config.SEMANTIC_PAD_TOKEN]]))  # eos
        if top_p is not None:
            # faster to convert to numpy
            logits_device = relevant_logits.device
            logits_dtype = relevant_logits.type()
            relevant_logits = relevant_logits.detach().cpu().type(torch.float32).numpy()
            sorted_indices = np.argsort(relevant_logits)[::-1]
            sorted_logits = relevant_logits[sorted_indices]
            cumulative_probs = np.cumsum(softmax(sorted_logits))
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
            sorted_indices_to_remove[0] = False
            relevant_logits[sorted_indices[sorted_indices_to_remove]] = -np.inf
            relevant_logits = torch.from_numpy(relevant_logits)
            relevant_logits = relevant_logits.to(logits_device).type(logits_dtype)
        if top_k is not None:
            v, _ = torch.topk(relevant_logits, min(top_k, relevant_logits.size(-1)))
            relevant_logits[relevant_logits < v[-1]] = -float("Inf")
        probs = torch.softmax(relevant_logits / temp, dim=-1)
        item_next = torch.multinomial(probs, num_samples=1)
        if allow_early_stop and (
            item_next == model.config.SEMANTIC_VOCAB_SIZE or (min_eos_p is not None and probs[-1] >= min_eos_p)
        ):
            # eos found, so break
            pbar.update(100 - pbar_state)
            break
        x = torch.cat((x, item_next.unsqueeze(0)), dim=1)
        tot_generated_duration_s += 1 / model.config.SEMANTIC_RATE_HZ
        if max_gen_duration_s is not None and tot_generated_duration_s > max_gen_duration_s:
            pbar.update(100 - pbar_state)
            break
        if n == n_tot_steps - 1:
            pbar.update(100 - pbar_state)
            break
        del logits, relevant_logits, probs, item_next
        req_pbar_state = np.min([100, int(round(100 * n / n_tot_steps))])
        if req_pbar_state > pbar_state:
            pbar.update(req_pbar_state - pbar_state)
        pbar_state = req_pbar_state
    pbar.close()
    out = x.squeeze()[256 + 256 + 1 :]
    assert all(out >= 0) and all(out < model.config.SEMANTIC_VOCAB_SIZE)
    clear_cuda_cache()
    return out


@torch.inference_mode()
def _flatten_codebooks(arr: torch.Tensor, offset_size: int | None) -> torch.Tensor:
    assert len(arr.shape) == 2
    arr = arr.clone()
    if offset_size is not None:
        for n in range(1, arr.shape[0]):
            arr[n, :] += offset_size * n
    return arr.T.flatten()


@torch.inference_mode()
def generate_coarse(
    x_semantic: torch.Tensor,
    model: "Bark",
    history_prompt: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
    temp: float = 0.7,
    top_k: int | None = None,
    top_p: float | None = None,
    silent: bool = False,
    max_coarse_history: int = 630,  # min 60 (faster), max 630 (more context)
    sliding_window_len: int = 60,
    base: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    use_kv_caching: bool = True,
) -> torch.Tensor:
    """Generate coarse audio codes from semantic tokens.

    Args:
        x_semantic: The semantic tokens to generate coarse audio codes from.
        model: The BarkModel to use for generating the coarse audio codes.
        history_prompt: A tuple of (semantic_history, coarse_history, fine_history) to use as a prompt for the generation.
        temp: The temperature to use for the generation.
        top_k: The number of top tokens to consider for the generation.
        top_p: The cumulative probability to consider for the generation.
        silent: Whether to silence the tqdm progress bar.
        max_coarse_history: The maximum number of coarse audio codes to use as history.
        sliding_window_len: The length of the sliding window to use for the generation.
        base: A tuple of (semantic_history, coarse_history, fine_history) to use as a base for the generation.
        use_kv_caching: Whether to use key-value caching for the generation.

    Returns:
        The generated coarse audio codes.
    """
    assert (
        isinstance(x_semantic, torch.Tensor)
        and len(x_semantic.shape) == 1
        and len(x_semantic) > 0
        and x_semantic.min() >= 0
        and x_semantic.max() <= model.config.SEMANTIC_VOCAB_SIZE - 1
    )
    assert 60 <= max_coarse_history <= 630
    assert max_coarse_history + sliding_window_len <= 1024 - 256
    semantic_to_coarse_ratio = (
        model.config.COARSE_RATE_HZ / model.config.SEMANTIC_RATE_HZ * model.config.N_COARSE_CODEBOOKS
    )
    max_semantic_history = int(np.floor(max_coarse_history / semantic_to_coarse_ratio))
    if all(v is not None for v in history_prompt) or base is not None:
        x_history = history_prompt
        x_semantic_history = x_history[0]
        x_coarse_history = x_history[1]
        if base is not None:
            x_semantic_history = base[0]
            x_coarse_history = base[1]
        assert (
            isinstance(x_semantic_history, torch.Tensor)
            and len(x_semantic_history.shape) == 1
            and len(x_semantic_history) > 0
            and x_semantic_history.min() >= 0
            and x_semantic_history.max() <= model.config.SEMANTIC_VOCAB_SIZE - 1
            and isinstance(x_coarse_history, torch.Tensor)
            and len(x_coarse_history.shape) == 2
            and x_coarse_history.shape[0] == model.config.N_COARSE_CODEBOOKS
            and x_coarse_history.shape[-1] >= 0
            and x_coarse_history.min() >= 0
            and x_coarse_history.max() <= model.config.CODEBOOK_SIZE - 1
            and (
                round(x_coarse_history.shape[-1] / len(x_semantic_history), 1)
                == round(semantic_to_coarse_ratio / model.config.N_COARSE_CODEBOOKS, 1)
            )
        )
        x_coarse_history = (
            _flatten_codebooks(x_coarse_history, model.config.CODEBOOK_SIZE) + model.config.SEMANTIC_VOCAB_SIZE
        )
        # trim histories correctly
        n_semantic_hist_provided = np.min(
            [
                max_semantic_history,
                len(x_semantic_history) - len(x_semantic_history) % 2,
                int(np.floor(len(x_coarse_history) / semantic_to_coarse_ratio)),
            ]
        )
        n_coarse_hist_provided = int(round(n_semantic_hist_provided * semantic_to_coarse_ratio))
        x_semantic_history = x_semantic_history[-n_semantic_hist_provided:]
        x_coarse_history = x_coarse_history[-n_coarse_hist_provided:]
        # TODO: bit of a hack for time alignment (sounds better)
        x_coarse_history = x_coarse_history[:-2]
    else:
        x_semantic_history = torch.tensor([], dtype=torch.int32)
        x_coarse_history = torch.tensor([], dtype=torch.int32)
    # start loop
    n_steps = int(
        round(
            np.floor(len(x_semantic) * semantic_to_coarse_ratio / model.config.N_COARSE_CODEBOOKS)
            * model.config.N_COARSE_CODEBOOKS
        )
    )
    assert n_steps > 0 and n_steps % model.config.N_COARSE_CODEBOOKS == 0
    x_semantic = torch.cat([x_semantic_history.to(model.device), x_semantic]).to(dtype=torch.int32)
    base_semantic_idx = len(x_semantic_history)

    x_semantic_in = x_semantic.unsqueeze(0)
    x_coarse_in = x_coarse_history.unsqueeze(0).to(model.device)
    n_window_steps = int(np.ceil(n_steps / sliding_window_len))
    n_step = 0
    for _ in tqdm.tqdm(range(n_window_steps), total=n_window_steps, disable=silent):
        semantic_idx = base_semantic_idx + int(round(n_step / semantic_to_coarse_ratio))
        # pad from right side
        x_in = x_semantic_in[:, np.max([0, semantic_idx - max_semantic_history]) :]
        x_in = x_in[:, :256]
        x_in = F.pad(
            x_in,
            (0, 256 - x_in.shape[-1]),
            "constant",
            model.config.COARSE_SEMANTIC_PAD_TOKEN,
        )
        x_in = torch.hstack(
            [
                x_in,
                torch.tensor([model.config.COARSE_INFER_TOKEN]).unsqueeze(0).to(model.device),
                x_coarse_in[:, -max_coarse_history:],
            ]
        )
        kv_cache = None
        for _ in range(sliding_window_len):
            if n_step >= n_steps:
                continue
            is_major_step = n_step % model.config.N_COARSE_CODEBOOKS == 0

            if use_kv_caching and kv_cache is not None:
                x_input = x_in[:, [-1]]
            else:
                x_input = x_in

            logits, kv_cache = model.coarse_model(x_input, use_cache=use_kv_caching, past_kv=kv_cache)
            logit_start_idx = model.config.SEMANTIC_VOCAB_SIZE + (1 - int(is_major_step)) * model.config.CODEBOOK_SIZE
            logit_end_idx = model.config.SEMANTIC_VOCAB_SIZE + (2 - int(is_major_step)) * model.config.CODEBOOK_SIZE
            relevant_logits = logits[0, 0, logit_start_idx:logit_end_idx]
            if top_p is not None:
                # faster to convert to numpy
                logits_device = relevant_logits.device
                logits_dtype = relevant_logits.type()
                relevant_logits = relevant_logits.detach().cpu().type(torch.float32).numpy()
                sorted_indices = np.argsort(relevant_logits)[::-1]
                sorted_logits = relevant_logits[sorted_indices]
                cumulative_probs = np.cumsum(torch.nn.functional.softmax(sorted_logits))
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                sorted_indices_to_remove[0] = False
                relevant_logits[sorted_indices[sorted_indices_to_remove]] = -np.inf
                relevant_logits = torch.from_numpy(relevant_logits)
                relevant_logits = relevant_logits.to(logits_device).type(logits_dtype)
            if top_k is not None:
                v, _ = torch.topk(relevant_logits, min(top_k, relevant_logits.size(-1)))
                relevant_logits[relevant_logits < v[-1]] = -float("Inf")
            probs = torch.nn.functional.softmax(relevant_logits / temp, dim=-1)
            item_next = torch.multinomial(probs, num_samples=1)
            item_next += logit_start_idx
            x_coarse_in = torch.cat((x_coarse_in, item_next.unsqueeze(0)), dim=1)
            x_in = torch.cat((x_in, item_next.unsqueeze(0)), dim=1)
            del logits, relevant_logits, probs, item_next
            n_step += 1
        del x_in
    del x_semantic_in
    gen_coarse_arr = x_coarse_in.squeeze()[len(x_coarse_history) :]
    del x_coarse_in
    assert len(gen_coarse_arr) == n_steps
    gen_coarse_audio_arr = (
        gen_coarse_arr.reshape(-1, model.config.N_COARSE_CODEBOOKS).T - model.config.SEMANTIC_VOCAB_SIZE
    )
    for n in range(1, model.config.N_COARSE_CODEBOOKS):
        gen_coarse_audio_arr[n, :] -= n * model.config.CODEBOOK_SIZE
    clear_cuda_cache()
    return gen_coarse_audio_arr


@torch.inference_mode()
def generate_fine(
    x_coarse_gen: torch.Tensor,
    model: "Bark",
    history_prompt: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
    temp: float = 0.5,
    silent: bool = True,
    base: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Generate full audio codes from coarse audio codes.

    Args:
        x_coarse_gen: The coarse audio codes to generate full audio codes from.
        model: The BarkModel to use for generating the full audio codes.
        history_prompt: A tuple of (semantic_history, coarse_history, fine_history) to use as a prompt for the generation.
        temp: The temperature to use for the generation.
        silent: Whether to silence the tqdm progress bar.
        base: A tuple of (semantic_history, coarse_history, fine_history) to use as a base for the generation.

    Returns:
        The generated full audio codes.
    """
    assert (
        isinstance(x_coarse_gen, torch.Tensor)
        and len(x_coarse_gen.shape) == 2
        and 1 <= x_coarse_gen.shape[0] <= model.config.N_FINE_CODEBOOKS - 1
        and x_coarse_gen.shape[1] > 0
        and x_coarse_gen.min() >= 0
        and x_coarse_gen.max() <= model.config.CODEBOOK_SIZE - 1
    )
    if all(v is not None for v in history_prompt) or base is not None:
        x_fine_history = history_prompt[2]
        if base is not None:
            x_fine_history = base[2]
        assert (
            isinstance(x_fine_history, torch.Tensor)
            and len(x_fine_history.shape) == 2
            and x_fine_history.shape[0] == model.config.N_FINE_CODEBOOKS
            and x_fine_history.shape[1] >= 0
            and x_fine_history.min() >= 0
            and x_fine_history.max() <= model.config.CODEBOOK_SIZE - 1
        )
    else:
        x_fine_history = None
    n_coarse = x_coarse_gen.shape[0]
    # make input arr
    padding = torch.full(
        (model.config.N_FINE_CODEBOOKS - n_coarse, x_coarse_gen.shape[1]), model.config.CODEBOOK_SIZE, dtype=torch.int32
    ).to(model.device)
    in_arr = torch.cat([x_coarse_gen, padding], dim=0)
    # prepend history if available (max 512)
    if x_fine_history is not None:
        in_arr = torch.cat([x_fine_history[:, -512:].to(model.device), in_arr], dim=1)
        n_history = x_fine_history[:, -512:].shape[1]
    else:
        n_history = 0
    n_remove_from_end = 0
    # need to pad if too short (since non-causal model)
    if in_arr.shape[1] < 1024:
        n_remove_from_end = 1024 - in_arr.shape[1]
        padding = torch.full(
            (model.config.N_FINE_CODEBOOKS, n_remove_from_end), model.config.CODEBOOK_SIZE, dtype=torch.int32
        ).to(model.device)
        in_arr = torch.cat([in_arr, padding], dim=1)
    # we can be lazy about fractional loop and just keep overwriting codebooks
    n_loops = np.max([0, int(np.ceil((x_coarse_gen.shape[1] - (1024 - n_history)) / 512))]) + 1

    in_arr = in_arr.T.to(model.device)
    for n in tqdm.tqdm(range(n_loops), disable=silent):
        start_idx = np.min([n * 512, in_arr.shape[0] - 1024])
        start_fill_idx = np.min([n_history + n * 512, in_arr.shape[0] - 512])
        rel_start_fill_idx = start_fill_idx - start_idx
        in_buffer = in_arr[start_idx : start_idx + 1024, :].unsqueeze(0)
        for nn in range(n_coarse, model.config.N_FINE_CODEBOOKS):
            logits = model.fine_model(nn, in_buffer)
            if temp is None:
                relevant_logits = logits[0, rel_start_fill_idx:, : model.config.CODEBOOK_SIZE]
                codebook_preds = torch.argmax(relevant_logits, -1)
            else:
                relevant_logits = logits[0, :, : model.config.CODEBOOK_SIZE] / temp
                probs = F.softmax(relevant_logits, dim=-1)
                codebook_preds = torch.hstack(
                    [torch.multinomial(probs[n], num_samples=1) for n in range(rel_start_fill_idx, 1024)]
                )
            in_buffer[0, rel_start_fill_idx:, nn] = codebook_preds
            del logits, codebook_preds
        # transfer over info into model_in and convert to numpy
        for nn in range(n_coarse, model.config.N_FINE_CODEBOOKS):
            in_arr[start_fill_idx : start_fill_idx + (1024 - rel_start_fill_idx), nn] = in_buffer[
                0, rel_start_fill_idx:, nn
            ]
        del in_buffer
    gen_fine_arr = in_arr.squeeze().T
    del in_arr
    gen_fine_arr = gen_fine_arr[:, n_history:]
    if n_remove_from_end > 0:
        gen_fine_arr = gen_fine_arr[:, :-n_remove_from_end]
    assert gen_fine_arr.shape[-1] == x_coarse_gen.shape[-1]
    clear_cuda_cache()
    return gen_fine_arr


@torch.inference_mode()
def codec_decode(fine_tokens: torch.Tensor, model) -> torch.Tensor:
    """Turn quantized audio codes into audio array using encodec."""
    arr = fine_tokens.unsqueeze(0)
    arr = arr.transpose(0, 1)
    emb = model.encodec.quantizer.decode(arr)
    out = model.encodec.decoder(emb)
    return out.squeeze().cpu()
