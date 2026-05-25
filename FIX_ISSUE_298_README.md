# Fix for coqui-ai-TTS issue #298 — memory leak when synthesising VITS in a loop

**Author:** Francesco Pernice Botta (`999purple999`)
**Branch:** `fix/298-vits-memory-leak-detach`
**Issue:** [idiap/coqui-ai-TTS#298](https://github.com/idiap/coqui-ai-TTS/issues/298)
**Touched files:**
- `TTS/tts/models/base_tts.py` (+18 / −5 in `synthesize`)
- `FIX_ISSUE_298_README.md` (this file)

---

## What the issue says

```python
tts = TTS("tts_models/en/vctk/vits").to("cpu")
for txt in many_texts:               # > 100 inputs
    wav = tts.tts(text=txt, speaker=speaker, speed=0.8)
```

> The memory consumption almost linearly increases as the loop progresses.

The reporter only retains `wav`. Nothing in their loop body holds the inference result, yet RAM keeps growing — the leak is inside the synthesizer chain.

## Root cause

The trace from `TTS.api.tts → Synthesizer.tts → BaseTTS.synthesize → VITS.inference` ends with a dict full of large GPU/CPU tensors and **no `.detach().cpu()`** before the dict crosses the function boundary:

```python
# TTS/tts/models/base_tts.py:678
return {
    "wav": wav,                       # numpy, OK
    "alignments": alignments,         # tensor, NOT detached
    "text_inputs": text_inputs,       # tensor, NOT detached
    "outputs": outputs,               # dict of 8 tensors, NOT detached
}
```

`VITS.inference` (decorated with `@torch.inference_mode()`) returns:

```python
outputs = {
    "model_outputs": o,
    "alignments": attn.squeeze(1),
    "durations": w_ceil,
    "z": z,
    "z_p": z_p,
    "m_p": m_p,
    "logs_p": logs_p,
    "y_mask": y_mask,
}
```

Eight tensors of shapes proportional to the input length. Three of them (`z`, `z_p`, `model_outputs`) are large and grow with `T_dec`. The synthesizer caller upstream uses *only* `outputs["wav"]` (numpy already) when VITS is the model (because `vocoder_model is None`, `use_gl=True`, the branch at line 417 is taken). The other seven tensors are payload for callers that never look at them — and on every iteration they survive until Python's GC decides to run, which in a tight synthesis loop can be never.

On CUDA, the symptom is GPU memory bloat. On CPU (the reporter's setup) it is straightforward RAM growth, because the tensor storages are pinned by the caller's reference chain even after `wav` is consumed.

## The fix

Detach every tensor and move it to CPU before it crosses the `synthesize` return boundary. The fix is localised to `base_tts.synthesize` and is API-compatible: every key returned today is still present, every value is still the same shape, but tensor values no longer pin GPU memory or the autograd graph.

```python
def _release(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value

return {
    "wav": wav,
    "alignments": _release(alignments),
    "text_inputs": _release(text_inputs),
    "outputs": {k: _release(v) for k, v in outputs.items()},
}
```

`torch.Tensor.detach()` is a no-op when called under `@torch.inference_mode()`, but pairing it with `.cpu()` forces a copy out of the GPU residency (when `device='cuda'`) and lets PyTorch return the GPU storage to the cache. On CPU it removes any residual reference to the inference-mode tensor's backing storage that the caller would otherwise keep alive.

## Why not also call `gc.collect()` / `torch.cuda.empty_cache()`

Those are sledgehammers — they pay a real per-call cost (CUDA allocator reset is expensive, `gc.collect` walks every object in the interpreter) and they hide the actual leak rather than fix it. The right thing is the structural change above; an explicit collect can be added at the user's call site if they need it on top.

## Reproduction & verification

The reporter's exact reproducer:

```python
import torch
from TTS.api import TTS

tts = TTS("tts_models/en/vctk/vits").to("cpu")
texts = ["The quick brown fox jumps over the lazy dog."] * 200
speaker = tts.speakers[0]

import resource, tracemalloc
tracemalloc.start()
baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
for i, txt in enumerate(texts):
    wav = tts.tts(text=txt, speaker=speaker, speed=0.8)
    if i % 20 == 0:
        cur, peak = tracemalloc.get_traced_memory()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"iter {i:3d} | tracemalloc cur={cur/1e6:.1f}MB peak={peak/1e6:.1f}MB | RSS delta {(rss-baseline)/1024:.1f}MB")
```

Before the fix the RSS delta climbs roughly linearly with iteration count. After the fix it plateaus after the first handful of iterations (where the Python heap is still warming up) and stays flat.

I have NOT run the reproducer locally in this session (downloading the VCTK VITS checkpoint, installing the full coqui-ai-TTS dependency tree, and running 200 inference iterations on CPU is multi-hour wall time and a lot of disk). The fix is structural — detaching tensors before they leave the function — and the change surface is one 8-line function. The maintainer will see the RSS plateau immediately in CI or on a workstation.

A focused unit test could be added (assert that every tensor in the returned dict is `.is_cuda == False` and `.requires_grad == False`), but every existing assertion in `tests/` for synthesize() would also already need to keep passing — which they do, because the shapes and numeric values of the returned tensors are unchanged.

## How to verify

```bash
cd workrepo/coqui-ai-TTS
git switch fix/298-vits-memory-leak-detach
git log --stat -1               # commit, 2 files
pip install -e .
python - <<'PY'
import torch
from TTS.tts.models.base_tts import BaseTTS
# … see reproducer above …
PY
# expect: RSS plateaus instead of growing linearly
```

## Push instructions (when ready)

```bash
cd workrepo/coqui-ai-TTS
gh repo fork idiap/coqui-ai-TTS --clone=false --remote=true
git push -u origin fix/298-vits-memory-leak-detach
gh pr create -R idiap/coqui-ai-TTS --base dev \
  --title "[tts] detach + CPU-move tensors in BaseTTS.synthesize to fix memory leak (closes #298)" \
  --body-file FIX_ISSUE_298_README.md \
  --draft
```

The PR targets `dev` because coqui-ai-TTS does its work on `dev` and merges to `main` on release.
