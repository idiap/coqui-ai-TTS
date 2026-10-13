"""Optional output checksums for zoo tests, to detect changes in model outputs during local testing.

Checksums are only computed if ``TTS_ZOO_CHECKSUMS`` points to a JSON file:

    TTS_ZOO_CHECKSUMS=zoo_checksums.json uv run pytest tests/zoo_tests

Missing entries are recorded, existing ones are compared and the test fails on a mismatch. Set
``TTS_ZOO_UPDATE_CHECKSUMS=1`` to overwrite existing entries. Checksums are stored per device because
CPU and GPU outputs differ. The latest output of each test is saved to ``<checksums file stem>_wavs/``
for listening.
"""

import hashlib
import json
import os
import random
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

SEED = 1
CHECKSUMS_PATH = os.getenv("TTS_ZOO_CHECKSUMS")
UPDATE_CHECKSUMS = os.getenv("TTS_ZOO_UPDATE_CHECKSUMS") == "1"

# Required for deterministic cuBLAS, must be set before CUDA is initialised
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@pytest.fixture(autouse=True)
def seed_everything():
    """Reseed before each test so outputs don't depend on test order."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    yield
    torch.use_deterministic_algorithms(False)


def _load_checksums(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


@pytest.fixture
def check_output(request: pytest.FixtureRequest, device: torch.device) -> Callable[..., None]:
    """Return a function that records/compares the checksum of a model output.

    Accepts a wav file path or a waveform array with its sample rate. Does nothing if
    ``TTS_ZOO_CHECKSUMS`` isn't set.
    """

    def _check(output: str | os.PathLike | np.ndarray | torch.Tensor, sample_rate: int | None = None, name: str = ""):
        if CHECKSUMS_PATH is None:
            return
        if isinstance(output, (str, os.PathLike)):
            wav, sample_rate = sf.read(output, dtype="int16")
        else:
            if isinstance(output, torch.Tensor):
                output = output.detach().cpu().numpy()
            assert sample_rate is not None, "sample_rate is required for array outputs"
            wav = np.clip(np.asarray(output, dtype=np.float32).squeeze() * 2**15, -(2**15), 2**15 - 1)
            wav = wav.astype(np.int16)
        entry = {
            "sha256": hashlib.sha256(wav.tobytes()).hexdigest(),
            "num_samples": len(wav),
            "sample_rate": sample_rate,
        }

        key = request.node.nodeid + (f"::{name}" if name else "")
        path = Path(CHECKSUMS_PATH)
        wav_dir = path.parent / f"{path.stem}_wavs"
        wav_dir.mkdir(parents=True, exist_ok=True)
        sf.write(wav_dir / (re.sub(r"[^\w.-]+", "_", key) + ".wav"), wav, sample_rate)

        # Read-modify-write after every check so results survive interrupted runs
        checksums = _load_checksums(path)
        device_checksums = checksums.setdefault(device.type, {})
        expected = device_checksums.get(key)
        if expected is None or UPDATE_CHECKSUMS:
            device_checksums[key] = entry
            path.write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n")
        elif expected != entry:
            pytest.fail(f"Output changed for {key} on {device.type}:\n  expected {expected}\n  got      {entry}")

    return _check
