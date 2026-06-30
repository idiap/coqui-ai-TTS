# Remote storage (S3-compatible object stores)

🐸TTS loads checkpoints, configs, and datasets through
[fsspec](https://filesystem-spec.readthedocs.io/), which is already pinned in
the dependencies (`fsspec[http]>=2023.6.0`). Anywhere the codebase opens a path
through fsspec, you can pass any URL fsspec understands, including `s3://`
against an S3-compatible backend such as
[Backblaze B2](https://www.backblaze.com/cloud-storage).

This page shows the minimum configuration needed to load a checkpoint from
**Backblaze B2** via its S3-compatible API. The same steps apply to any
S3-compatible store, including AWS S3.

## Install the S3 fsspec backend

`s3fs` is the fsspec implementation for S3-compatible stores; install it
alongside TTS:

```bash
pip install s3fs
```

## Configure credentials and endpoint

A Backblaze B2 bucket has a region-specific S3 endpoint of the form
`https://s3.<region>.backblazeb2.com` (find yours in the Backblaze web console
under *Buckets → Endpoint*). AWS S3 needs no endpoint override.

`s3fs` (through `boto3`/`aiobotocore`) reads the standard AWS environment
variables. Map your B2 Application Key onto them once and every fsspec call,
including the checkpoint loader, works without extra arguments:

```bash
export AWS_ACCESS_KEY_ID="$B2_APPLICATION_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$B2_APPLICATION_KEY"
export AWS_ENDPOINT_URL_S3="https://s3.us-west-004.backblazeb2.com"
```

For AWS S3, set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` and leave
`AWS_ENDPOINT_URL_S3` unset.

## Load a checkpoint from B2

With the environment configured, pass an `s3://` path anywhere TTS expects a
checkpoint. `trainer.io.load_fsspec` opens the path with a default `s3fs`
filesystem, so no extra arguments are needed:

```python
from trainer.io import load_fsspec

state = load_fsspec("s3://my-bucket/checkpoints/best_model.pth", map_location="cpu")
```

A runnable version of this is in
[`recipes/b2/load_from_b2.py`](https://github.com/idiap/coqui-ai-TTS/blob/dev/recipes/b2/load_from_b2.py).

## Load configs and manifests from B2

`TTS.config.load_config` and the speaker/embedding managers open their files
through fsspec as well, so the same `s3://` paths work once the environment is
configured:

```python
from TTS.config import load_config

config = load_config("s3://my-bucket/configs/config.json")
```

If you would rather not rely on the environment, the lower-level fsspec entry
points accept per-call connection options. For example, to read a config with
an explicit endpoint:

```python
import json

import fsspec

with fsspec.open(
    "s3://my-bucket/configs/config.json",
    "r",
    encoding="utf-8",
    key="<B2_APPLICATION_KEY_ID>",
    secret="<B2_APPLICATION_KEY>",
    client_kwargs={"endpoint_url": "https://s3.us-west-004.backblazeb2.com"},
) as f:
    config_dict = json.load(f)
```

## Conventions and caveats

- **Env-var convention used in our recipes.** We standardise on
  `B2_APPLICATION_KEY_ID` and `B2_APPLICATION_KEY` for B2 credentials, plus
  `B2_ENDPOINT_URL` for the endpoint, and map them onto `AWS_ACCESS_KEY_ID` /
  `AWS_SECRET_ACCESS_KEY` / `AWS_ENDPOINT_URL_S3` so any S3-aware library
  (s3fs, boto3, aiobotocore) picks them up. See
  [`recipes/b2/load_from_b2.py`](https://github.com/idiap/coqui-ai-TTS/blob/dev/recipes/b2/load_from_b2.py)
  for the canonical wiring.
- **Use bucket-scoped Application Keys, not the master key.** Create a
  dedicated Application Key in the Backblaze console scoped to a single bucket
  with the minimum capabilities needed (typically `listFiles`, `readFiles`,
  and, only when writing, `writeFiles`). Never embed the master key in env
  files or scripts.
- **Endpoint must match the bucket's region.** Each B2 bucket lives in a single
  region (e.g. `us-west-004`); using the wrong endpoint returns 401.

## See also

- Recipe: [`recipes/b2/load_from_b2.py`](https://github.com/idiap/coqui-ai-TTS/blob/dev/recipes/b2/load_from_b2.py)
- fsspec entry points in TTS: `TTS/utils/io.py` (`open_fsspec`),
  `TTS/config/__init__.py`, `TTS/tts/utils/managers.py`, and every call site of
  `trainer.io.load_fsspec` (`TTS/model.py`,
  `TTS/encoder/models/base_encoder.py`, etc.).
- [fsspec docs](https://filesystem-spec.readthedocs.io/) ·
  [s3fs docs](https://s3fs.readthedocs.io/) ·
  [Backblaze B2 S3-compatible API](https://www.backblaze.com/docs/cloud-storage-s3-compatible-api).
```
