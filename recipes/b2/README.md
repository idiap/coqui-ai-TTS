# Backblaze B2 (S3-compatible) storage

[Backblaze B2](https://www.backblaze.com/cloud-storage) exposes an S3-compatible
API, so 🐸TTS can read and write checkpoints, configs, and datasets straight
from a B2 bucket through the existing [fsspec](https://filesystem-spec.readthedocs.io/)
layer. The same approach works for any S3-compatible store, including AWS S3.

`load_from_b2.py` loads a checkpoint from `s3://<bucket>/...` against a B2
endpoint:

```bash
pip install s3fs

export B2_APPLICATION_KEY_ID="..."          # bucket-scoped key, NOT the master key
export B2_APPLICATION_KEY="..."
export B2_ENDPOINT_URL="https://s3.us-west-004.backblazeb2.com"
export B2_CHECKPOINT_URI="s3://my-bucket/checkpoints/best_model.pth"

python recipes/b2/load_from_b2.py
```

See [`docs/source/remote_storage.md`](../../docs/source/remote_storage.md) for
the full pattern, including loading configs and the credential conventions.
