"""Load a TTS checkpoint from Backblaze B2 via fsspec / s3fs.

Backblaze B2 exposes an S3-compatible API, so any path that fsspec understands
as ``s3://...`` works as long as the S3 client is pointed at a B2 endpoint.
``trainer.io.load_fsspec`` opens the path with a default ``s3fs`` filesystem,
which reads the standard AWS environment variables. Mapping the B2 credentials
and endpoint onto those variables is all that is needed.

Usage::

    export B2_APPLICATION_KEY_ID="..."          # bucket-scoped key, NOT the master key
    export B2_APPLICATION_KEY="..."
    export B2_ENDPOINT_URL="https://s3.us-west-004.backblazeb2.com"
    export B2_CHECKPOINT_URI="s3://my-bucket/checkpoints/best_model.pth"

    python recipes/b2/load_from_b2.py

See ``docs/source/remote_storage.md`` for the full pattern.
"""

from __future__ import annotations

import os

from trainer.io import load_fsspec


def main() -> None:
    # Map B2 credentials and endpoint onto the standard AWS env vars that
    # s3fs/boto3 read, so a default s3fs filesystem talks to B2.
    os.environ.setdefault("AWS_ACCESS_KEY_ID", os.environ["B2_APPLICATION_KEY_ID"])
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", os.environ["B2_APPLICATION_KEY"])
    os.environ.setdefault("AWS_ENDPOINT_URL_S3", os.environ["B2_ENDPOINT_URL"])

    uri = os.environ["B2_CHECKPOINT_URI"]
    state = load_fsspec(uri, map_location="cpu")
    print(f"Loaded checkpoint from {uri} ({len(state)} top-level keys).")


if __name__ == "__main__":
    main()
