from unittest.mock import patch

import TTS.utils.io as io_utils
from TTS.utils.io import open_fsspec


def _captured_kwargs(path, **kwargs):
    """Return the kwargs ``open_fsspec`` forwards to ``fsspec.open`` for ``path``."""
    with patch.object(io_utils.fsspec, "open") as mocked:
        open_fsspec(path, **kwargs)
    return mocked.call_args.kwargs


def test_local_path_is_untouched():
    kwargs = _captured_kwargs("/tmp/model.pth", mode="rb")
    assert "config_kwargs" not in kwargs


def test_http_path_is_untouched():
    kwargs = _captured_kwargs("https://example.com/model.pth")
    assert "config_kwargs" not in kwargs


def test_s3_path_gets_distribution_tag():
    kwargs = _captured_kwargs("s3://bucket/model.pth", mode="rb")
    assert kwargs["config_kwargs"]["user_agent_extra"] == io_utils._distribution_tag()


def test_s3_path_appends_to_existing_user_agent_extra():
    kwargs = _captured_kwargs(
        "s3://bucket/model.pth",
        config_kwargs={"user_agent_extra": "custom/1.0"},
    )
    user_agent = kwargs["config_kwargs"]["user_agent_extra"]
    assert user_agent.startswith("custom/1.0 ")
    assert user_agent.endswith(io_utils._distribution_tag())


def test_s3_path_keeps_other_storage_options():
    kwargs = _captured_kwargs("s3://bucket/model.pth", key="id", secret="key")
    assert kwargs["key"] == "id"
    assert kwargs["secret"] == "key"
    assert "config_kwargs" in kwargs


def test_distribution_tag_falls_back_to_dev():
    def _raise(_name):
        raise io_utils.importlib.metadata.PackageNotFoundError

    with patch.object(io_utils.importlib.metadata, "version", _raise):
        assert io_utils._distribution_tag() == "coqui-tts/dev"
