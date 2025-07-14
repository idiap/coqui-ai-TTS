import datetime
import importlib
import logging
import os
import re
import unicodedata
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO, TypeVar

import torch
from packaging.version import Version
from typing_extensions import TypeIs

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def exists(val: _T | None) -> TypeIs[_T]:
    return val is not None


def default(val: _T | None, d: _T | Callable[[], _T]) -> _T:
    if exists(val):
        return val
    return d() if callable(d) else d


def to_camel(text):
    text = text.capitalize()
    text = re.sub(r"(?!^)_([a-zA-Z])", lambda m: m.group(1).upper(), text)
    text = text.replace("Tts", "TTS")
    text = text.replace("vc", "VC")
    text = text.replace("Knn", "KNN")
    return text


def slugify(text: str) -> str:
    """Convert a string (e.g. speaker IDs) into a safe filename base."""
    # Normalize to ASCII (e.g., Zoë -> Zoe)
    normalized = unicodedata.normalize("NFKD", text)
    ascii_str = normalized.encode("ascii", "ignore").decode("ascii")

    # Replace unsafe characters with underscores
    safe = re.sub(r"[^\w\-]", "_", ascii_str)

    # Collapse repeated underscores
    return re.sub(r"_+", "_", safe).strip("_")


def find_module(module_path: str, module_name: str) -> type[Any]:
    module_name = module_name.lower()
    module = importlib.import_module(module_path + "." + module_name)
    class_name = to_camel(module_name)
    return getattr(module, class_name)


def import_class(module_path: str) -> type[Any]:
    """Import a class from a module path.

    Args:
        module_path (str): The module path of the class.

    Returns:
        object: The imported class.
    """
    class_name = module_path.split(".")[-1]
    module_path = ".".join(module_path.split(".")[:-1])
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_import_path(obj: object) -> str:
    """Get the import path of a class.

    Args:
        obj (object): The class object.

    Returns:
        str: The import path of the class.
    """
    return ".".join([type(obj).__module__, type(obj).__name__])


def format_aux_input(def_args: dict, kwargs: dict) -> dict:
    """Format kwargs to hande auxilary inputs to models.

    Args:
        def_args (Dict): A dictionary of argument names and their default values if not defined in `kwargs`.
        kwargs (Dict): A `dict` or `kwargs` that includes auxilary inputs to the model.

    Returns:
        Dict: arguments with formatted auxilary inputs.
    """
    kwargs = kwargs.copy()
    for name, arg in def_args.items():
        if name not in kwargs or kwargs[name] is None:
            kwargs[name] = arg
    return kwargs


def get_timestamp() -> str:
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


class ConsoleFormatter(logging.Formatter):
    """Custom formatter that prints logging.INFO messages without the level name.

    Source: https://stackoverflow.com/a/62488520
    """

    def format(self, record):
        if record.levelno == logging.INFO:
            self._style._fmt = "%(message)s"
        else:
            self._style._fmt = "%(levelname)s: %(message)s"
        return super().format(record)


def setup_logger(
    logger_name: str,
    level: int = logging.INFO,
    *,
    formatter: logging.Formatter | None = None,
    stream: TextIO | None = None,
    log_dir: str | os.PathLike[Any] | None = None,
    log_name: str = "log",
) -> None:
    """Set up a logger.

    Args:
        logger_name: Name of the logger to set up
        level: Logging level
        formatter: Formatter for the logger
        stream: Add a StreamHandler for the given stream, e.g. sys.stderr or sys.stdout
        log_dir: Folder to write the log file (no file created if None)
        log_name: Prefix of the log file name
    """
    lg = logging.getLogger(logger_name)
    if formatter is None:
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d - %(levelname)-8s - %(name)s: %(message)s", datefmt="%y-%m-%d %H:%M:%S"
        )
    lg.setLevel(level)
    if log_dir is not None:
        Path(log_dir).mkdir(exist_ok=True, parents=True)
        log_file = Path(log_dir) / f"{log_name}_{get_timestamp()}.log"
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if stream is not None:
        sh = logging.StreamHandler(stream)
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def is_pytorch_at_least_2_4() -> bool:
    """Check if the installed Pytorch version is 2.4 or higher."""
    return Version(torch.__version__) >= Version("2.4")


def optional_to_str(x: Any | None) -> str:
    """Convert input to string, using empty string if input is None."""
    return "" if x is None else str(x)


def warn_synthesize_config_deprecated() -> None:
    warnings.warn(
        "The `config` argument of synthesize() is deprecated and will be removed soon. You can safely leave it out.",
        DeprecationWarning,
    )


def warn_synthesize_speaker_id_deprecated() -> None:
    warnings.warn(
        "The `speaker_id` argument of synthesize() is deprecated and will be removed soon. Use `speaker` instead.",
        DeprecationWarning,
    )
