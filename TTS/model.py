import os
import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Any, Self

import torch
from coqpit import Coqpit
from trainer import TrainerModel
from trainer.io import load_fsspec

from TTS.config.shared_configs import BaseTrainingConfig


class BaseTrainerModel(TrainerModel):
    """BaseTrainerModel model expanding TrainerModel with required functions by 🐸TTS.

    Every new Coqui model must inherit it.
    """

    config: BaseTrainingConfig

    @classmethod
    def init_from_config(cls, config: Coqpit) -> Self:
        """Init the model and all its attributes from the given config.

        Override this depending on your model.
        """
        warnings.warn(
            f"{cls.__name__}.init_from_config(config) is deprecated and will be removed soon, "
            f"just initialize with {cls.__name__}(config)",
            UserWarning,
            stacklevel=2,
        )
        return cls(config)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | os.PathLike[Any],
        config_path: str | os.PathLike[Any] | None = None,
    ) -> Self:
        """Load a pre-trained model in Coqui format for inference.

        Args:
            model_path: Path to the model checkpoint file, or to a directory for
                models that load multiple files from a directory.
            config_path: Path to the model config. Defaults to ``config.json``
                in the checkpoint directory.
        """
        from TTS.config import load_config

        model_path = Path(model_path)
        model_dir = model_path if model_path.is_dir() else model_path.parent
        config = load_config(config_path if config_path is not None else model_dir / "config.json")
        model = cls(config)
        if model_path.is_dir():
            model.load_checkpoint(config, checkpoint_dir=model_path, eval=True)
        else:
            model.load_checkpoint(config, checkpoint_path=model_path, eval=True)
        return model

    @abstractmethod
    def inference(self, input: torch.Tensor, aux_input: dict[str, Any] | None = None) -> dict[str, Any]:
        """Forward pass for inference.

        Must return a dictionary with the main model output and all the auxiliary outputs. The key ```model_outputs```
        is considered to be the main output and you can add any other auxiliary outputs as you want.

        We don't use `*kwargs` since it is problematic with the TorchScript API.

        Args:
            input (torch.Tensor): [description]
            aux_input (Dict): Auxiliary inputs like speaker embeddings, durations etc.

        Returns:
            Dict: [description]

        """
        if aux_input is None:
            aux_input = {}
        outputs_dict = {"model_outputs": None}
        ...
        return outputs_dict

    def load_checkpoint(
        self,
        config: Coqpit,
        checkpoint_path: str | os.PathLike[Any],
        *,
        eval: bool = False,
        strict: bool = True,
        cache: bool = False,
        **kwargs: Any,
    ) -> None:
        """Load a model checkpoint file and get ready for training or inference.

        Args:
            config (Coqpit): Model configuration.
            checkpoint_path (str | os.PathLike): Path to the model checkpoint file.
            eval (bool, optional): If true, init model for inference else for training. Defaults to False.
            strict (bool, optional): Match all checkpoint keys to model's keys. Defaults to True.
            cache (bool, optional): If True, cache the file locally for subsequent calls.
                It is cached under `trainer.io.get_user_data_dir()/tts_cache`. Defaults to False.

        """
        state = load_fsspec(checkpoint_path, map_location="cpu", cache=cache)
        self.load_state_dict(state["model"], strict=strict)
        if eval:
            self.eval()

    @property
    def device(self) -> torch.device:
        """Return device of the model based on its parameters."""
        return next(self.parameters()).device
