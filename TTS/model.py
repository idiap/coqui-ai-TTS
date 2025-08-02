import os
from abc import abstractmethod
from typing import Any

import torch
from coqpit import Coqpit
from trainer import TrainerModel
from trainer.io import load_fsspec

# pylint: skip-file


class BaseTrainerModel(TrainerModel):
    """BaseTrainerModel model expanding TrainerModel with required functions by 🐸TTS.

    Every new Coqui model must inherit it.
    """

    @staticmethod
    @abstractmethod
    def init_from_config(config: Coqpit) -> "BaseTrainerModel":
        """Init the model and all its attributes from the given config.

        Override this depending on your model.
        """
        ...

    @abstractmethod
    def inference(self, input: torch.Tensor, aux_input: dict[str, Any] = {}) -> dict[str, Any]:
        """Forward pass for inference.

        It must return a dictionary with the main model output and all the auxiliary outputs. The key ```model_outputs```
        is considered to be the main output and you can add any other auxiliary outputs as you want.

        We don't use `*kwargs` since it is problematic with the TorchScript API.

        Args:
            input (torch.Tensor): [description]
            aux_input (Dict): Auxiliary inputs like speaker embeddings, durations etc.

        Returns:
            Dict: [description]
        """
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
        return next(self.parameters()).device
