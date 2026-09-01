from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from mytorch_lightning.logging import LoggingModule

if TYPE_CHECKING:
    from mytorch_lightning.trainer import BatchInfo, Trainer


DataType = DataLoader | Dataset


class Mydule(LoggingModule):
    def __init__(self):
        super().__init__()
        self.trainer: "Trainer" | None = None

    def create_model(self) -> nn.Module:
        """
        Create the model object.

        This is distributed sharding, so it's wasteful to
        initialize weights here (i.e. use the meta device if you can).
        """
        raise NotImplementedError

    def initialize_model(self, model: nn.Module):
        """
        Initialize the model weights.
        """
        pass

    def finalize_model(self, model: nn.Module):
        self.model = model

    def configure_optimizer(
        self,
    ) -> torch.optim.Optimizer:
        raise NotImplementedError

    def training_step(self, batch, batch_info: "BatchInfo") -> Tensor:
        raise NotImplementedError

    def validation_step(self, batch, batch_info: "BatchInfo"):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def configure_training_dl(self, args: dict) -> dict:
        return args

    def configure_validation_dl(self, args: dict) -> dict:
        return args

    def hyperparams_to_log(self) -> dict:
        return {}

    def train_data(self) -> DataType:
        raise NotImplementedError

    def val_data(self) -> DataType:
        raise NotImplementedError

    def device(self):
        assert self.trainer is not None
        return self.trainer.device()

    def get_fsdp_modules(self, model: nn.Module) -> list[nn.Module]:
        """
        https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
        """
        raise NotImplementedError
