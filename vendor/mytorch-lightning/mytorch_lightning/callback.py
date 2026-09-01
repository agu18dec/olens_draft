from typing import TYPE_CHECKING

from mytorch_lightning.config import TrainingConfig

if TYPE_CHECKING:
    from mytorch_lightning.trainer import Trainer


class Callback:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.trainer: "Trainer" | None = None

    def on_train_start(self):
        pass

    def on_train_end(self):
        pass

    def on_train_epoch_start(self):
        pass

    def on_train_epoch_end(self):
        pass

    def on_train_step_start(self):
        pass

    def on_train_step_end(self):
        pass

    def on_validation_epoch_start(self):
        pass

    def on_validation_epoch_end(self):
        pass

    def on_validation_step_start(self):
        pass

    def on_validation_step_end(self):
        pass

    def on_before_optimizer_step(self):
        pass

    def on_after_optimizer_step(self):
        pass
