import math

from torch import Tensor

from mytorch_lightning.callback import Callback


class LR_Scheduler(Callback):
    def initial_lr(self) -> float:
        return self.config.init_lr

    def calc_lr(self) -> float:
        raise NotImplementedError

    def on_train_start(self):
        """Set LR based on current global_step (handles both fresh start and resume)"""
        if self.trainer.optimizer:
            self.set_lr(self.calc_lr())
        return

    def set_lr(self, lr: float):
        for pg in self.trainer.optimizer.param_groups:
            existing_lr = pg['lr']
            if isinstance(existing_lr, float):
                pg["lr"] = lr
            elif isinstance(existing_lr, Tensor):
                existing_lr.fill_(lr)
            else:
                raise ValueError(f"Invalid lr type: {type(existing_lr)}")

    def on_after_optimizer_step(self):
        self.set_lr(self.calc_lr())


class ConstantLR_Scheduler(LR_Scheduler):
    def calc_lr(self) -> float:
        optimizer_step = self.trainer.get_optimizer_step()

        if optimizer_step >= self.config.warmup_steps:
            return self.config.lr
        else:
            return (
                (self.config.lr - self.config.init_lr)
                * optimizer_step
                / self.config.warmup_steps
            )


class LinearLR_Scheduler(LR_Scheduler):
    def calc_lr(self) -> float:
        optimizer_step = self.trainer.get_optimizer_step()

        if optimizer_step >= self.config.warmup_steps:
            interval = self.config.max_steps - self.config.warmup_steps
            elapsed_steps = optimizer_step - self.config.warmup_steps

            return (
                self.config.lr
                + (self.config.final_lr - self.config.lr) * elapsed_steps / interval
            )
        else:
            return (
                self.config.init_lr
                + (self.config.lr - self.config.init_lr)
                * optimizer_step
                / self.config.warmup_steps
            )


class CosineLR_Scheduler(LR_Scheduler):
    def calc_lr(self) -> float:
        optimizer_step = self.trainer.get_optimizer_step()

        if optimizer_step >= self.config.warmup_steps:
            interval = self.config.max_steps - self.config.warmup_steps
            elapsed_steps = optimizer_step - self.config.warmup_steps

            period = elapsed_steps / interval * math.pi / 2

            return self.config.final_lr + (
                self.config.lr - self.config.final_lr
            ) * math.cos(period)
        else:
            return (
                self.config.init_lr
                + (self.config.lr - self.config.init_lr)
                * optimizer_step
                / self.config.warmup_steps
            )
