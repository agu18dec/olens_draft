from datetime import datetime, timedelta
from pathlib import Path

import pydra
import torch
from torch.distributed.fsdp import MixedPrecisionPolicy


class WandbConfig(pydra.Config):
    def __init__(self):
        self.project: str | None = None
        self.entity: str | None = None
        self.tags: list[str] = []
        self.group: str | None = None
        self.resume: bool = False


class FSDP_MixedPrecisionPolicy(pydra.Config):
    def __init__(self):
        self.param_dtype: str | None = None
        self.reduce_dtype: str | None = None
        self.output_dtype: str | None = None
        self.cast_forward_inputs: bool = True

    def build(self) -> MixedPrecisionPolicy:
        def get_dtype(dtype: str | None) -> torch.dtype | None:
            if dtype is None:
                return None
            return getattr(torch, dtype)

        return MixedPrecisionPolicy(
            param_dtype=get_dtype(self.param_dtype),
            reduce_dtype=get_dtype(self.reduce_dtype),
            output_dtype=get_dtype(self.output_dtype),
            cast_forward_inputs=self.cast_forward_inputs,
        )


class TrainingConfig(pydra.Config):
    def __init__(self):
        # number of gpus per node
        self.ngpu: int = 1
        self.nodes: int = 1
        self.node_rank: int = 0

        self.master_addr: str = "localhost"
        self.master_port: int = 10210
        self.dist_timeout: timedelta | None = None

        # strategy
        self.strategy: str = "ddp"
        self.fsdp_mp = FSDP_MixedPrecisionPolicy()
        self.ddp_find_unused_parameters: bool = False

        self.max_epochs: int | None = None
        self.max_steps: int | None = None
        self.gradient_accumulation_steps: int = 1
        self.amp_dtype: str | None = "bfloat16"

        # validation args
        self.validation_step_interval: int | None = None
        self.validation_epoch_interval: int | None = None
        self.validate_on_start: bool = False
        self.validation_limit_batches: int | None = None

        # logging args
        self.train_log_interval = 1
        self.val_log_interval = 1
        self.logger: str | None = None

        self.wandb_config = WandbConfig()
        self.log_train_step_prefix = "train_step"
        self.log_train_epoch_prefix = "train_epoch"
        self.log_val_step_prefix = "val_step"
        self.log_val_epoch_prefix = "val_epoch"

        self.gradient_clip_val: float | None = None
        self.log_grad_norm = True

        # data args
        self.pin_memory = True
        self.num_workers = 8
        self.persistent_workers = False
        self.load_checkpoint_path: str | None = None

        self.train_shuffle = True
        self.val_shuffle = False

        self.train_drop_last = False
        self.val_drop_last = False

        self.train_batch_size = 1
        self.val_batch_size = 1

        # saving args
        self.save_every_n_steps: int | None = None
        self.save_every_n_epochs: int | None = None
        self.base_save_dir: str = "mtl-runs"
        self.num_checkpoints_to_keep: int | None = None
        self.async_checkpointing: bool = False
        self.save_optimizer: bool = True

        # optimizer args
        self.decay_policy = "all_linears"

        # if the model has a tied param, and in one location our decay policy
        # says to apply weight decay, and in another location it doesn't,
        # this flag decides whether to apply weight decay to the tied param.
        self.decay_tied_params = False

        self.weight_decay = 0.0
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.foreach: bool | None = None
        self.fused: bool | None = None
        self.eps = 1e-8
        self.compile_optimizer = False
        self.cudagraph_optimizer = False

        # lr sched args
        self.lr = 3e-4
        self.lr_sched_type: str | None = None
        self.init_lr = 0.0
        self.final_lr = 0.0
        self.warmup_steps = 0

        self.seed: int | None = None

        self.name: str | None = None

    def world_size(self) -> int:
        return self.ngpu * self.nodes

    def local_to_global_rank(self, local_rank: int) -> int:
        return self.node_rank * self.ngpu + local_rank

    def save_dir(self):
        return Path(self.base_save_dir) / self.name

    def finalize(self):
        if self.name is None:
            self.name = f"run-{datetime.now().strftime('%Y-%m-%d--%H-%M-%S')}"
