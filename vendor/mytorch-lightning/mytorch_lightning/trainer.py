import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import wandb
import yaml
from loguru import logger
from tabulate import tabulate
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.distributed.fsdp import fully_shard
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset
from tqdm import tqdm

from mytorch_lightning.callback import Callback
from mytorch_lightning.config import TrainingConfig
from mytorch_lightning.logging import Reducer, collect_logs
from mytorch_lightning.lr_schedule import (
    ConstantLR_Scheduler,
    CosineLR_Scheduler,
    LinearLR_Scheduler,
)
from mytorch_lightning.mydule import DataType, Mydule
from mytorch_lightning.utils import seed_everything


def add_prefix(name: str, prefix: str) -> str:
    if prefix.endswith("/"):
        return f"{prefix}{name}"
    else:
        return f"{prefix}/{name}"


@dataclass
class BatchInfo:
    batch_idx: int


@dataclass
class LoggingFlags:
    should_generate_logs_this_step: bool
    should_commit_logs_this_step: bool
class DataLoaderState(Stateful):
    """Wrapper for dataloader state dict to make it work with DCP's Stateful protocol"""
    def __init__(self, state_dict=None):
        self._state = state_dict or {}
    
    def state_dict(self):
        return self._state
    
    def load_state_dict(self, state_dict):
        self._state = state_dict
    
    def get(self):
        return self._state if self._state else None


class AppState(Stateful):
    """
    See these torch docs:

    https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
    https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html
    """

    def __init__(self, model, optimizer=None, epoch_idx=0, global_step=0, dataloader_state=None):
        self.model = model
        self.optimizer = optimizer
        self.epoch_idx = epoch_idx
        self.global_step = global_step
        self.dataloader_state = DataLoaderState(dataloader_state)
        

    def state_dict(self):
        if self.optimizer is not None:
            model_state_dict, optimizer_state_dict = get_state_dict(
                self.model, self.optimizer
            )
        else:
            model_state_dict = get_model_state_dict(self.model)
            optimizer_state_dict = None

        return {
            "model": model_state_dict, 
            "optim": optimizer_state_dict, 
            "epoch_idx": self.epoch_idx, 
            "global_step": self.global_step, 
            "dataloader_state": self.dataloader_state
        }

    def load_state_dict(self, state_dict):
        if self.optimizer is not None:
            set_state_dict(
                self.model,
                self.optimizer,
                model_state_dict=state_dict["model"],
                optim_state_dict=state_dict["optim"],
            )
        else:
            set_model_state_dict(
                self.model,
                state_dict["model"],
            )
        
        self.epoch_idx = state_dict["epoch_idx"]
        self.global_step = state_dict["global_step"]
        if "dataloader_state" in state_dict:
            self.dataloader_state = state_dict["dataloader_state"]


class Trainer:
    def __init__(
        self,
        config: TrainingConfig,
        local_rank: int = 0,
        callbacks: list[Callback] = [],
    ):
        self.config = config
        self.local_rank = local_rank
        self.global_rank = config.local_to_global_rank(local_rank)
        self.mydule: Mydule | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.raw_model: nn.Module | None = None
        self.model: nn.Module | None = None
        self.callbacks = callbacks.copy()

        self.train_start_time: float | None = None
        self.epoch_start_time: float | None = None
        self.step_start_time: float | None = None

        self.async_checkpoint_future: Future | None = None
        self.async_checkpointing_process_group: dist.ProcessGroup | None = None
        self.train_dataloader: DataLoader | None = None
        self.dataloader_state_to_restore: DataLoaderState | None = None
        # Initialize logger with global rank information
        self.setup_logging()

    def get_amp_context_manager(self):
        amp_dtype = (
            getattr(torch, self.config.amp_dtype) if self.config.amp_dtype else None
        )
        if amp_dtype is not None:
            assert isinstance(amp_dtype, torch.dtype)

        return torch.autocast(
            device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None
        )

    def is_rank_zero(self):
        return self.global_rank == 0

    def setup_logging(self):
        """Setup colorful loguru logging following tokasaurus pattern"""
        logger.remove()  # Remove default handler
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level}</level> | "
            "{extra[process_name]} | "
            "<level>{message}</level>",
            level="INFO",
        )
        self.logger = logger.bind(process_name=f"global_rank{self.global_rank}")

    def lprint(self, message: str):
        if self.is_rank_zero():
            self.logger.info(message)

    def setup_distributed(self):
        if self.config.world_size() > 1 or self.config.async_checkpointing:
            torch.cuda.set_device(self.local_rank)

            if self.config.async_checkpointing:
                # https://pytorch.org/blog/reducing-checkpointing-times/
                backend = "cpu:gloo,cuda:nccl"
            else:
                backend = "nccl"

            self.logger.info(
                f"Initializing process group with backend={backend}, world_size={self.config.world_size()}"
            )

            dist.init_process_group(
                backend=backend,
                init_method="env://",
                world_size=self.config.world_size(),
                rank=self.global_rank,
                timeout=self.config.dist_timeout,
            )

        if self.config.async_checkpointing:
            self.logger.info("Initializing async checkpointing process group")
            self.async_checkpointing_process_group = dist.new_group(
                ranks=list(range(self.config.world_size())),
                backend="gloo",
            )

    def fully_shard(self, raw_model: nn.Module, device_mesh: dist.DeviceMesh):
        fsdp_modules = self.mydule.get_fsdp_modules(raw_model)
        mp_policy = self.config.fsdp_mp.build()
        for module in fsdp_modules:
            fully_shard(
                module,
                mesh=device_mesh,
                mp_policy=mp_policy,
            )

        fully_shard(
            raw_model,
            mesh=device_mesh,
            mp_policy=mp_policy,
        )

    def setup_model_and_optimizer(self):
        self.setup_distributed()

        raw_model = self.mydule.create_model()
        self.raw_model = raw_model

        if self.config.world_size() > 1:
            match self.config.strategy:
                case "ddp":
                    device_mesh = dist.device_mesh.init_device_mesh(
                        device_type="cuda",
                        mesh_shape=(self.config.world_size(),),
                        mesh_dim_names=("dp",),
                    )
                    self.mydule.initialize_model(raw_model)
                    model = DistributedDataParallel(
                        raw_model,
                        device_mesh=device_mesh,
                        find_unused_parameters=self.config.ddp_find_unused_parameters,
                    )
                case "fsdp":
                    device_mesh = dist.device_mesh.init_device_mesh(
                        device_type="cuda",
                        mesh_shape=(self.config.world_size(),),
                        mesh_dim_names=("fsdp",),
                    )
                    self.mydule.initialize_model(raw_model)
                    self.fully_shard(raw_model, device_mesh)
                    model = raw_model
                case "hsdp":
                    device_mesh = dist.device_mesh.init_device_mesh(
                        device_type="cuda",
                        mesh_shape=(self.config.nodes, self.config.ngpu),
                        mesh_dim_names=("dp", "fsdp"),
                    )
                    self.fully_shard(raw_model, device_mesh)
                    model = raw_model
                case _:
                    raise ValueError(f"Invalid strategy: {self.config.strategy}")
        else:
            self.mydule.initialize_model(raw_model)
            model = raw_model

        self.model = model

        # Only print model on rank 0
        self.lprint(f"Model:\n{model}")

        #self.print_param_summary()
        self.mydule.finalize_model(model)

        try:
            optimizer = self.mydule.configure_optimizer()
        except NotImplementedError:
            optimizer = self.create_adam_optimizer()

        # Only print optimizer on rank 0
        if self.is_rank_zero():
            self.lprint(f"Optimizer:\n{optimizer}")

        if self.config.compile_optimizer:
            mode = "reduce-overhead" if self.config.cudagraph_optimizer else "default"
            self.logger.info(f"Compiling optimizer with mode={mode}")
            optimizer.step = torch.compile(
                optimizer.step,
                mode=mode,
            )

        self.optimizer = optimizer
        if self.config.load_checkpoint_path:
            self.load_checkpoint(self.config.load_checkpoint_path)
            self.lprint("Optimizer param groups after loading checkpoint:")
            for i, pg in enumerate(self.optimizer.param_groups):
                lr = pg.get("lr", None)
                self.lprint(f"  param_group[{i}]: lr = {lr} (type: {type(lr)})")

    def get_optimizer_step(self):
        return self.global_step // self.config.gradient_accumulation_steps

    def get_step_in_accumulation(self):
        num_steps_completed = self.global_step + 1
        return num_steps_completed % self.config.gradient_accumulation_steps

    def is_zero_grad_step(self):
        return self.global_step % self.config.gradient_accumulation_steps == 0

    def is_optimizer_step(self):
        return (
            self.global_step % self.config.gradient_accumulation_steps
        ) == self.config.gradient_accumulation_steps - 1

    def is_training(self):
        return self.mydule.training

    def should_log_this_step(self):
        if self.is_training():
            interval = self.config.train_log_interval
        else:
            interval = self.config.val_log_interval

        generate_this_step = self.get_optimizer_step() % interval == 0
        commit_this_step = generate_this_step and self.get_step_in_accumulation() == 0
        return LoggingFlags(
            should_generate_logs_this_step=generate_this_step,
            should_commit_logs_this_step=commit_this_step,
        )

    def handle_logs(
        self,
        epoch_log_accumulator: dict[str, Reducer],
        pbar: tqdm,
    ):
        if self.should_log_this_step().should_commit_logs_this_step:
            logs = collect_logs(self.mydule)

            to_log_now: dict[str, Reducer] = {}

            for k, v in logs.items():
                # by default, auto = log on step for training, log on epoch for validation

                on_epoch = v.details.on_epoch
                on_step = v.details.on_step

                if not self.is_training():
                    assert not on_step, "on_step is not supported for validation"

                if on_step or (on_step is None and self.is_training()):
                    to_log_now[k] = v

                if on_epoch or (on_epoch is None and not self.is_training()):
                    if k not in epoch_log_accumulator:
                        epoch_log_accumulator[k] = v

                    epoch_log_accumulator[k].merge(v)

            if len(to_log_now) == 0:
                return

            if self.is_training():
                prefix = self.config.log_train_step_prefix
            else:
                prefix = self.config.log_val_step_prefix

            self.commit_logs(
                to_log_now,
                prefix=prefix,
                pbar=pbar,
            )

    def commit_logs(self, data: dict[str, Reducer], prefix: str, pbar: tqdm):
        values = {k: v.get_value() for k, v in data.items()}

        if self.global_rank == 0:
            items = {add_prefix(k, prefix): v.item() for k, v in values.items()}
            items["global_step"] = self.global_step
            items["epoch"] = self.epoch_idx

            if self.is_training():
                items["times/train_seconds"] = time.time() - self.train_start_time
                items["times/epoch_seconds"] = time.time() - self.epoch_start_time
                items["times/step_seconds"] = time.time() - self.step_start_time

                lrs_per_param_group = []
                for pg in self.optimizer.param_groups:
                    lr = pg["lr"]
                    if isinstance(lr, torch.Tensor):
                        lr = lr.item()
                    lrs_per_param_group.append(lr)

                if len(lrs_per_param_group) == 1:
                    items["optim/lr"] = lrs_per_param_group[0]
                else:
                    for i, lr in enumerate(lrs_per_param_group):
                        items[f"optim/lr_group_{i}"] = lr

            if self.wandb_enabled():
                wandb.log(items, step=self.get_optimizer_step())

            prog_bar_dict = {}
            for k, v in data.items():
                details = v.details
                if details.prog_bar:
                    prog_name = details.prog_bar_name or k
                    full_key = add_prefix(k, prefix)
                    prog_bar_dict[prog_name] = items[full_key]

            pbar.set_postfix(prog_bar_dict)

    def wandb_enabled(self):
        return self.config.logger == "wandb"

    def get_dataloader_length(self, dataloader: DataLoader):
        try:
            return len(dataloader)
        except TypeError:
            return None

    def make_dl(
        self,
        data: DataType,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        update_args_func,
    ) -> DataLoader:
        if isinstance(data, DataLoader):
            return data

        if self.config.world_size() > 1:
            if isinstance(data, IterableDataset):
                print(
                    "WARNING: You're using an iterable dataset with distributed training. You are responsible for manually ensuring that data is not duplicated across processes."
                )
                sampler = None
            else:
                sampler = DistributedSampler(
                    data,
                    num_replicas=self.config.world_size(),
                    rank=self.global_rank,
                    shuffle=shuffle,
                )
        else:
            sampler = None

        dl_args = {
            "dataset": data,
            "sampler": sampler,
            "num_workers": self.config.num_workers,
            "persistent_workers": self.config.persistent_workers,
            "pin_memory": self.config.pin_memory,
            "batch_size": batch_size,
            "drop_last": drop_last,
        }
        if sampler is None:
            dl_args["shuffle"] = shuffle

        dl_args = update_args_func(dl_args)

        return StatefulDataLoader(**dl_args)

    def make_train_dl(self, mydule: Mydule, train_data: DataType) -> DataLoader:
        return self.make_dl(
            train_data,
            batch_size=self.config.train_batch_size,
            shuffle=self.config.train_shuffle,
            drop_last=self.config.train_drop_last,
            update_args_func=mydule.configure_training_dl,
        )

    def make_val_dl(self, mydule: Mydule, val_data: DataType) -> DataLoader:
        return self.make_dl(
            val_data,
            batch_size=self.config.val_batch_size,
            shuffle=self.config.val_shuffle,
            drop_last=self.config.val_drop_last,
            update_args_func=mydule.configure_validation_dl,
        )

    def max_train_steps_reached(self):
        return (
            self.config.max_steps is not None
            and self.get_optimizer_step() >= self.config.max_steps
            and self.is_optimizer_step()
        )

    def max_train_epochs_reached(self):
        return (
            self.config.max_epochs is not None
            and self.epoch_idx >= self.config.max_epochs
        )

    def should_validate_this_step(self):
        return (
            self.config.validation_step_interval is not None
            and self.get_optimizer_step() > 0
            and self.get_optimizer_step() % self.config.validation_step_interval == 0
            and self.is_optimizer_step()
        )

    def should_validate_this_epoch(self):
        return (
            self.config.validation_epoch_interval is not None
            and self.epoch_idx % self.config.validation_epoch_interval == 0
        )

    def should_save_this_step(self):
        return (
            self.config.save_every_n_steps is not None
            and self.get_optimizer_step() % self.config.save_every_n_steps == 0
            and self.is_optimizer_step()
        )

    def should_save_this_epoch(self):
        return (
            self.config.save_every_n_epochs is not None
            and self.epoch_idx % self.config.save_every_n_epochs == 0
        )

    def train_epoch(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
    ):
        self.epoch_start_time = time.time()

        for callback in self.callbacks:
            callback.on_train_epoch_start()

        self.model.train()
        # to offset the tqdm from the start of the epoch
        starting_batch_idx = self.global_step % self.get_dataloader_length(train_dataloader)
        pbar = tqdm(
            enumerate(train_dataloader),
            total=self.get_dataloader_length(train_dataloader),
            initial=starting_batch_idx,
            desc=f"Training Epoch {self.epoch_idx}",
            disable=not self.is_rank_zero(),
        )

        epoch_log_accumulator: dict[str, Reducer] = {}

        for batch_idx, batch in pbar:

            if self.is_zero_grad_step():
                self.step_start_time = time.time()

            batch = self.move_batch_to_device(batch)

            self.mydule.set_logging_enabled(
                self.should_log_this_step().should_generate_logs_this_step
            )

            for callback in self.callbacks:
                callback.on_train_step_start()

            with self.get_amp_context_manager():
                loss = self.mydule.training_step(batch, BatchInfo(batch_idx=batch_idx))
                self.mydule.log("loss", loss, prog_bar=True)

            for callback in self.callbacks:
                callback.on_train_step_end()

            if self.is_zero_grad_step():
                self.optimizer.zero_grad()

            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            if self.is_optimizer_step():
                all_params = []
                for pg in self.optimizer.param_groups:
                    all_params.extend(pg["params"])

                if self.config.gradient_clip_val is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        all_params, self.config.gradient_clip_val
                    )
                elif self.config.log_grad_norm:
                    grad_norm = torch.nn.utils.get_total_norm(all_params)
                else:
                    grad_norm = None

                if self.config.log_grad_norm:
                    assert grad_norm is not None
                    self.mydule.log(
                        "grad_norm",
                        grad_norm,
                    )

                for callback in self.callbacks:
                    callback.on_before_optimizer_step()
                #breakpoint()
                self.optimizer.step()

                for callback in self.callbacks:
                    callback.on_after_optimizer_step()

            self.handle_logs(
                epoch_log_accumulator=epoch_log_accumulator,
                pbar=pbar,
            )

            self.global_step += 1

            if self.should_validate_this_step():
                self.validation_epoch(val_dataloader)

            if self.should_save_this_step():
                self.save_checkpoint()

            if self.max_train_steps_reached():
                break

        if len(epoch_log_accumulator) > 0:
            self.commit_logs(
                epoch_log_accumulator,
                prefix=self.config.log_train_epoch_prefix,
            )

        for callback in self.callbacks:
            callback.on_train_epoch_end()

    def train(
        self,
        mydule: Mydule,
    ):
        self.logger.info(f"Starting training on rank {self.global_rank}")
        self.logger.info(f"Training config: {self.config.to_dict()}")
        # initialize epoch and step early (will be overwritten by checkpoint if loading)
        if not self.config.load_checkpoint_path:
            self.epoch_idx = 0
            self.global_step = 0

        if self.config.seed is not None:
            # TODO: do we need to change the seed like this per rank?
            seed = self.config.seed + self.global_rank
            seed_everything(seed)
            self.logger.info(f"Seeded with seed {seed}")

        self.train_start_time = time.time()

        match self.config.lr_sched_type:
            case "constant":
                self.callbacks.append(ConstantLR_Scheduler(self.config))
            case "linear":
                self.callbacks.append(LinearLR_Scheduler(self.config))
            case "cosine":
                self.callbacks.append(CosineLR_Scheduler(self.config))
            case _:
                raise ValueError(f"Invalid lr_sched_type: {self.config.lr_sched_type}")

        for callback in self.callbacks:
            callback.trainer = self

        for callback in self.callbacks:
            callback.on_train_start()

        self.mydule = mydule
        self.mydule.trainer = self
        self.mydule.train()
        model_optimizer_start_time = time.time()
        self.setup_model_and_optimizer()
        model_optimizer_elapsed_time = time.time() - model_optimizer_start_time
        self.logger.info(
            f"Model and optimizer setup time: {model_optimizer_elapsed_time:.2f}s"
        )
        
        if self.is_rank_zero():
            hyperparams = self.config.to_dict()
            hyperparams.update(mydule.hyperparams_to_log())

            save_dir = self.config.save_dir()
            save_dir.mkdir(parents=True, exist_ok=True)

            if self.wandb_enabled():
                wandb_kwargs = {
                    "project": self.config.wandb_config.project,
                    "name": self.config.name,
                    "entity": self.config.wandb_config.entity,
                    "tags": self.config.wandb_config.tags,
                    "group": self.config.wandb_config.group,
                    "dir": save_dir,
                    "config": hyperparams,
                }
                if self.config.wandb_config.resume:
                    wandb_kwargs["resume"] = self.config.wandb_config.resume
                wandb.init(**wandb_kwargs)

            save_config_path = save_dir / "train_config.yaml"

            with open(save_config_path, "w") as f:
                yaml.dump(hyperparams, f)

            self.logger.info(f"Saved config to {save_config_path}")
        
        data_start_time = time.time()

        # can be a lot faster to use fork for dataloader workers
        # compared to spawn
        mp.set_start_method("fork", force=True)

        train_data = mydule.train_data()
        train_dataloader = self.make_train_dl(mydule, train_data)
        self.train_dataloader = train_dataloader # storing ref for ckpt

        if self.dataloader_state_to_restore:
            train_dataloader.load_state_dict(self.dataloader_state_to_restore)
            self.logger.info("Restored dataloader state from checkpoint")
            del self.dataloader_state_to_restore

        val_data = mydule.val_data()
        if val_data is not None:
            val_dataloader = self.make_val_dl(mydule, val_data)
        else:
            val_dataloader = None

        data_elapsed_time = time.time() - data_start_time
        self.logger.info(f"Data setup time: {data_elapsed_time:.2f}s")

        if val_dataloader is not None and self.config.validate_on_start:
            self.validation_epoch(val_dataloader)

        done = False
        while not done:
            self.train_epoch(train_dataloader, val_dataloader)

            if self.max_train_steps_reached():
                done = True

            self.epoch_idx += 1

            if self.should_validate_this_epoch():
                self.validation_epoch(val_dataloader)

            if self.should_save_this_epoch():
                self.save_checkpoint()

            if self.max_train_epochs_reached():
                done = True

        for callback in self.callbacks:
            callback.on_train_end()

    def validation_epoch(
        self,
        val_data: DataType,
    ):
        for callback in self.callbacks:
            callback.on_validation_epoch_start()

        prev_training_mode = self.mydule.training
        self.mydule.eval()
        self.model.eval()

        val_dataloader = self.make_val_dl(self.mydule, val_data)

        pbar = tqdm(
            enumerate(val_dataloader),
            total=self.get_dataloader_length(val_dataloader),
            desc=f"Validation at epoch={self.epoch_idx} optimizer_step={self.get_optimizer_step()}",
            disable=not self.is_rank_zero(),
        )

        epoch_log_accumulator: dict[str, Reducer] = {}

        for batch_idx, batch in pbar:
            batch = self.move_batch_to_device(batch)

            with self.get_amp_context_manager():
                with torch.no_grad():
                    self.mydule.validation_step(batch, BatchInfo(batch_idx=batch_idx))

            self.handle_logs(
                epoch_log_accumulator=epoch_log_accumulator,
                pbar=pbar,
            )

            if (
                self.config.validation_limit_batches is not None
                and batch_idx >= self.config.validation_limit_batches
            ):
                break

        if len(epoch_log_accumulator) > 0:
            self.commit_logs(
                epoch_log_accumulator,
                prefix=self.config.log_val_epoch_prefix,
                pbar=pbar,
            )

        self.mydule.train(prev_training_mode)

        for callback in self.callbacks:
            callback.on_validation_epoch_end()

    def maybe_prune_checkpoints_before_save(self):
        existing_checkpoints = list(self.config.save_dir().glob("ckpt_*"))
        num_existing_checkpoints = len(existing_checkpoints)

        # +1 because we're about to save a new checkpoint
        if (
            num_to_keep := self.config.num_checkpoints_to_keep
        ) is not None and num_existing_checkpoints + 1 > num_to_keep:
            num_to_remove = num_existing_checkpoints - num_to_keep + 1

            parsed: list[tuple[int, Path]] = []
            for check in existing_checkpoints:
                match = re.match(r"ckpt_epoch_(\d+)_step_(\d+)", check.name)
                assert match is not None
                step = int(match.group(2))
                parsed.append((step, check))

            sorted_checkpoints = sorted(parsed, key=lambda x: x[0])
            for step, check in sorted_checkpoints[:num_to_remove]:
                if self.is_rank_zero():
                    os.system(f"rm -rf {str(check.absolute())}")

    def save_checkpoint(self):
        torch.cuda.synchronize()

        if self.config.world_size() > 1:
            dist.barrier()
        self.lprint(f"Saving checkpoint with dataloader_state: {self.train_dataloader.state_dict()}")
        state_dict = {
            "app": AppState(
                self.model,
                optimizer=self.optimizer if self.config.save_optimizer else None,
                epoch_idx=self.epoch_idx,
                global_step=self.global_step,
                dataloader_state=self.train_dataloader.state_dict(),
            )
        }

        if self.async_checkpoint_future is not None:
            self.async_checkpoint_future.result()

        self.maybe_prune_checkpoints_before_save()

        save_dir = self.config.save_dir()

        ckpt_dir = (
            save_dir / f"ckpt_epoch_{self.epoch_idx}_step_{self.get_optimizer_step()}"
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if self.config.async_checkpointing:
            assert self.async_checkpointing_process_group is not None
            self.async_checkpoint_future = dcp.async_save(
                state_dict,
                checkpoint_id=str(ckpt_dir),
                process_group=self.async_checkpointing_process_group,
            )
        else:
            dcp.save(state_dict, checkpoint_id=str(ckpt_dir))

        # TODO: with async checkpointing, we may point at the latest checkpoint
        # before it's done saving.
        if self.is_rank_zero():
            # symlink to the latest directory
            latest = save_dir / "latest"
            latest.unlink(missing_ok=True)
            latest.symlink_to(ckpt_dir)
    
    def recover_optimizer_param_groups(self):
        """
        DCP doesn't properly restore hyperparameters for empty parameter groups.
        Copy missing keys from a non-empty group to fix this.
        """
        reference_group = None
        for pg in self.optimizer.param_groups:
            if len(pg['params']) > 0:
                reference_group = pg
                break
        
        if reference_group is None:
            return  # all groups empty
        
        for pg in self.optimizer.param_groups:
            for key in reference_group.keys():
                if key not in pg and key != 'params': 
                    pg[key] = reference_group[key]

    def load_checkpoint(self, checkpoint_path: str):
        self.lprint(f"Loading checkpoint from: {checkpoint_path}")
        app_state = AppState(
            self.model,
            optimizer=self.optimizer if self.config.save_optimizer else None,
            epoch_idx=0,
            global_step=0,
            dataloader_state=None,
        )
        state_dict = {"app": app_state}
        dcp.load(state_dict, checkpoint_id=checkpoint_path)
        self.recover_optimizer_param_groups()
        self.lprint("Optimizer param groups after loading checkpoint:")
        for i, pg in enumerate(self.optimizer.param_groups):
            lr = pg.get("lr", None)
            self.lprint(f"  param_group[{i}]: lr = {lr} (type: {type(lr)})")
        self.epoch_idx = app_state.epoch_idx
        self.global_step = app_state.global_step
        self.dataloader_state_to_restore = app_state.dataloader_state.get()
        self.step_start_time = time.time()
        self.lprint(f"Checkpoint loaded successfully (epoch={self.epoch_idx}, step={self.global_step})")
        self.lprint(f"Dataloader state to restore: {self.dataloader_state_to_restore}")
    
    def get_weight_decay_params(self):
        wd_params = set[nn.Parameter]()
        wd_param_names = set[str]()

        if self.config.weight_decay > 0:
            match self.config.decay_policy:
                case "all_linears":
                    for name, mod in self.model.named_modules():
                        if isinstance(mod, nn.Linear) and mod.weight.requires_grad:
                            wd_params.add(mod.weight)
                            wd_param_names.add(f"{name}.weight")

                case _:
                    raise ValueError(
                        f"Invalid decay policy: {self.config.decay_policy}"
                    )

        non_wd_params = set[nn.Parameter]()

        for name, param in self.model.named_parameters():
            if not param.requires_grad or name in wd_param_names:
                continue

            # the interesting case: the param itself is tied to a wd param
            if param in wd_params:
                if self.config.decay_tied_params:
                    continue
                else:
                    wd_params.remove(param)

            non_wd_params.add(param)

        return wd_params, non_wd_params

    def create_adam_optimizer(self):
        wd_params, non_wd_params = self.get_weight_decay_params()

        param_groups = [
            {
                "params": list(wd_params),
                "weight_decay": self.config.weight_decay,
            },
            {"params": list(non_wd_params), "weight_decay": 0.0},
        ]

        if self.config.compile_optimizer:
            lr = torch.tensor(self.config.lr, device=self.device())
        else:
            lr = self.config.lr

        return torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=(self.config.beta1, self.config.beta2),
            foreach=self.config.foreach,
            fused=self.config.fused,
            eps=self.config.eps,
            capturable=self.config.cudagraph_optimizer,
        )

    def device(self):
        return f"cuda:{self.local_rank}"

    def move_batch_to_device(self, batch: dict):
        device = self.device()
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        return batch

    def print_param_summary(
        self,
    ) -> None:
        rows = []

        param_to_names = defaultdict(set)
        for name, param in self.model.named_parameters(remove_duplicate=False):
            param_to_names[param].add(name)

        wd_params, non_wd_params = self.get_weight_decay_params()

        dtype_offset = len("torch.")

        for i, (name, param) in enumerate(
            self.model.named_parameters(remove_duplicate=False)
        ):
            shape = param.shape
            numel = param.numel()
            dtype = str(param.dtype)[dtype_offset:]
            trainable = param.requires_grad
            has_wd = param in wd_params
            tied_names = param_to_names[param] - {name}
            rows.append(
                (
                    i,
                    name,
                    list(shape),
                    format_value(numel),
                    dtype,
                    trainable,
                    has_wd,
                    tied_names,
                )
            )

        table_str = tabulate(
            rows,
            headers=[
                "Idx",
                "Name",
                "Shape",
                "Numel",
                "Dtype",
                "Trainable",
                "Has WD",
                "Tied to",
            ],
            tablefmt="grid",
        )
        self.lprint(table_str)

        total_trainable_params = 0
        total_frozen_params = 0

        for param in param_to_names.keys():
            numel = param.numel()
            if param.requires_grad:
                total_trainable_params += numel
            else:
                total_frozen_params += numel

        self.lprint(f"Trainable params: {format_value(total_trainable_params)}")
        self.lprint(f"Frozen params:    {format_value(total_frozen_params)}")
        self.lprint(
            f"Total params:     {format_value(total_trainable_params + total_frozen_params)}"
        )


def format_value(value: int):
    str_value = str(value)
    num_digits = len(str_value)

    digit_map = {
        3: "",
        6: "K",
        9: "M",
        12: "B",
        15: "T",
        18: "Q",
    }

    for pow10, suffix in digit_map.items():
        if num_digits <= pow10:
            if pow10 == 3:
                val = value
            else:
                val = round(value / 10 ** (pow10 - 3), 4)
            return f"{val}{suffix}"

    return f"{str_value} 👀"
