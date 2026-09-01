from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist
from torch import Tensor, nn

ValueType = Tensor | Callable[[], Tensor]


@dataclass
class LoggingDetails:
    reduction: str
    across_devices: bool
    prog_bar: bool
    prog_bar_name: str | None
    on_epoch: bool | None
    on_step: bool | None


@dataclass
class LogValue:
    val: Tensor
    details: LoggingDetails


class Reducer:
    def __init__(
        self,
        details: LoggingDetails,
        device,
    ):
        self.details = details

        self.accumulator = torch.zeros(1, device=device)
        self.running_count = torch.zeros(1, device=device, dtype=torch.long)

    def accumulate(self, t: Tensor):
        match self.details.reduction:
            case "mean":
                self.accumulator += t.sum()
                self.running_count += t.numel()
            case "sum":
                self.accumulator += t.sum()
            case "max":
                self.accumulator = torch.max(self.accumulator, t.max())
            case "min":
                self.accumulator = torch.min(self.accumulator, t.min())
            case _:
                raise ValueError(f"Invalid reduction: {self.details.reduction}")

    def merge(self, other: "Reducer"):
        assert self.details == other.details

        match self.details.reduction:
            case "mean":
                self.accumulator += other.accumulator
                self.running_count += other.running_count
            case "sum":
                self.accumulator += other.accumulator
            case "max":
                self.accumulator = torch.max(self.accumulator, other.accumulator)
            case "min":
                self.accumulator = torch.min(self.accumulator, other.accumulator)
            case _:
                raise ValueError(f"Invalid reduction: {self.details.reduction}")

    def get_value(self) -> Tensor:
        reduce_across_devices = self.details.across_devices and dist.is_initialized()

        match self.details.reduction:
            case "mean":
                if reduce_across_devices:
                    total_count = self.running_count.clone()
                    total_sum = self.accumulator.clone()
                    dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
                    dist.all_reduce(total_sum, op=dist.ReduceOp.SUM)
                    return total_sum / total_count
                else:
                    return self.accumulator / self.running_count
            case "sum":
                if reduce_across_devices:
                    total_sum = self.accumulator.clone()
                    dist.all_reduce(total_sum, op=dist.ReduceOp.SUM)
                    return total_sum
                else:
                    return self.accumulator
            case "max":
                if reduce_across_devices:
                    total_max = self.accumulator.clone()
                    dist.all_reduce(total_max, op=dist.ReduceOp.MAX)
                    return total_max
                else:
                    return self.accumulator
            case "min":
                if reduce_across_devices:
                    total_min = self.accumulator.clone()
                    dist.all_reduce(total_min, op=dist.ReduceOp.MIN)
                    return total_min
                else:
                    return self.accumulator
            case _:
                raise ValueError(f"Invalid reduction: {self.details.reduction}")


def collect_logs(
    base_mod: nn.Module,
    clear: bool = True,
):
    raw_collected_logs: dict[str, list[LogValue]] = defaultdict(list)

    for mod in base_mod.modules():
        if isinstance(mod, LoggingModule):
            for k, v_list in mod._log_dict.items():
                for v in v_list:
                    raw_collected_logs[k].append(v)

    collected_logs: dict[str, Reducer] = {}

    for k, v_list in raw_collected_logs.items():
        details = [v.details for v in v_list]

        for i in range(1, len(details)):
            if details[0] != details[i]:
                raise ValueError(
                    f"Accumulating logs for key '{k}' that are configured differently: {details[0]} != {details[i]}"
                )

        reducer = Reducer(
            details=details[0],
            device=v_list[0].val.device,
        )

        for v in v_list:
            reducer.accumulate(v.val)

        collected_logs[k] = reducer

    if clear:
        LoggingModule.clear_logs(base_mod)

    return collected_logs


class LoggingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._clear_log_dict()
        self.logging_enabled = True

    def set_logging_enabled(self, enabled: bool):
        for m in self.modules():
            if isinstance(m, LoggingModule):
                m.logging_enabled = enabled

    def _clear_log_dict(self):
        self._log_dict: dict[str, list[LogValue]] = defaultdict(list)

    def clear_logs(self, recursive: bool = True):
        if not recursive:
            self._clear_log_dict()
        else:
            for m in self.modules():
                if isinstance(m, LoggingModule):
                    m._clear_log_dict()

    def log(
        self,
        key: str,
        value: ValueType,
        on_epoch: bool | None = None,
        on_step: bool | None = None,
        reduction: str = "mean",
        across_devices: bool = True,
        prog_bar: bool = False,
        prog_bar_name: str | None = None,
    ):
        if not self.logging_enabled:
            return

        if callable(value):
            tensor_value = value()
            assert isinstance(tensor_value, Tensor), (
                "logged callables must return a Tensor"
            )
        else:
            assert isinstance(value, Tensor), "logged values must be a Tensor"
            tensor_value = value

        details = LoggingDetails(
            reduction=reduction,
            across_devices=across_devices,
            prog_bar=prog_bar,
            prog_bar_name=prog_bar_name,
            on_epoch=on_epoch,
            on_step=on_step,
        )
        value_obj = LogValue(
            val=tensor_value.detach(),
            details=details,
        )

        self._log_dict[key].append(value_obj)
