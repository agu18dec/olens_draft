import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from mytorch_lightning.mydule import Mydule
from mytorch_lightning.trainer import BatchInfo

MASK_VALUE = -100


def make_labels(batch):
    if "labels" in batch and batch["labels"] is not None:
        return batch["labels"]

    labels = torch.roll(batch["input_ids"], shifts=-1, dims=1)
    labels[:, -1] = MASK_VALUE

    if "loss_mask" in batch and batch["loss_mask"] is not None:
        loss_mask = batch["loss_mask"]
        rolled_loss_mask = torch.roll(loss_mask, shifts=-1, dims=1)
        labels = torch.where(rolled_loss_mask.bool(), labels, MASK_VALUE)

    return labels


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    return (x * mask).sum() / mask.sum()


@dataclass
class LM_Config:
    z_loss_coef: float = 0.0
    init_std: float = 0.02
    init_cutoff_factor: float | None = 3.0

    # reports loss across chunks of the input ids
    loss_chunk_size: int | None = None


def initialize_linears_and_embeddings(config: LM_Config, model: nn.Module):
    def init_normal(param: Tensor):
        if config.init_cutoff_factor is not None:
            cutoff_val = config.init_cutoff_factor * config.init_std
            nn.init.trunc_normal_(
                param, mean=0, std=config.init_std, a=-cutoff_val, b=cutoff_val
            )
        else:
            nn.init.normal_(param, mean=0, std=config.init_std)

    def init_zeros(param: Tensor):
        nn.init.zeros_(param)

    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            init_normal(mod.weight)
            if mod.bias is not None:
                init_zeros(mod.bias)
        elif isinstance(mod, nn.Embedding):
            init_normal(mod.weight)


class LM_Mydule(Mydule):
    def __init__(self, config: LM_Config):
        super().__init__()
        self.config = config
        self.cumulative_tokens = 0
        self.cumulative_tokens_this_opt_step = 0
        self.last_step_timestamp: float | None = None

    def get_logits(self, batch) -> Tensor:
        raise NotImplementedError

    def step_body(self, batch):
        """
        Input format:

        loss_mask: boolean tensor that controls whether a token *should be predicted*.
        Note that this does not mean that the logits at this position should be included in the loss,
        rather it means that the logits at the previous position should be included in the loss.

        """
        logits: Tensor = self.get_logits(batch)
        vocab_size = logits.size(-1)

        labels = make_labels(batch)

        # this is a different mask than the loss_mask passed in. this tells which logits
        # should be included in the loss, not which tokens should be predicted.
        mask = labels != MASK_VALUE

        # TODO: aggregating ce loss across grad accum steps or devices is not correct
        # if different batches have different sequence lengths / masking patterns.

        if (chunk_size := self.config.loss_chunk_size) is not None:
            num_seqs, seq_len = batch["input_ids"].shape
            assert seq_len % chunk_size == 0, (
                f"seq_len {seq_len} must be divisible by loss_chunk_size {chunk_size}"
            )

            ce_losses = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                labels.reshape(-1),
                reduction="none",
                ignore_index=MASK_VALUE,
            )

            ce_losses_as_chunks = rearrange(
                ce_losses,
                "(num_seqs num_chunks chunk_size) -> num_chunks (num_seqs chunk_size)",
                num_seqs=num_seqs,
                chunk_size=chunk_size,
            )

            mask_as_chunks = rearrange(
                mask,
                "num_seqs (num_chunks chunk_size) -> num_chunks (num_seqs chunk_size)",
                num_seqs=num_seqs,
                chunk_size=chunk_size,
            )

            multiply_with_mask = ce_losses_as_chunks * mask_as_chunks
            chunk_mean = multiply_with_mask.sum(dim=-1) / mask_as_chunks.sum(dim=-1)
            ce_loss = chunk_mean.mean()

            num_chunks = seq_len // chunk_size

            for i in range(num_chunks):
                self.log(
                    f"ce_loss_chunks/chunk_{i}",
                    chunk_mean[i],
                )

        else:
            ce_loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                labels.reshape(-1),
                reduction="mean",
                ignore_index=MASK_VALUE,
            )

        self.log("ce_loss", ce_loss)

        loss = ce_loss
        if self.config.z_loss_coef > 0:
            z_loss = masked_mean(logits.logsumexp(-1).square(), mask)
            self.log("z_loss", z_loss)

            loss = loss + z_loss * self.config.z_loss_coef

        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            corrects = predictions == labels
            acc = corrects.sum() / (labels != MASK_VALUE).sum()

            ppl = loss.exp()

        self.log("acc", acc)
        self.log("ppl", ppl)
        self.log("mask_fraction", mask.float().mean())

        if self.training:
            tokens_this_step = batch["input_ids"].numel()
            self.cumulative_tokens += tokens_this_step
            self.cumulative_tokens_this_opt_step += tokens_this_step

        # we don't want this to accumulate across gradient accumulation steps
        if self.training and self.trainer.is_optimizer_step():
            cumulative_tokens = torch.tensor(
                self.cumulative_tokens,
                device=self.device(),
                dtype=torch.float32,
            )

            gtotal_tokens = cumulative_tokens / 1e9

            self.log(
                "progress/total_tokens",
                cumulative_tokens,
                across_devices=True,
                reduction="sum",
            )

            self.log(
                "progress/gtotal_tokens",
                gtotal_tokens,
                across_devices=True,
                reduction="sum",
                prog_bar=True,
                prog_bar_name="gtok",
            )

            delta_from_start = time.time() - self.trainer.train_start_time
            tps = cumulative_tokens / delta_from_start
            gtph = (tps / 1e9) * 3600
            self.log(
                "progress/cumulative_tps", tps, across_devices=True, reduction="sum"
            )
            self.log(
                "progress/cumulative_gtph", gtph, across_devices=True, reduction="sum"
            )

            if self.last_step_timestamp is not None:
                cumulative_tokens_this_opt_step = torch.tensor(
                    self.cumulative_tokens_this_opt_step,
                    device=self.device(),
                    dtype=torch.float32,
                )

                delta = time.time() - self.last_step_timestamp
                tps = cumulative_tokens_this_opt_step / delta
                gtph = (tps / 1e9) * 3600
                self.log("progress/tps", tps, across_devices=True, reduction="sum")
                self.log(
                    "progress/gtph",
                    gtph,
                    across_devices=True,
                    reduction="sum",
                    prog_bar=True,
                    prog_bar_name="gtph",
                )
            self.last_step_timestamp = time.time()
            self.cumulative_tokens_this_opt_step = 0

        # we already log loss in training
        if not self.training:
            self.log("loss", loss, prog_bar=True)

        return loss

    def training_step(self, batch, batch_info: BatchInfo) -> Tensor:
        return self.step_body(batch)

    def validation_step(self, batch, batch_info: BatchInfo) -> Tensor:
        return self.step_body(batch)
