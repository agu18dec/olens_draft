"""Always-save / always-resume for multi-layer AR runs (lightweight, LoRA-scale).

Adapted from the jspace-ar branch's ``ola/jobs/resume.py``. The harness's own checkpointing
(``save_every_n_steps`` → ``AppState``) serialises the WHOLE PEFT-wrapped 27B (~54 GB) per save
— unusable. These helpers save only the trainable state — ``{lora, head, layer_emb, optimizer,
global_step, tokens_span, world_size}`` (~2-3 GB fp32) — atomically (``.tmp`` →
``os.replace``), rank-0 only. On start, every rank loads the latest ``resume.pt`` if present
(replicas are identical under DDP, so all ranks may read the same file), and the caller
fast-forwards the data order via the streaming dataset's ``skip_batches`` so a resumed run is
step-exact: no span is seen twice, none skipped. Resume REFUSES a changed world size or
DataLoader worker count — the stream replay is only exact at the same (world, workers).
"""

import json
import os
import time
from pathlib import Path
from typing import Any

from mytorch_lightning.callback import Callback
from mytorch_lightning.config import TrainingConfig


def resume_path(ckpt_dir: Path) -> Path:
    return ckpt_dir / "resume.pt"


def load_resume_state(ckpt_dir: Path) -> dict[str, Any] | None:
    """The latest saved trainable state, or None. Meta (step/token counts) rides in the blob."""
    import torch

    p = resume_path(ckpt_dir)
    if not p.exists():
        return None
    state: dict[str, Any] = torch.load(p, map_location="cpu", weights_only=False)
    return state


def save_resume_state(
    ckpt_dir: Path,
    *,
    recon: Any,
    peft_inner: Any,
    optimizer: Any,
    global_step: int,
    tokens_span: int,
    examples: int = 0,
    world_size: int | None = None,
    num_workers: int | None = None,
) -> None:
    """Atomic lightweight save of the trainable state (call from rank 0 only)."""
    import torch

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lora = {
        k: v.detach().cpu() for k, v in peft_inner.state_dict().items() if "lora" in k.lower()
    }
    blob: dict[str, Any] = {
        "global_step": int(global_step),
        "tokens_span": int(tokens_span),
        "examples": int(examples),
        "lora": lora,
        "optimizer": optimizer.state_dict() if optimizer is not None else {},
    }
    if world_size is not None:
        blob["world_size"] = int(world_size)
    if num_workers is not None:
        blob["num_workers"] = int(num_workers)
    if hasattr(recon, "head"):
        blob["head"] = recon.head.state_dict()
    if hasattr(recon, "layer_emb"):
        blob["layer_emb"] = recon.layer_emb.state_dict()
    tmp = resume_path(ckpt_dir).with_suffix(".pt.tmp")
    torch.save(blob, tmp)
    os.replace(tmp, resume_path(ckpt_dir))
    (ckpt_dir / "resume_meta.json").write_text(
        json.dumps(
            {
                "global_step": int(global_step),
                "tokens_span": int(tokens_span),
                "wall_s": round(time.time() - t0, 1),
            }
        )
    )
    print(f"[resume] saved step {global_step} ({time.time() - t0:.0f}s)", flush=True)


def apply_resume_state(
    state: dict[str, Any], *, recon: Any, peft_inner: Any, optimizer: Any = None
) -> int:
    """Load a resume blob into the live modules; returns the saved global_step."""
    peft_state = peft_inner.state_dict()
    peft_state.update({k: v for k, v in state["lora"].items() if k in peft_state})
    peft_inner.load_state_dict(peft_state, strict=False)
    if "head" in state and hasattr(recon, "head"):
        recon.head.load_state_dict(state["head"])
    if "layer_emb" in state and hasattr(recon, "layer_emb"):
        recon.layer_emb.load_state_dict(state["layer_emb"])
    if optimizer is not None and state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])
    return int(state["global_step"])


class RestoreTrainerState(Callback):  # type: ignore[misc]
    """Fast-forward ``trainer.global_step`` + optimizer state on train start (resume path).

    The weights are loaded into the modules BEFORE spawn (``apply_resume_state``); this callback
    only restores the trainer-owned state the modules can't carry.
    """

    def __init__(
        self,
        config: TrainingConfig,
        *,
        restore_step: int,
        optimizer_state: dict[str, Any] | None,
    ) -> None:
        super().__init__(config)
        self._restore_step = restore_step
        self._optimizer_state = optimizer_state

    def on_train_start(self) -> None:
        if self.trainer is None:
            return
        self.trainer.global_step = self._restore_step
        # Fast-forwarding global_step can land it straight on a log-interval boundary, so the
        # harness commits logs before any step has been timed and dies on
        # `time.time() - self.step_start_time` with step_start_time still None. Seed it here;
        # the worst case is one step_seconds reading that measures setup instead of a step.
        if getattr(self.trainer, "step_start_time", None) is None:
            self.trainer.step_start_time = time.time()
        if self._optimizer_state and self.trainer.optimizer is not None:
            self.trainer.optimizer.load_state_dict(self._optimizer_state)
        print(
            f"[resume] trainer fast-forwarded to step {self._restore_step} "
            f"(optimizer state {'restored' if self._optimizer_state else 'absent'})",
            flush=True,
        )
