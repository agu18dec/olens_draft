"""Always-save / always-resume for multi-layer AR runs (lightweight, LoRA-scale).

The harness's own checkpointing (``save_every_n_steps`` → ``AppState``) serialises the WHOLE
PEFT-wrapped 27B (~54 GB) per save — unusable on NFS. This callback saves only the trainable
state — ``{lora, head, layer_emb, optimizer, global_step}`` (~2-3 GB fp32) — atomically
(``.tmp`` → ``os.replace``), rank-0 only, every ``save_every_steps`` optimizer steps and at
train end. On start, every rank loads the latest ``resume.pt`` if present (replicas are
identical under DDP, so all ranks may read the same file), and the caller fast-forwards the
data order via the bucket sampler's ``skip_batches`` so a resumed run is step-exact: no span is
seen twice, none skipped.
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
    """The latest saved trainable state, or None. Meta (step count) rides inside the blob."""
    import torch

    p = resume_path(ckpt_dir)
    if not p.exists():
        return None
    state: dict[str, Any] = torch.load(p, map_location="cpu", weights_only=False)
    return state


class ResumeCheckpoint(Callback):  # type: ignore[misc]
    """Periodic lightweight save of the trainable state (rank-0), for always-resume."""

    def __init__(
        self,
        config: TrainingConfig,
        *,
        recon: Any,
        peft_inner: Any,
        ckpt_dir: Path,
        save_every_steps: int,
        rank: int,
    ) -> None:
        super().__init__(config)
        self._recon = recon
        self._peft = peft_inner
        self._dir = ckpt_dir
        self._every = max(1, save_every_steps)
        self._rank = rank
        self._last_saved = -1
        self.restore_step: int | None = None
        self.optimizer_state: dict[str, Any] | None = None

    def on_train_start(self) -> None:
        if self.trainer is None or self.restore_step is None:
            return
        self.trainer.global_step = self.restore_step
        if self.optimizer_state and self.trainer.optimizer is not None:
            self.trainer.optimizer.load_state_dict(self.optimizer_state)
        print(f"[resume] trainer fast-forwarded to step {self.restore_step} "
              f"(optimizer state {'restored' if self.optimizer_state else 'absent'})", flush=True)

    def _save(self, tag: str) -> None:
        if self._rank != 0 or self.trainer is None:
            return
        import torch

        step = int(self.trainer.global_step)
        if step == self._last_saved:
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        lora = {
            k: v.detach().cpu()
            for k, v in self._peft.state_dict().items()
            if "lora" in k.lower()
        }
        blob: dict[str, Any] = {
            "global_step": step,
            "lora": lora,
            "optimizer": self.trainer.optimizer.state_dict() if self.trainer.optimizer else {},
        }
        if hasattr(self._recon, "head"):
            blob["head"] = self._recon.head.state_dict()
        if hasattr(self._recon, "layer_emb"):
            blob["layer_emb"] = self._recon.layer_emb.state_dict()
        tmp = resume_path(self._dir).with_suffix(".pt.tmp")
        torch.save(blob, tmp)
        os.replace(tmp, resume_path(self._dir))
        (self._dir / "resume_meta.json").write_text(
            json.dumps({"global_step": step, "saved_at": tag, "wall_s": round(time.time() - t0, 1)})
        )
        self._last_saved = step
        print(f"[resume] saved step {step} ({tag}, {time.time() - t0:.0f}s)", flush=True)

    def on_train_step_end(self) -> None:
        if self.trainer is None:
            return
        step = int(self.trainer.global_step)
        if step > 0 and step % self._every == 0:  # step 0 would save the fresh init
            self._save("periodic")

    def on_train_end(self) -> None:
        self._save("final")


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
