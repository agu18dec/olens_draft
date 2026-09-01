"""Storage root for oracle-lens artifacts — one env var.

``OLA_ROOT`` unset ⇒ ``./artifacts`` relative to the working directory (run commands from the
repo root). Point it at a large disk for real runs: ``export OLA_ROOT=/data/ola``. The fetch and
production scripts create subdirs under it (``rollouts_iolens/``, ``ml_pairs_iolens_*/``,
``ao_pool/``, ``ml_checkpoints/``, …).
"""

import os
import subprocess
from pathlib import Path

DEFAULT_ROOT = Path("artifacts")


def ola_root() -> Path:
    """The artifact root: ``$OLA_ROOT`` if set, else ``./artifacts``."""
    return Path(os.environ["OLA_ROOT"]) if os.environ.get("OLA_ROOT") else DEFAULT_ROOT


def git_commit() -> str:
    """Provenance: ``$GIT_COMMIT`` if set, else ask git, else unknown."""
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def hf_offline() -> None:
    """Force offline HF reads (model+tokenizer fully cached). Cold-start races against a shared
    cache ('does not appear to have a file named model-000NN-of-00015') came from concurrent hub
    re-resolution; offline mode reads the pinned snapshot only."""
    os.environ["HF_HUB_OFFLINE"] = "1"
