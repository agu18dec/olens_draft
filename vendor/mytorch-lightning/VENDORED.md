# Vendored: jordan-benjamin/mytorch-lightning

Local clone of the **private** `mytorch-lightning` training harness, so this repo builds without
access to the upstream private GitHub repo.

- Source: https://github.com/jordan-benjamin/mytorch-lightning (private)
- Commit: `3cd74228aefac2e5c08bf26ec988bf834dc8bce4` (`3cd7422`)
- Vendored: 2026-07-20 (carried into this extracted repo 2026-09-01)

## ⚠️ Before making this repo public

Upstream is a PRIVATE repository. Vendoring it here is fine for private sharing, but **get the
upstream owner's permission before publishing this repo publicly, or replace the dependency**
(the surface used is small: `Mydule`, `TrainingConfig`, `train`/`do_train`, the checkpoint
callbacks — see `src/oracle_lens/pipeline/{train_recon,soft_token_sft,resume}.py`).

## How it's wired

`pyproject.toml` installs it from this local path (`[tool.uv.sources] mytorch-lightning = { path =
"vendor/mytorch-lightning" }`), so `uv sync` picks it up here — no private-repo credentials needed.
It is a normal installable package (has its own `pyproject.toml` and declares its own deps), **not a
symlink**. Import it as usual:

```python
from mytorch_lightning.mydule import Mydule
from mytorch_lightning.config import TrainingConfig
from mytorch_lightning.entry import do_train, train
```

Usage examples in this repo: `src/oracle_lens/pipeline/train_recon.py` (`ReconMydule`) — subclass `Mydule`, build a `TrainingConfig`, launch single-GPU
via `train(tc, mydule)` or DDP via `mp.spawn` + `do_train(tc, local_rank, mydule)` per rank.

## Updating

Re-clone at the new commit, delete the nested `.git`, and update the commit hash above:

```
rm -rf vendor/mytorch-lightning
git clone https://github.com/jordan-benjamin/mytorch-lightning vendor/mytorch-lightning
( cd vendor/mytorch-lightning && git checkout <rev> && rm -rf .git )
uv sync
```

Treat as a frozen copy — do not edit the sources here; local training code lives in
`src/oracle_lens`.
