"""Force offline HuggingFace reads — one definition, previously eight copies in four variants.

Concurrent hub re-resolution against a shared cache (the Modal volume, or the cluster's NFS
`HF_HOME`) races and 404s mid-download: *"does not appear to have a file named
model-000NN-of-00015"*. Offline mode reads the pinned snapshot only, so every job that loads an
already-cached model calls this before its first `transformers` import.

Stdlib only, so the cluster scripts can call it before any heavy import.
"""

import os

ONLINE_ENV = "AO_HF_ONLINE"


def hf_offline() -> None:
    """Pin HF to the local cache unless ``AO_HF_ONLINE=1`` (for steps that must reach the hub).

    Uses ``setdefault``: an explicitly pre-set ``HF_HUB_OFFLINE`` wins, which is what makes the
    override usable from a job script.
    """
    if os.environ.get(ONLINE_ENV) != "1":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
