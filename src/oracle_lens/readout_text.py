"""Normalizing raw lens/oracle output text — the one definition of "strip the scaffolding".

Deliberately dependency-free (stdlib ``re`` only) so every consumer can import it: the AR
scorer, the oracle visualizer backend, and the eval pipeline. Three copies of this regex used
to drift independently, guarded only by a test asserting two of them stayed byte-identical.

One consumer still keeps its own copy on purpose:
``scripts/oracle_lens_evals/olens_sglang/common.py`` is imported by the dedicated sglang worker
venv, which has only stdlib + torch + safetensors — not this package. That copy must stay in
sync with this module by hand; its comment says so.
"""

import re

# chat scaffolding + NLA explanation tags that must never reach a tokenizer or a scorer
STRIP_PATTERNS = re.compile(
    r"</?explanation>|<\|im_start\|>(system|user|assistant)?|<\|im_end\|>|<think>|</think>"
)


def strip_scaffolding(text: str) -> str:
    """Strip chat scaffolding + ``<explanation>`` tags from one generated sample."""
    return STRIP_PATTERNS.sub("", text).strip()
