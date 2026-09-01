"""Single-layer oracle-lens primitives: whitening, pair shards, the truncated-backbone
reconstructor head, NNOMP selection, and span sampling.

These are the shared building blocks the multilayer pipeline (``oracle_lens.pipeline``) is
built on. The single-layer *trainer* was not carried into this repo — train with
``oracle_lens.pipeline`` (see ``docs/pipeline.md``).
"""

from oracle_lens.core.dump import (
    PairBatch,
    PairShardMeta,
    load_pair_shards,
    save_pair_shard,
)
from oracle_lens.core.masks import assistant_token_mask
from oracle_lens.core.reconstructor import (
    Reconstructor,
    ReconstructorHead,
    truncate_backbone,
    whitened_cosine_loss,
)
from oracle_lens.core.sampling import PhraseSpan, sample_phrase_spans, split_of
from oracle_lens.core.stats import MomentAccumulator
from oracle_lens.core.whitening import Whitener, load_whitener, save_moments

__all__ = [
    "MomentAccumulator",
    "PairBatch",
    "PairShardMeta",
    "PhraseSpan",
    "Reconstructor",
    "ReconstructorHead",
    "Whitener",
    "assistant_token_mask",
    "load_pair_shards",
    "load_whitener",
    "sample_phrase_spans",
    "save_moments",
    "save_pair_shard",
    "split_of",
    "truncate_backbone",
    "whitened_cosine_loss",
]
