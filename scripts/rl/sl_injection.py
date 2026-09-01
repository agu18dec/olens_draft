"""Pure injection logic for the self-contained AO RL trainer.

PROVENANCE: copied from github.com/ceselder/skip-lens @ 13547ae, nla/injection.py
(`inject_at_marked_positions` + `marker_well_formed`; their `karvonen_inject_in_residual`
deliberately dropped — our contract is embedding REPLACEMENT with pre-transformed
vectors, audit doc §5 P1). This is the same-ancestry function our sglang rollout path
splices with (scripts/rl/ao_generate.py), so rollout ≡ training injection by code
identity. See docs/project/experiments/ola/skip_lens_audit.md.

The most correctness-critical path in the stack: if injection fails or hits the wrong
position, the model sees the literal marker character and the AO reads garbage. These
functions are the one place that must be right, so they're pure and unit-testable.
"""

import torch


def inject_at_marked_positions(
    input_ids: torch.Tensor,
    embeddings: torch.Tensor,
    vectors: torch.Tensor,
    inj_id: int,
    left_id: int,
    right_id: int,
    seq_slice: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Overwrite embedding rows at injection-marker positions with activation vectors.

    input_ids: [B, S] — the FULL token stream.
    embeddings: [B, S, d]. The embedding layer output. Cloned; original unchanged.
    vectors: [N, d] — activation vectors in microbatch order. N = number of
        injection sites expected. Must equal the count of valid matches found in
        the FULL input_ids.
    inj_id, left_id, right_id: the injection token + its canonical neighbors.
    seq_slice: (start, end) if embeddings holds only positions [start:end) of the
        sequence dim (unused in this trainer — kept for parity with the source).

    A match is valid iff input_ids[b, p] == inj_id AND input_ids[b, p-1] == left_id
    AND input_ids[b, p+1] == right_id. The neighbor check rejects false positives
    from the marker char appearing bare in response text.

    Raises:
        RuntimeError if count of valid matches != vectors.shape[0] — means prompt
        template drift, tokenizer version mismatch, or data corruption.
    """
    seq_len = input_ids.shape[-1]
    if seq_slice is None:
        start, end = 0, seq_len
        assert input_ids.shape == embeddings.shape[:-1], (
            f"input_ids {tuple(input_ids.shape)} and embeddings "
            f"{tuple(embeddings.shape[:-1])} batch dims must match"
        )
    else:
        start, end = seq_slice
        assert input_ids.shape[0] == embeddings.shape[0], (
            f"batch dim mismatch: input_ids {input_ids.shape[0]}, "
            f"embeddings {embeddings.shape[0]}"
        )
        assert embeddings.shape[1] == end - start, (
            f"seq_slice={seq_slice} spans {end - start} positions but "
            f"embeddings seq dim is {embeddings.shape[1]}."
        )
    assert vectors.ndim == 2 and vectors.shape[1] == embeddings.shape[-1], (
        f"vectors must be [N, d_model], got {tuple(vectors.shape)}, "
        f"d_model={embeddings.shape[-1]}"
    )
    out = embeddings.clone()
    vectors = vectors.to(out.device, out.dtype)
    matches = (input_ids == inj_id).nonzero()  # [M, 2] — (batch_idx, seq_idx), row-major sorted
    vec_idx = 0
    for b, p in matches.tolist():
        if p == 0 or p == seq_len - 1:
            continue
        if input_ids[b, p - 1] != left_id or input_ids[b, p + 1] != right_id:
            continue
        if start <= p < end and vec_idx < vectors.shape[0]:
            # vec_idx guard: surplus valid sites must reach the diagnostic
            # count check below, not die on a bare IndexError here.
            out[b, p - start] = vectors[vec_idx]
        vec_idx += 1
    expected = vectors.shape[0]
    if vec_idx != expected:
        msg = (
            f"found {vec_idx} injection sites with correct neighbors, expected {expected}. "
            f"Check prompt template drift, tokenizer version, or a response echoing the "
            f"full marker trigram past the stop token (should be impossible — bug 2 fix)."
        )
        raise RuntimeError(msg)
    return out


def count_valid_sites(token_ids, inj_id: int, left_id: int, right_id: int) -> int:
    """Number of NEIGHBOR-VALID marker sites — the hook's own criterion, exactly.

    The trainer's per-rollout pre-check is `count_valid_sites(full_ids) == 1`, so it
    agrees with `inject_at_marked_positions`'s count assert BY CONSTRUCTION (bug-2
    hardening). Deliberately NOT `marker_well_formed`: that counts RAW occurrences,
    so a marker-TERMINATED rollout (prompt slot + the trailing stop-token marker,
    which has no right neighbor) would be dropped — but a sampled stop-marker is a
    legitimate policy action, exactly like an eos, and stays in training.
    """
    n = len(token_ids)
    return sum(
        1
        for i, t in enumerate(token_ids)
        if t == inj_id
        and 0 < i < n - 1
        and token_ids[i - 1] == left_id
        and token_ids[i + 1] == right_id
    )


def marker_well_formed(token_ids, inj_id: int, left_id: int, right_id: int) -> bool:
    """True iff `token_ids` contains EXACTLY ONE `inj_id` marker with its canonical
    left/right neighbors — the same validity test `inject_at_marked_positions` applies
    per position.

    This is a MECHANISM check, not an output-text proxy like CJK fraction: it can't be
    eroded by RL shifting the model's output distribution. Cheap (pure token scan).

    Bug-2 hardening vs the skip-lens original: the trainer calls this over the FULL
    sequence (prompt + response), never a prompt-only slice, so the pre-check agrees
    with the injection hook's count assert by construction. A marker-terminated
    response cannot add a valid site: the marker is a stop token, so no right-neighbor
    can follow it inside the generated ids.
    """
    n = len(token_ids)
    positions = [i for i, t in enumerate(token_ids) if t == inj_id]
    if len(positions) != 1:
        return False
    p = positions[0]
    if p == 0 or p == n - 1:
        return False
    return token_ids[p - 1] == left_id and token_ids[p + 1] == right_id
