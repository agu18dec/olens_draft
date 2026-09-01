"""Prompt batches for Jacobian averaging (jlens_construction.md, section 4).

Take the first N corpus documents that are at least ``t_max`` tokens long, truncate each
to exactly ``t_max`` (fixed-length batches keep the position-count bookkeeping trivial and need no
padding/attention-mask handling), and group them into batches. Deterministic and
resumable: a batch reports how many corpus docs have been consumed so far, and a resumed
run skips exactly that many.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from jaxtyping import Bool, Int
from torch import Tensor

Conversation = list[dict[str, str]]  # [{"role": "user"|"assistant"|..., "content": str}, ...]


class TokenizerLike(Protocol):
    """The slice of a HF tokenizer the corpus iterator needs."""

    def encode(self, text: str) -> list[int]: ...


class ChatTokenizerLike(Protocol):
    """A tokenizer that can render chat conversations (Q5 chat corpora).

    ``apply_chat_template(tokenize=True)`` returns token ids, but the concrete return type varies by
    transformers version — a flat ``list[int]``, a batched ``[[int]]``, or a ``BatchEncoding`` with
    an ``input_ids`` field. :func:`_render_ids` normalizes all of these to a flat ``list[int]``.
    """

    def apply_chat_template(self, conversation: Conversation, **kwargs: Any) -> Any: ...


def _render_ids(
    tokenizer: ChatTokenizerLike, conversation: Conversation, *, add_generation_prompt: bool
) -> list[int]:
    """Render a conversation to a flat ``list[int]``, normalizing across transformers versions."""
    out = tokenizer.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=add_generation_prompt
    )
    ids = out["input_ids"] if hasattr(out, "keys") else out  # BatchEncoding -> input_ids
    if len(ids) and isinstance(ids[0], (list, tuple)):  # batched [[...]] -> first row
        ids = ids[0]
    return [int(t) for t in ids]


@dataclass(frozen=True)
class PromptBatch:
    """A fixed-length batch plus the corpus cursor after it (for checkpoint/resume).

    ``source_mask`` (Q5 ``chat_assistant``): True at positions kept as Jacobian sources
    (assistant-turn tokens); None means every position is a source (pretrain / chat_full).
    """

    input_ids: Int[Tensor, "batch pos"]
    docs_consumed: int
    source_mask: Bool[Tensor, "batch pos"] | None = None


def iter_prompt_batches(
    texts: Iterable[str],
    tokenizer: TokenizerLike,
    *,
    t_max: int,
    n_prompts: int,
    batch_size: int = 1,
    skip_docs: int = 0,
) -> Iterator[PromptBatch]:
    """Yield up to ``n_prompts`` prompts as fixed-length batches of ``t_max`` tokens.

    Documents shorter than ``t_max`` tokens are skipped (but still advance the corpus
    cursor). ``skip_docs`` fast-forwards past already-consumed documents on resume.
    """
    if t_max < 1 or n_prompts < 1 or batch_size < 1:
        raise ValueError("t_max, n_prompts, and batch_size must all be >= 1")
    docs_consumed = skip_docs
    yielded = 0
    pending: list[list[int]] = []
    iterator = iter(texts)
    for _ in range(skip_docs):
        next(iterator)
    for text in iterator:
        docs_consumed += 1
        ids = tokenizer.encode(text)
        if len(ids) < t_max:
            continue
        pending.append(ids[:t_max])
        yielded += 1
        if len(pending) == batch_size or yielded == n_prompts:
            yield PromptBatch(
                input_ids=torch.tensor(pending, dtype=torch.long),
                docs_consumed=docs_consumed,
            )
            pending = []
        if yielded == n_prompts:
            return
    if pending:  # corpus ran out mid-batch
        yield PromptBatch(
            input_ids=torch.tensor(pending, dtype=torch.long), docs_consumed=docs_consumed
        )


def load_corpus_texts(dataset_id: str = "NeelNanda/pile-10k") -> Iterator[str]:
    """Stream document texts from a HF dataset (train split, ``text`` column).

    A config name rides after a colon: ``Salesforce/wikitext:wikitext-103-raw-v1``.
    Configless datasets (e.g. pile-10k) are just the repo id, unchanged.
    """
    from datasets import load_dataset  # heavy import, deferred so unit tests never pay it

    name, _, config = dataset_id.partition(":")
    dataset = load_dataset(name, config or None, split="train")
    for row in dataset:
        yield str(row["text"])


def load_corpus_with_meta(
    dataset_id: str = "NeelNanda/pile-10k",
) -> Iterator[tuple[str, str]]:
    """Stream ``(text, source)`` where ``source`` is the Pile subset (``meta.pile_set_name``).

    Used by the drummer #2 characterization to attribute top-activating documents to a Pile
    subset (Pile-CC, GitHub, ArXiv, …). ``source`` is ``""`` for corpora without that metadata.
    """
    from datasets import load_dataset  # heavy import, deferred so unit tests never pay it

    name, _, config = dataset_id.partition(":")
    dataset = load_dataset(name, config or None, split="train")
    for row in dataset:
        meta = row.get("meta") if hasattr(row, "get") else None
        source = ""
        if isinstance(meta, dict) and "pile_set_name" in meta:
            source = str(meta["pile_set_name"])
        yield str(row["text"]), source


def _alpaca_to_messages(row: dict[str, Any]) -> Conversation:
    """Map an Alpaca row {instruction, input, output} to a user+assistant conversation."""
    instr = str(row["instruction"])
    inp = str(row.get("input") or "")
    user = instr if not inp else f"{instr}\n\n{inp}"
    return [{"role": "user", "content": user}, {"role": "assistant", "content": str(row["output"])}]


def load_chat_messages(dataset_id: str = "tatsu-lab/alpaca") -> Iterator[Conversation]:
    """Stream conversations from a HF chat dataset (train split).

    Supports a ``messages``/``conversation`` column (list of {role, content}) or the Alpaca
    {instruction, input, output} schema (mapped to one user + one assistant turn). Config name
    rides after a colon, as in :func:`load_corpus_texts`.
    """
    from datasets import load_dataset

    name, _, config = dataset_id.partition(":")
    dataset = load_dataset(name, config or None, split="train")
    for row in dataset:
        if row.get("messages"):
            yield list(row["messages"])
        elif row.get("conversation"):
            yield list(row["conversation"])
        elif "instruction" in row and "output" in row:
            yield _alpaca_to_messages(row)
        else:
            raise ValueError(f"{dataset_id}: row has no messages/conversation/instruction column")


def _assistant_mask(
    tokenizer: ChatTokenizerLike, conversation: Conversation, t_max: int
) -> list[bool]:
    """Boolean mask over the first ``t_max`` rendered tokens: True on the final assistant turn.

    Found by rendering the conversation up to (but excluding) the final assistant message with the
    generation prompt appended, then marking everything after that prefix as assistant-generated.
    """
    full = _render_ids(tokenizer, conversation, add_generation_prompt=False)
    prefix = _render_ids(tokenizer, conversation[:-1], add_generation_prompt=True)
    start = len(prefix)
    return [start <= i < len(full) for i in range(min(t_max, len(full)))]


def iter_chat_batches(
    conversations: Iterable[Conversation],
    tokenizer: ChatTokenizerLike,
    *,
    t_max: int,
    n_prompts: int,
    mode: str,  # "chat_full" | "chat_assistant"
    skip_docs: int = 0,
) -> Iterator[PromptBatch]:
    """Yield up to ``n_prompts`` rendered chat conversations as fixed-length ``t_max`` batches.

    One conversation per batch (``batch_size=1``). Conversations rendering to < ``t_max`` tokens are
    skipped. ``chat_assistant`` attaches a ``source_mask`` over assistant-turn tokens; conversations
    with no assistant tokens left after truncation are skipped (they would contribute no sources).
    """
    if mode not in ("chat_full", "chat_assistant"):
        raise ValueError(f"mode must be chat_full|chat_assistant, got {mode!r}")
    docs_consumed = skip_docs
    yielded = 0
    iterator = iter(conversations)
    for _ in range(skip_docs):
        next(iterator)
    for conversation in iterator:
        docs_consumed += 1
        ids = _render_ids(tokenizer, conversation, add_generation_prompt=False)
        if len(ids) < t_max:
            continue
        mask = None
        if mode == "chat_assistant":
            kept = _assistant_mask(tokenizer, conversation, t_max)
            if not any(kept):  # truncation left no assistant tokens — no sources to average
                continue
            mask = torch.tensor([kept], dtype=torch.bool)
        yielded += 1
        yield PromptBatch(
            input_ids=torch.tensor([ids[:t_max]], dtype=torch.long),
            docs_consumed=docs_consumed,
            source_mask=mask,
        )
        if yielded == n_prompts:
            return
