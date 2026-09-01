"""Assistant-turn token masks over a rendered conversation.

``data._assistant_mask`` marks only the *final* assistant turn (its Jacobian use case).
The reconstructor samples phrases from *every* assistant turn, so this generalizes it: each
assistant message's token span is found the same way — render the conversation up to (but
excluding) that message with the generation prompt appended, then everything from there to the
end of the message's own render is assistant-generated.
"""

from oracle_lens.data import ChatTokenizerLike, Conversation, _render_ids


def assistant_token_mask(
    tokenizer: ChatTokenizerLike, conversation: Conversation, t_max: int
) -> list[bool]:
    """Boolean mask over the first ``t_max`` rendered tokens: True on ANY assistant turn.

    For assistant message ``i``: ``start`` = len(render(conversation[:i]) + generation prompt),
    ``end`` = len(render(conversation[:i+1])). Tokens in ``[start, end)`` are the assistant's
    content plus its end-of-turn marker (the same span convention as ``data._assistant_mask``,
    which this must agree with on single-assistant-turn conversations — unit-tested).
    """
    full = _render_ids(tokenizer, conversation, add_generation_prompt=False)
    length = min(t_max, len(full))
    mask = [False] * length
    for i, message in enumerate(conversation):
        if message["role"] != "assistant":
            continue
        prefix = _render_ids(tokenizer, conversation[:i], add_generation_prompt=True)
        upto = _render_ids(tokenizer, conversation[: i + 1], add_generation_prompt=False)
        for pos in range(min(len(prefix), length), min(len(upto), length)):
            mask[pos] = True
    return mask
