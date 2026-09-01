"""assistant_token_mask: agreement with data._assistant_mask, multi-turn coverage."""

from typing import Any

from oracle_lens.core.masks import assistant_token_mask
from oracle_lens.data import _assistant_mask


class _StubChatTokenizer:
    """Same stub as tests/test_data.py: role-header sentinels, one token per word."""

    USER_HDR, ASST_HDR, EOS = 101, 102, 103

    def apply_chat_template(
        self, conversation: list[dict[str, str]], *, add_generation_prompt: bool = False, **_: Any
    ) -> list[int]:
        ids: list[int] = []
        for msg in conversation:
            content = [len(w) for w in msg["content"].split()]
            if msg["role"] == "user":
                ids += [self.USER_HDR, *content]
            else:
                ids += [self.ASST_HDR, *content, self.EOS]
        if add_generation_prompt:
            ids.append(self.ASST_HDR)
        return ids


def test_agrees_with_final_turn_mask_on_single_assistant_turn() -> None:
    conv = [
        {"role": "user", "content": "one two three"},
        {"role": "assistant", "content": "four five six seven"},
    ]
    tok = _StubChatTokenizer()
    for t_max in (4, 7, 100):
        assert assistant_token_mask(tok, conv, t_max) == _assistant_mask(tok, conv, t_max)


def test_marks_all_assistant_turns() -> None:
    conv = [
        {"role": "user", "content": "q one"},
        {"role": "assistant", "content": "a one"},
        {"role": "user", "content": "q two"},
        {"role": "assistant", "content": "a two two"},
    ]
    tok = _StubChatTokenizer()
    mask = assistant_token_mask(tok, conv, 100)
    rendered = tok.apply_chat_template(conv)
    assert len(mask) == len(rendered)
    # Assistant content + EOS marked; user content and headers not.
    for pos, marked in enumerate(mask):
        token = rendered[pos]
        if token == tok.USER_HDR or token == tok.ASST_HDR:
            assert not marked
    # rendered: [U,q,one | A,a,one,EOS | U,q,two | A,a,two,two,EOS] -> spans [4,7) and [11,15)
    # (the ASST_HDR positions 3 and 10 are headers, not assistant content)
    assert all(mask[4:7]) and all(mask[11:15])
    assert not any(mask[0:4]) and not any(mask[7:11])


def test_truncation() -> None:
    conv = [
        {"role": "user", "content": "one two"},
        {"role": "assistant", "content": "three four"},
    ]
    tok = _StubChatTokenizer()
    mask = assistant_token_mask(tok, conv, 4)
    assert len(mask) == 4
    # rendered[:4] = [USER_HDR, 'one', 'two', ASST_HDR]; the header is not assistant content,
    # so nothing is marked — and this must agree with data._assistant_mask exactly.
    assert mask == [False, False, False, False]
    assert mask == _assistant_mask(tok, conv, 4)
