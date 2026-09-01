"""Candidate-phrase filters for the r2sf filtered-selection round.

F1 (degenerate) heuristics — calibrated on the full pool64n32 candidate pool 2026-08-03
(3.62% flagged; manual review in ``docs/project/experiments/ola/r2sf_filter_plan.md``):
empty / short / low-alphanumeric / dominant-token repetition / char-4gram loops. No LLM.

Desc-band filter (r2sf-dp round, Camila 2026-08-04): drop candidates labeled CONT/JUNK by the
gpt-5.5 F3 judge, but only for rows whose activation layer lies in the workspace band
(L20-60). Labels come from a judge pass over the pool (``olens_r2sf_f3_judge.py``, jsonl rows
``{row, layer, phrases, labels}`` with labels per unique phrase TEXT); rows without labels
(non-band, or the judge's ~small malformed remainder) are kept unfiltered (fail-open).
"""

import collections
import json
from pathlib import Path

__all__ = [
    "DEFAULT_DESC_BAND",
    "DescLabels",
    "degenerate_flags",
    "is_degenerate",
    "load_desc_labels",
    "parse_band",
]

DEFAULT_DESC_BAND = (20, 60)


def parse_band(spec: str) -> tuple[int, int]:
    """Parse a ``"20-60"`` band spec into an inclusive (lo, hi) layer range."""
    lo, hi = (int(x) for x in spec.split("-"))
    if lo > hi:
        raise ValueError(f"band lo > hi in {spec!r}")
    return lo, hi


class DescLabels:
    """Per-row phrase→label lookup from an F3-judge jsonl, restricted to a layer band.

    ``keep(row, layer, phrase)`` is True unless the row is in-band AND the phrase is labeled
    something other than DESC. Unlabeled rows/phrases are kept (fail-open); ``stats`` counts
    masked/kept decisions for reporting.
    """

    def __init__(self, path: Path, band: tuple[int, int] = DEFAULT_DESC_BAND) -> None:
        self.band = band
        self.by_row: dict[int, dict[str, str]] = {}
        with Path(path).open() as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                labels = r.get("labels")
                if not isinstance(labels, list):
                    continue  # judge failure row -> fail-open
                if not (band[0] <= int(r["layer"]) <= band[1]):
                    continue
                self.by_row[int(r["row"])] = {
                    p: str(la) for p, la in zip(r["phrases"], labels, strict=False)
                }
        self.stats = {"masked": 0, "kept_labeled": 0, "kept_unlabeled": 0}

    def in_band(self, layer: int) -> bool:
        return self.band[0] <= int(layer) <= self.band[1]

    def keep(self, row: int, layer: int, phrase: str) -> bool:
        if not self.in_band(layer):
            return True
        lab = self.by_row.get(int(row), {}).get(phrase)
        if lab is None:
            self.stats["kept_unlabeled"] += 1
            return True
        if lab == "DESC":
            self.stats["kept_labeled"] += 1
            return True
        self.stats["masked"] += 1
        return False

    def labeled_bad(self, phrase: str) -> bool:
        """True iff the phrase text is labeled CONT/JUNK in ANY band row and never DESC.

        Dict-method (m1/m6) band-pass semantics: the cross-prompt dictionary drops atoms
        KNOWN to be continuation/junk; unlabeled atoms (from non-band rows) are kept.
        """
        verdicts = self._text_verdicts().get(phrase)
        return verdicts is not None and "DESC" not in verdicts

    def _text_verdicts(self) -> dict[str, set[str]]:
        cached = getattr(self, "_tv", None)
        if cached is None:
            cached = {}
            for labs in self.by_row.values():
                for p, la in labs.items():
                    cached.setdefault(p, set()).add(la)
            self._tv: dict[str, set[str]] = cached
        return cached


def load_desc_labels(path: str | Path, band: str = "20-60") -> DescLabels:
    """Load an F3-judge labels jsonl as a :class:`DescLabels` band filter."""
    return DescLabels(Path(path), parse_band(band))


def degenerate_flags(phrase: str) -> list[str]:
    """Return the list of F1 heuristic flags for a (already tag-stripped) phrase.

    Empty list = phrase passes F1. Flags: ``empty``, ``short`` (<8 non-whitespace chars),
    ``lowalpha`` (<30% alphanumeric), ``repeat`` (one whitespace token >=50% of >=4 tokens),
    ``charloop`` (a single char 4-gram covering >50% of a >=20-char string).
    """
    p = phrase.strip()
    if not p:
        return ["empty"]
    out: list[str] = []
    if len(p.replace(" ", "").replace("\n", "")) < 8:
        out.append("short")
    alnum = sum(c.isalnum() for c in p)
    if alnum / max(1, len(p)) < 0.30:
        out.append("lowalpha")
    toks = p.split()
    if len(toks) >= 4:
        top = collections.Counter(toks).most_common(1)[0][1]
        if top / len(toks) >= 0.5:
            out.append("repeat")
    if len(p) >= 20:
        grams = collections.Counter(p[i : i + 4] for i in range(len(p) - 3))
        _g, c = grams.most_common(1)[0]
        if 4 * c / len(p) > 0.5 and c >= 4:
            out.append("charloop")
    return out


def is_degenerate(phrase: str) -> bool:
    """True iff the phrase trips any F1 heuristic."""
    return bool(degenerate_flags(phrase))
