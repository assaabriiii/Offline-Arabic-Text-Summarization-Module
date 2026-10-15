"""Lexical fact checker: are the summary's numbers and content words supported by the source?

Used three ways: (1) to filter XL-Sum training pairs whose highlight adds facts not in the
article, (2) as a model/decoding selection signal, (3) as the automatic part of the factual
accuracy check in 05 (names and numbers must be copied exactly, proposal §7).

It is deliberately conservative and explainable: a summary *number* is supported only if the
same digit string occurs in the source; a summary *content word* is supported if its light stem
(clitic prefixes and common suffixes removed, after matching normalization) occurs among the
source's stems. It cannot detect wrong relations between supported words (e.g. swapped roles);
that remains the human evaluation's job.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

from ..text.normalize import normalize_for_matching

# after matching normalization: alef folded, ة->ه, ى->ي
PREFIXES = ("وبال", "وكال", "وال", "فال", "بال", "كال", "لل", "ال", "و", "ف", "ب", "ل", "ك", "س")
SUFFIXES = ("هما", "كما", "ات", "ون", "ين", "ان", "ها", "هم", "هن", "نا", "كم", "ه", "ي")
STOPWORDS = frozenset(normalize_for_matching(w) for w in """
في من على الى إلى عن مع هذا هذه ذلك تلك التي الذي الذين ان أن إن او أو ثم قد لقد كان كانت يكون
تكون كما بعد قبل بين حتى عند منذ خلال حول ضد لدى غير كل بعض اي أي ما ماذا لا لم لن ليس هو هي هم
هن نحن انت أنت وقال قال قالت وقالت ويقول تقول يقول اضاف أضاف واضاف وأضاف ايضا أيضا اذ إذ اذا إذا
فقط فيما بينما وفق وفقا نحو اكثر أكثر اقل أقل عدد عام يوم الذى التى لها له لهم عليه عليها فيه فيها
منه منها به بها انه أنه انها أنها الا إلا وهو وهي كذلك هناك هنا حيث تم يتم جرى
""".split())
_NUM = re.compile(r"\d+")


def stem(tok: str) -> str:
    for p in PREFIXES:
        if tok.startswith(p) and len(tok) - len(p) >= 3:
            tok = tok[len(p):]
            break
    for s in SUFFIXES:
        if tok.endswith(s) and len(tok) - len(s) >= 3:
            tok = tok[: -len(s)]
            break
    return tok


def _tokens(text: str) -> list[str]:
    return normalize_for_matching(text).split()


def numbers(text: str) -> list[str]:
    """Digit strings (any script) as ASCII; separators are dropped by the matching policy."""
    return [m.lstrip("0") or "0" for t in _tokens(text) for m in _NUM.findall(t)]


def content_stems(text: str) -> list[str]:
    out = []
    for t in _tokens(text):
        if t in STOPWORDS or _NUM.search(t) or len(t) < 3:
            continue
        s = stem(t)
        if s not in STOPWORDS:
            out.append(s)
    return out


@dataclass
class FactReport:
    n_numbers: int
    unsupported_numbers: list[str] = field(default_factory=list)
    n_content: int = 0
    novel_content: list[str] = field(default_factory=list)

    @property
    def number_precision(self) -> float | None:
        return None if not self.n_numbers else 1 - len(self.unsupported_numbers) / self.n_numbers

    @property
    def novel_rate(self) -> float:
        return len(self.novel_content) / self.n_content if self.n_content else 0.0

    @property
    def numbers_ok(self) -> bool:
        return not self.unsupported_numbers

    def to_dict(self) -> dict:
        return {**asdict(self), "number_precision": self.number_precision, "novel_rate": self.novel_rate,
                "numbers_ok": self.numbers_ok}


class SourceIndex:
    """Precomputed source numbers/stems, so one source can be checked against many summaries."""

    def __init__(self, source: str):
        self.numbers = set(numbers(source))
        toks = _tokens(source)
        self.stems = {stem(t) for t in toks} | set(toks)

    def check(self, summary: str) -> FactReport:
        nums = numbers(summary)
        stems = content_stems(summary)
        return FactReport(
            n_numbers=len(nums),
            unsupported_numbers=[n for n in nums if n not in self.numbers],
            n_content=len(stems),
            novel_content=[s for s in stems if s not in self.stems],
        )


def check(summary: str, source: str) -> FactReport:
    return SourceIndex(source).check(summary)


def aggregate(reports: Iterable[FactReport]) -> dict:
    reps = list(reports)
    with_nums = [r for r in reps if r.n_numbers]
    n = max(1, len(reps))
    return {
        "n": len(reps),
        # mean over summaries that contain numbers; 1.0 when no summary has numbers
        "number_precision": (sum(r.number_precision for r in with_nums) / len(with_nums)) if with_nums else 1.0,
        "numbers_ok_share": sum(r.numbers_ok for r in reps) / n,
        "novel_rate": sum(r.novel_rate for r in reps) / n,
    }


def check_many(summaries: Sequence[str], sources: Sequence[str]) -> list[FactReport]:
    return [check(s, src) for s, src in zip(summaries, sources)]
