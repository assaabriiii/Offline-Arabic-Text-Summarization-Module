"""Token-budgeted chunking at sentence / paragraph boundaries.

Proposal §11 step 3: inputs longer than the model's direct capacity are split at sentence or
paragraph boundaries into pieces of at most `max_tokens` (900) tokens with at most
`overlap_tokens` (80) tokens of overlap. Overlap is made of whole trailing sentences of the
previous chunk, so a chunk never starts mid-sentence. A single sentence longer than the
budget is split at commas, then at word boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

from .segment import Sentence

TokenCounter = Callable[[str], int]


def whitespace_tokens(text: str) -> int:
    return len(text.split())


@dataclass(frozen=True)
class Chunk:
    text: str
    sentence_ids: tuple[int, ...]   # indices into the sentence list (overlap included)
    n_tokens: int
    n_overlap_sentences: int


def _split_long(s: Sentence, count: TokenCounter, max_tokens: int) -> list[Sentence]:
    if count(s.text) <= max_tokens:
        return [s]
    parts = [p.strip() for p in re.split(r"(?<=[،,؛;])\s+", s.text) if p.strip()]
    if len(parts) == 1:
        parts = s.text.split()
    out, cur = [], ""
    for p in parts:
        cand = f"{cur} {p}".strip()
        if cur and count(cand) > max_tokens:
            out.append(cur)
            cur = p
        else:
            cur = cand
    if cur:
        out.append(cur)
    # a single word longer than the budget can't be split further; keep it as is
    return [Sentence(t, s.paragraph, s.index) for t in out]


def chunk_sentences(
    sentences: Sequence[Sentence],
    count: TokenCounter = whitespace_tokens,
    max_tokens: int = 900,
    overlap_tokens: int = 80,
    paragraph_break_min_fill: float = 0.6,
) -> list[Chunk]:
    if max_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("need 0 <= overlap_tokens < max_tokens")
    units: list[Sentence] = []
    for s in sentences:
        units.extend(_split_long(s, count, max_tokens))
    sizes = [count(u.text) for u in units]
    if sum(sizes) <= max_tokens:  # everything fits: never split
        return [Chunk(" ".join(u.text for u in units), tuple(u.index for u in units), sum(sizes), 0)] if units else []

    chunks: list[Chunk] = []
    start = 0
    n_overlap = 0
    while start < len(units):
        # drop the overlap if it leaves no room for the next new sentence
        first_new = start + n_overlap
        if n_overlap and first_new < len(units) and sum(sizes[start:first_new]) + sizes[first_new] > max_tokens:
            start, n_overlap = first_new, 0
        end, total = start, 0
        while end < len(units):
            # prefer to end a chunk at a paragraph boundary once it is reasonably full
            if (end > start and units[end].paragraph != units[end - 1].paragraph
                    and total >= paragraph_break_min_fill * max_tokens and end - start > n_overlap):
                break
            if total + sizes[end] > max_tokens and end > start:
                break
            total += sizes[end]
            end += 1
        text = " ".join(u.text for u in units[start:end])
        chunks.append(Chunk(text, tuple(u.index for u in units[start:end]), total, n_overlap))
        if end >= len(units):
            break
        # overlap: whole trailing sentences whose total size <= overlap_tokens
        back, ov = end, 0
        while back - 1 > start and ov + sizes[back - 1] <= overlap_tokens:
            back -= 1
            ov += sizes[back]
        n_overlap = end - back
        start = back if back > start else end
    return chunks
