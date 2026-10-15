"""Aggregation of partial summaries: sentence dedup across segments and length enforcement."""

from __future__ import annotations

from typing import Sequence

from ..text.normalize import normalize_for_matching
from ..text.segment import split_sentences
from ..text.words import count_words


def _tokset(s: str) -> set[str]:
    return set(normalize_for_matching(s).split())


def dedup_sentences(sentences: Sequence[str], threshold: float = 0.7) -> list[str]:
    """Drop a sentence if most of its words are already covered by one kept sentence.

    Similarity = |A ∩ B| / min(|A|, |B|) (overlap coefficient), so a short sentence contained
    in a longer one is removed; on a tie the earlier (document-order) sentence is kept,
    except that a longer sentence replaces a shorter kept one it fully covers.
    """
    kept: list[tuple[str, set[str]]] = []
    for s in sentences:
        toks = _tokset(s)
        if not toks:
            continue
        dup = False
        for j, (ks, kt) in enumerate(kept):
            sim = len(toks & kt) / min(len(toks), len(kt))
            if sim >= threshold:
                if len(toks) > len(kt) and kt <= toks:
                    kept[j] = (s, toks)
                dup = True
                break
        if not dup:
            kept.append((s, toks))
    return [s for s, _ in kept]


def trim_to_words(sentences: Sequence[str], max_words: int) -> list[str]:
    """Keep whole sentences while within `max_words`; cut the last one at a word boundary only
    if nothing would be kept otherwise."""
    out, n = [], 0
    for s in sentences:
        w = count_words(s)
        if n + w <= max_words:
            out.append(s)
            n += w
        else:
            break
    if not out and sentences:
        out = [" ".join(sentences[0].split()[:max_words])]
    return out


def to_sentences(texts: Sequence[str]) -> list[str]:
    return [s.text for t in texts for s in split_sentences(t)]


def join(sentences: Sequence[str]) -> str:
    return " ".join(s.strip() for s in sentences if s.strip())
