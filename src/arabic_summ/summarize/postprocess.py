"""Aggregation of partial summaries: sentence dedup across segments and length enforcement."""

from __future__ import annotations

import re
from typing import Sequence

from ..text.normalize import normalize_for_matching
from ..text.segment import split_sentences
from ..text.words import count_words

# sentence-final punctuation, optionally followed by closing quotes / brackets
_SENTENCE_END = re.compile(r"[.!?؟…][\"'»”)\]]*\s*$")
_LAST_END = re.compile(r"^(.*[.!?؟…][\"'»”)\]]*)", re.S)


def _tokset(s: str) -> set[str]:
    return set(normalize_for_matching(s).split())


def drop_unfinished(text: str) -> str:
    """Remove a trailing fragment that `max_new_tokens` cut off mid-sentence (05, 2026-10-18: 16 of
    240 summaries ended mid-sentence). Keeps everything up to the last sentence-final punctuation; an
    output without any complete sentence is returned unchanged (the length fill handles it)."""
    text = text.strip()
    if not text or _SENTENCE_END.search(text):
        return text
    m = _LAST_END.match(text)
    return m.group(1).strip() if m else text


def is_complete(sentence: str) -> bool:
    """True if the sentence ends in sentence-final punctuation (not a heading or a cut-off fragment)."""
    return bool(_SENTENCE_END.search(sentence.strip()))


def covered(sentence: str, kept: Sequence[str], threshold: float = 0.7) -> bool:
    """True if most words of `sentence` already occur in one of `kept` (same measure as dedup)."""
    toks = _tokset(sentence)
    if not toks:
        return True
    for k in kept:
        kt = _tokset(k)
        if kt and len(toks & kt) / min(len(toks), len(kt)) >= threshold:
            return True
    return False


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


def trim_to_words(sentences: Sequence[str], max_words: int, min_words: int = 0) -> list[str]:
    """Keep whole sentences while within `max_words`. If that leaves fewer than `min_words` and
    another sentence follows, append its first words (word boundary) to reach the minimum; if
    nothing would be kept at all, cut the first sentence at `max_words`."""
    out, n, rest = [], 0, []
    for k, s in enumerate(sentences):
        w = count_words(s)
        if n + w <= max_words:
            out.append(s)
            n += w
        else:
            rest = list(sentences[k:])
            break
    if not out and sentences:
        return [" ".join(sentences[0].split()[:max_words])]
    if n < min_words and rest:
        need = min(min_words, max_words) - n
        toks = rest[0].split()
        take, got = [], 0
        for t in toks:
            if got >= need:
                break
            take.append(t)
            got += 1 if any(ch.isalnum() for ch in t) else 0
        out.append(" ".join(take))
    return out


def to_sentences(texts: Sequence[str]) -> list[str]:
    return [s.text for t in texts for s in split_sentences(t)]


def join(sentences: Sequence[str]) -> str:
    return " ".join(s.strip() for s in sentences if s.strip())
