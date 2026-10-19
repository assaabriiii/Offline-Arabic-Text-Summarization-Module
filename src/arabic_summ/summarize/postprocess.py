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


def is_question(sentence: str) -> bool:
    """True for a sentence ending in ? / ؟ (rhetorical questions of feature articles; 05 review)."""
    return bool(re.search(r"[?؟][\"'»”)\]]*\s*$", sentence.strip()))


_SAYS = re.compile(r"(?:^|\s)[وف]?(?:قال|قالت|قالوا|يقول|تقول|[أا]ضاف|[أا]ضافت|[أا]ضافوا|يضيف|تضيف|[أا]كد|[أا]كدت|يؤكد|"
                   r"تؤكد|ذكر|ذكرت|[أا]وضح|[أا]وضحت|يوضح|توضح|[أا]شار|[أا]شارت|صرح|صرحت|حذر|حذرت|يشرح|تشرح)(?=[\s:،,]|$)")
_QUOTE_PAIRS = (('"', '"'), ("«", "»"), ("“", "”"))
_FINAL = re.compile(r"([.!?؟…]*)\s*$")


def fix_quotes(sentence: str) -> str | None:
    """Sentence splitting cuts multi-sentence quotes into pieces. A piece whose quote a speaker opens
    ('قالت عائشة: "ما يقلقني ...') gets its closing mark, one a speaker closes ('... نرتديها"، هكذا
    قالت ليلى') its opening mark; a piece without a speaker ('" لم يكن شيء ليهيئني ...', first person
    out of context) returns None. Balanced sentences are returned unchanged (05 review, PROGRESS 37)."""
    s = sentence.strip()
    for op, cl in _QUOTE_PAIRS:
        if op == cl:
            if s.count(op) % 2 == 0:
                continue
            k = s.index(op)
            opening = bool(_SAYS.search(s[:k]) or s[:k].rstrip().endswith(":"))
        else:
            if s.count(op) == s.count(cl):
                continue
            opening, k = s.count(op) > s.count(cl), (s.index(op) if op in s else s.index(cl))
            if opening and not (_SAYS.search(s[:k]) or s[:k].rstrip().endswith(":")):
                return None
        if opening and (op != cl or k > 0):
            end = _FINAL.search(s)
            if s[:end.start()].rstrip().endswith(cl):     # '"... قبل."الآن ... الجميع".': malformed
                return None
            s = s[:end.start()].rstrip() + cl + end.group(1)
        elif _SAYS.search(s[s.rindex(cl) + 1:]):
            s = op + s.lstrip(" " + op)
        else:
            return None
    return s


# "جنيف/بورت سودان، 30 كانون الثاني 2026 - ", "القاهرة (رويترز) - ", "غزة، الأرض الفلسطينية المحتلة - ",
# "جنيف – ": a short lead ending in a dash that has a number, bracket, comma or slash, or of <= 2 words
# before an en/em dash; '"مهمة مجتمع" ': a quoted crosshead of <= 4 words glued to the next sentence
_DATELINE = re.compile(r"^([^.!?؟\"«»\n]{2,80}?)\s+([-–—])\s+(?=[^\s\d٠-٩])")
_CROSSHEAD = re.compile(r"^([\"«“])[^\"«»“”]{1,40}[\"»”]\s+(?=[^\s.!?؟،,:])")


def strip_lead(sentence: str, dateline: bool = True) -> str:
    """Remove a dateline (`dateline`: only for the first sentences of an article; "كوفيد - 19" or
    "التقزم – وهو ..." later on are not datelines) or a quoted crosshead from the start of a source
    sentence (05 review: the fill added "1 تشرين الأول/ أكتوبر 2026 - إن ..." and '"مهمة مجتمع" وعوضا ...')."""
    s = sentence.strip()
    m = _DATELINE.match(s) if dateline else None
    if m and len(m.group(1).split()) <= 8 and not _SAYS.search(m.group(1)) and (re.search(r"[\d٠-٩(\[،,/]", m.group(1))
                                               or m.group(2) != "-" and len(m.group(1).split()) <= 2):
        s = s[m.end():]
    m = _CROSSHEAD.match(s)
    if m and len(s[:m.end()].split()) <= 4:
        s = s[m.end():]
    return s


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
