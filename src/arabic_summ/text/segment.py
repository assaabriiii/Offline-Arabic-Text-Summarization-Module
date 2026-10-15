"""Paragraph and sentence segmentation for Modern Standard Arabic.

Rules (deterministic, documented for reproducibility):
* Paragraphs are separated by blank lines; single newlines are also hard sentence breaks
  (headings, list items and bylines are usually on their own line).
* A sentence ends at `.`, `!`, `?`, `؟`, `…` (one or more), optionally followed by closing
  quotes/brackets, when followed by whitespace. `؛` and `،` never end a sentence.
* No split after a single-letter abbreviation (د. م. أ. ص.), after a bare list number
  ("1. ", "٢. "), or inside numbers (3.5, ٣٫٥) and URLs (no whitespace there anyway).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_END = r"[.!?؟…]+"
_CLOSERS = "\"'»”’)]}"
_BOUNDARY_RE = re.compile(rf"({_END}[{re.escape(_CLOSERS)}]*)(\s+)")
_ABBREV_RE = re.compile(r"(?:^|\s)[ء-ي]$")        # single Arabic letter before the dot
_LIST_NUM_RE = re.compile(r"(?:^|\s)[0-9٠-٩۰-۹]{1,3}$")


@dataclass(frozen=True)
class Sentence:
    text: str
    paragraph: int
    index: int


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _split_line(line: str) -> list[str]:
    out, start = [], 0
    for m in _BOUNDARY_RE.finditer(line):
        head = line[start : m.start(1)]
        if m.group(1) == "." and (_ABBREV_RE.search(head) or _LIST_NUM_RE.search(head) and len(head.strip()) <= 3):
            continue
        piece = line[start : m.end(1)].strip()
        if piece:
            out.append(piece)
        start = m.end()
    tail = line[start:].strip()
    if tail:
        out.append(tail)
    return out


def split_sentences(text: str) -> list[Sentence]:
    sents: list[Sentence] = []
    for p_idx, para in enumerate(split_paragraphs(text)):
        for line in para.split("\n"):
            for s in _split_line(line.strip()):
                if any(ch.isalnum() for ch in s):
                    sents.append(Sentence(s, p_idx, len(sents)))
    return sents
