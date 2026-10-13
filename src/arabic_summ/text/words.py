"""Word counting used for every length rule (input 300-1000 words, output 10-30 %, 30-300 words).

A *word* is a whitespace-separated token containing at least one letter or digit, so
punctuation-only tokens ("-", "،", "...") are not counted while clitic-attached forms
("وقال", "بالمئة") count as one word, matching how Arabic word counts are usually reported.
"""

from __future__ import annotations


def words(text: str) -> list[str]:
    return [t for t in text.split() if any(ch.isalnum() for ch in t)]


def count_words(text: str) -> int:
    return len(words(text))
