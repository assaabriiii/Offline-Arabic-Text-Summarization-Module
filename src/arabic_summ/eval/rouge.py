"""ROUGE-1 / ROUGE-2 / ROUGE-L F1 for Arabic.

`rouge_score`'s default tokenizer keeps only [a-z0-9], so it silently scores Arabic as empty.
We tokenize on whitespace after the *matching* normalization policy (alef / yeh / teh-marbuta
folding, ASCII digits, no punctuation, no diacritics). The primary score has no stemming
(decision 17). `stem="snowball"` gives the secondary, XL-Sum-style score: the XL-Sum authors'
multilingual scorer stems Arabic with NLTK's Snowball stemmer, so clitic and inflected forms
("والمنظمة" / "المنظمة" / "منظمات") match. As in that scorer, words are stemmed in their own spelling
(punctuation removed, digits ASCII); our matching folds (alef, ى, ة) are applied to the stems
afterwards, so the score is close to, not identical with, the XL-Sum scorer. Scores are in [0, 1]; corpus scores are the mean
of per-pair F1 (as in rouge_score / XL-Sum reporting).
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Callable, Sequence

from ..text.normalize import NormalizationPolicy, normalize, normalize_for_matching

STEMMERS = ("none", "snowball")
# no orthographic folding before stemming: the Snowball rules expect ة, أ/إ, ى as written
_STEM_POLICY = NormalizationPolicy(digits="ascii", remove_punctuation=True, lowercase_latin=True)


@lru_cache(maxsize=None)
def _snowball() -> Callable[[str], str]:
    try:
        from nltk.stem import SnowballStemmer
    except ImportError as e:
        raise ImportError("stem='snowball' needs nltk (pip install nltk==3.9.1)") from e
    return lru_cache(maxsize=500_000)(SnowballStemmer("arabic").stem)


def tokenize(text: str, stem: str = "none") -> list[str]:
    toks = normalize_for_matching(text).split()
    if stem == "none":
        return toks
    if stem == "snowball":
        f = _snowball()
        out = []
        for t in normalize(text, _STEM_POLICY).split():
            out.extend(normalize_for_matching(f(t)).split())
        return out
    raise ValueError(f"stem must be one of {STEMMERS}")


def _ngrams(toks: Sequence[str], n: int) -> Counter:
    return Counter(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))


def _f1(overlap: int, n_hyp: int, n_ref: int) -> float:
    if overlap == 0 or n_hyp == 0 or n_ref == 0:
        return 0.0
    p, r = overlap / n_hyp, overlap / n_ref
    return 2 * p * r / (p + r)


def rouge_n(ref: Sequence[str], hyp: Sequence[str], n: int) -> float:
    r, h = _ngrams(ref, n), _ngrams(hyp, n)
    return _f1(sum((r & h).values()), sum(h.values()), sum(r.values()))


def lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(ref: Sequence[str], hyp: Sequence[str]) -> float:
    return _f1(lcs_len(ref, hyp), len(hyp), len(ref))


def score(reference: str, hypothesis: str, stem: str = "none") -> dict[str, float]:
    r, h = tokenize(reference, stem), tokenize(hypothesis, stem)
    return {"rouge1": rouge_n(r, h, 1), "rouge2": rouge_n(r, h, 2), "rougeL": rouge_l(r, h)}


def corpus(references: Sequence[str], hypotheses: Sequence[str], stem: str = "none") -> dict[str, float]:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses differ in length")
    rows = [score(r, h, stem) for r, h in zip(references, hypotheses)]
    n = max(1, len(rows))
    return {k: sum(x[k] for x in rows) / n for k in ("rouge1", "rouge2", "rougeL")}
