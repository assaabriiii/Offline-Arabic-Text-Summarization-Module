"""Extractive sentence scoring (TextRank over TF-IDF cosine + lead prior).

Used by the extractive baseline and by the hybrid pipeline's selection stage. Pure numpy,
deterministic, language-agnostic once text is matching-normalized.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import numpy as np

from ..text.normalize import normalize_for_matching

# Small closed list of very frequent MSA function words (matching-normalized forms).
STOPWORDS = set("""
في من علي الي عن ان او ما لا لم لن قد هذا هذه ذلك تلك التي الذي الذين ثم كما كان كانت
يكون مع بين بعد قبل حتي عند كل اي هو هي هم انه انها وقد وفي ومن وان وهو وهي الا اذا لكن
بل غير منذ خلال حول ضد تم كذلك ايضا فقط اكثر اقل بعض جميع عام مثل حيث وكان وقال قال
""".split())


def _tokens(s: str) -> list[str]:
    return [t for t in normalize_for_matching(s).split() if t not in STOPWORDS and len(t) > 1]


def textrank_scores(sentences: Sequence[str], damping: float = 0.85, iters: int = 50,
                    position_weight: float = 0.3) -> np.ndarray:
    n = len(sentences)
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)
    toks = [_tokens(s) for s in sentences]
    df = Counter(t for ts in toks for t in set(ts))
    vocab = {t: i for i, t in enumerate(df)}
    m = np.zeros((n, len(vocab)))
    for i, ts in enumerate(toks):
        for t, c in Counter(ts).items():
            m[i, vocab[t]] = c * math.log((1 + n) / (1 + df[t]) + 1)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    m = np.divide(m, norms, out=np.zeros_like(m), where=norms > 0)
    sim = m @ m.T
    np.fill_diagonal(sim, 0.0)
    row = sim.sum(axis=1, keepdims=True)
    trans = np.divide(sim, row, out=np.full_like(sim, 1.0 / n), where=row > 0)
    r = np.full(n, 1.0 / n)
    for _ in range(iters):
        r = (1 - damping) / n + damping * trans.T @ r
    r = r / r.max()
    lead = 1.0 / np.sqrt(np.arange(1, n + 1))  # news/report lead bias
    return (1 - position_weight) * r + position_weight * lead


def select(sentences: Sequence[str], word_counts: Sequence[int], budget_words: int,
           position_weight: float = 0.3) -> list[int]:
    """Indices (document order) of top-scored sentences fitting `budget_words`.
    Always returns at least one sentence."""
    scores = textrank_scores(sentences, position_weight=position_weight)
    chosen, total = [], 0
    for i in np.argsort(-scores, kind="stable"):
        if total + word_counts[i] <= budget_words or not chosen:
            chosen.append(int(i))
            total += word_counts[i]
        if total >= budget_words:
            break
    return sorted(chosen)


def lead(word_counts: Sequence[int], budget_words: int) -> list[int]:
    chosen, total = [], 0
    for i, w in enumerate(word_counts):
        if total + w > budget_words and chosen:
            break
        chosen.append(i)
        total += w
    return chosen
