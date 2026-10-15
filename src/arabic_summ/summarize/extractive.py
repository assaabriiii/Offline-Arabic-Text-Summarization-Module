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
           position_weight: float = 0.3, max_words: int | None = None) -> list[int]:
    """Indices (document order) of top-scored sentences up to `budget_words`.

    Without `max_words` (hybrid pre-selection) the budget is a soft cap and at least one sentence is
    returned. With `max_words` (extractive baseline) sentences that would exceed the hard maximum are
    skipped and selection continues until the budget (target) is reached."""
    scores = textrank_scores(sentences, position_weight=position_weight)
    return _fill(list(np.argsort(-scores, kind="stable")), word_counts, budget_words, max_words, False)


def lead(word_counts: Sequence[int], budget_words: int, max_words: int | None = None) -> list[int]:
    """First sentences up to the budget; with `max_words`, a too-long sentence is skipped instead
    of ending the summary early (so short leads don't fall below the minimum length)."""
    return _fill(list(range(len(word_counts))), word_counts, budget_words, max_words, True)


def _fill(order: Sequence[int], word_counts: Sequence[int], budget: int, max_words: int | None,
          stop_on_overflow: bool) -> list[int]:
    """Add sentences in `order` until the budget is reached. The cap is `max_words` if given (an
    overflowing sentence is skipped), else the budget itself (the first sentence is always kept;
    `stop_on_overflow` ends at the first overflowing sentence, as a plain lead does)."""
    chosen, total = [], 0
    cap = budget if max_words is None else max_words
    for i in order:
        if total >= budget:
            break
        w = word_counts[i]
        if total + w <= cap or (max_words is None and not chosen):
            chosen.append(int(i))
            total += w
        elif max_words is None and stop_on_overflow:
            break
    return sorted(chosen)
