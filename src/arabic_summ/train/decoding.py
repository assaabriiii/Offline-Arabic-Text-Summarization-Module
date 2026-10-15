"""Decoding-parameter tuning on XL-Sum validation (proposal: beams 1-4, length penalty 1-2,
no_repeat_ngram_size 3), selected with the fact checker; result saved next to the model."""

from __future__ import annotations

import itertools
import time
from typing import Mapping, Sequence

from ..backends.base import GenerationRequest
from ..eval import facts, rouge
from ..text.words import count_words
from .select import select_best


def grid(gcfg: Mapping) -> list[dict]:
    out = []
    for b, lp in itertools.product(gcfg["num_beams"], gcfg["length_penalty"]):
        if b == 1 and lp != gcfg["length_penalty"][0]:
            continue   # length_penalty only affects beam search
        out.append({"num_beams": int(b), "length_penalty": float(lp),
                    "no_repeat_ngram_size": int(gcfg["no_repeat_ngram_size"]), "early_stopping": b > 1})
    return out


def evaluate(backend, sources: Sequence[str], refs: Sequence[str], decoding: Mapping,
             max_new_tokens: int, min_new_tokens: int = 0) -> dict:
    t0 = time.perf_counter()
    hyps = backend.generate([GenerationRequest(s, min_new_tokens, max_new_tokens) for s in sources], decoding)
    sec = time.perf_counter() - t0
    m = rouge.corpus(refs, hyps)
    f = facts.aggregate(facts.check_many(hyps, sources))
    return {**decoding, **m, **{k: v for k, v in f.items() if k != "n"},
            "gen_words": sum(count_words(h) for h in hyps) / max(1, len(hyps)),
            "seconds_per_text": sec / max(1, len(sources))}


def tune(backend, sources: Sequence[str], refs: Sequence[str], gcfg: Mapping, max_new_tokens: int,
         scfg: Mapping, log=print) -> tuple[list[dict], dict]:
    rows = []
    for d in grid(gcfg):
        r = evaluate(backend, sources, refs, d, max_new_tokens)
        log({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        rows.append(r)
    best = select_best(rows, scfg["primary"], scfg["faith_key"], float(scfg["faith_tolerance"]))
    keys = ("num_beams", "length_penalty", "no_repeat_ngram_size", "early_stopping")
    return rows, {k: best[k] for k in keys}
