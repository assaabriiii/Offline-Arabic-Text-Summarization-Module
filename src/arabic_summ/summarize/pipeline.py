"""Summarization strategies sharing one length policy and one aggregation step.

* ``direct``      – proposal §11.3: whole text if it fits, else chunks of <= 900 tokens with
                    <= 80 overlap, each summarized, then aggregated (hierarchical).
* ``segmentwise`` – same mechanism with small segments (e.g. 320 tokens) so the summary
                    length scales with the input although the model was trained on short
                    XL-Sum highlights (strategy B).
* ``hybrid``      – extractive selection of salient sentences (TextRank) to ~1.6x the word
                    budget, then abstractive rewriting of sentence groups (strategy C).
* ``extractive``  – TextRank selection only (baseline, no model).
* ``lead``        – first sentences up to the budget (baseline, no model).

Aggregation ("final pass"): split partial summaries into sentences, drop near-duplicate
sentences across parts, keep document order, trim to the upper length bound at a sentence
boundary, optionally re-summarize abstractively (`merge: abstractive`).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

from ..backends.base import GenerationRequest, Seq2SeqBackend
from ..text.chunk import chunk_sentences
from ..text.normalize import NormalizationPolicy, normalize
from ..text.segment import Sentence, split_sentences
from ..text.words import count_words
from . import extractive
from .length import LengthPolicy, allocate, token_bounds
from .postprocess import dedup_sentences, join, to_sentences, trim_to_words

STRATEGIES = ("direct", "segmentwise", "hybrid", "extractive", "lead")


@dataclass
class SummaryResult:
    summary: str
    strategy: str
    ratio: float
    input_words: int
    target_words: int
    bounds: tuple[int, int]
    output_words: int
    length_ok: bool
    n_parts: int
    n_model_calls: int
    seconds: dict = field(default_factory=dict)
    parts: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class Summarizer:
    def __init__(self, cfg: Mapping, backend: Seq2SeqBackend | None = None, decoding: Mapping | None = None,
                 tokens_per_word: float | None = None):
        p = cfg["pipeline"]
        self.p = p
        self.policy = LengthPolicy.from_config(p)
        self.norm = NormalizationPolicy.from_dict(cfg["normalization"]["model"])
        self.backend = backend
        self.decoding = dict(decoding if decoding is not None else cfg.get("decoding", {}))
        lc = p["length_control"]
        # per-model calibration (02_preprocess) overrides the generic config value
        self.tokens_per_word = float(tokens_per_word or lc["tokens_per_word"])
        self.min_factor, self.max_factor = float(lc["min_factor"]), float(lc["max_factor"])

    # ------------------------------------------------------------------ public
    def summarize(self, text: str, ratio: float | None = None, strategy: str | None = None) -> SummaryResult:
        strategy = strategy or self.p["strategy"]
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if strategy in ("direct", "segmentwise", "hybrid") and self.backend is None:
            raise ValueError(f"strategy {strategy!r} needs a model backend")
        ratio = self.policy.clamp_ratio(self.p["default_ratio"] if ratio is None else ratio)
        t0 = time.perf_counter()
        clean = normalize(text, self.norm)
        sents = split_sentences(clean)
        n_in = count_words(clean)
        target = self.policy.target_words(n_in, ratio)
        lo, hi = self.policy.bounds(n_in, ratio, self.p["length_tolerance"])
        t1 = time.perf_counter()

        calls = 0
        if strategy in ("extractive", "lead"):
            parts, sentences_out = [], self._extract(sents, target, strategy, hi)
        else:
            units = self._units(sents, target, strategy)
            reqs = []
            for u_text, u_target in units:
                mn, mx = token_bounds(u_target, self.tokens_per_word, self.min_factor, self.max_factor)
                reqs.append(GenerationRequest(u_text, mn, mx))
            outs = self.backend.generate(reqs, self.decoding) if reqs else []
            calls += len(reqs)
            parts = [{"input_tokens": self.backend.count_tokens(u), "target_words": t, "output": o}
                     for (u, t), o in zip(units, outs)]
            sentences_out = dedup_sentences(to_sentences(outs), self.p["dedup_sentence_threshold"])
            if self.p["merge"] == "abstractive" and len(units) > 1:
                mn, mx = token_bounds(target, self.tokens_per_word, self.min_factor, self.max_factor)
                merged = self.backend.generate([GenerationRequest(join(sentences_out), mn, mx)], self.decoding)[0]
                calls += 1
                sentences_out = dedup_sentences(to_sentences([merged]), self.p["dedup_sentence_threshold"])
        t2 = time.perf_counter()

        final = trim_to_words(sentences_out, hi, lo)
        summary = join(final)
        n_out = count_words(summary)
        t3 = time.perf_counter()
        return SummaryResult(
            summary=summary, strategy=strategy, ratio=ratio, input_words=n_in, target_words=target,
            bounds=(lo, hi), output_words=n_out, length_ok=lo <= n_out <= hi,
            n_parts=max(1, len(parts)), n_model_calls=calls,
            seconds={"preprocess": round(t1 - t0, 4), "generate": round(t2 - t1, 4),
                     "postprocess": round(t3 - t2, 4), "total": round(t3 - t0, 4)},
            parts=parts,
        )

    # ------------------------------------------------------------------ internals
    def _units(self, sents: Sequence[Sentence], target: int, strategy: str) -> list[tuple[str, int]]:
        """(input text, word target) per model call."""
        count = self.backend.count_tokens
        min_seg = int(self.p["segmentwise"]["min_segment_words"])
        if strategy == "hybrid":
            h = self.p["hybrid"]
            wc = [count_words(s.text) for s in sents]
            keep = extractive.select([s.text for s in sents], wc, round(target * h["extract_expansion"]),
                                     h["position_weight"])
            sents = [sents[i] for i in keep]
            max_tokens, overlap = int(h["group_max_tokens"]), 0
        elif strategy == "segmentwise":
            max_tokens, overlap = int(self.p["segmentwise"]["segment_max_tokens"]), int(self.p["chunking"]["overlap_tokens"])
        else:  # direct
            max_tokens, overlap = int(self.p["chunking"]["max_tokens"]), int(self.p["chunking"]["overlap_tokens"])
        max_tokens = min(max_tokens, self.backend.max_input_tokens - int(self.p["chunking"]["special_tokens_reserve"]))
        chunks = chunk_sentences(sents, count, max_tokens, min(overlap, max_tokens - 1),
                                 self.p["chunking"]["paragraph_break_min_fill"])
        weights = [count_words(c.text) for c in chunks]
        budgets = allocate(target, weights, min_seg)
        return [(c.text, b) for c, b in zip(chunks, budgets) if b > 0]

    def _extract(self, sents: Sequence[Sentence], target: int, strategy: str, hi: int) -> list[str]:
        wc = [count_words(s.text) for s in sents]
        if strategy == "lead":
            idx = extractive.lead(wc, target, max_words=hi)
        else:
            idx = extractive.select([s.text for s in sents], wc, target, self.p["hybrid"]["position_weight"],
                                    max_words=hi)
        return [sents[i].text for i in idx]
