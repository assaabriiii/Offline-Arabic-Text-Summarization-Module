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

Aggregation ("final pass"): drop a fragment cut off by the token limit at the end of each
partial summary, split them into sentences, drop near-duplicate sentences across parts, keep
document order, trim to the upper length bound at a sentence boundary, optionally re-summarize
abstractively (`merge: abstractive`). A model summary still below its goal (`length_control.fill_to`)
is topped up with the best-ranked source sentences it doesn't cover (`length_control.fill`), so
the length rule holds even when the model writes too little. Before that, model sentences that fail
the faithfulness checks (`faithfulness`, see `faithful.py`) are dropped, so the fill replaces them with
source sentences.
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
from .faithful import filter_sentences
from .length import LengthPolicy, allocate, token_bounds
from .postprocess import (covered, dedup_sentences, drop_unfinished, fix_quotes, is_complete, is_question, join,
                          strip_lead, to_sentences, trim_to_words)

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
    filled_words: int = 0          # words added from the source by the length fill
    rejected: list = field(default_factory=list)   # model sentences dropped by the faithfulness filter
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
        self.drop_unfinished = bool(lc.get("drop_unfinished", False))
        self.fill, self.fill_to = lc.get("fill", "none"), lc.get("fill_to", "lo")
        if self.fill not in ("none", "extractive") or self.fill_to not in ("lo", "target"):
            raise ValueError("length_control.fill must be none|extractive and fill_to lo|target")
        self.fill_cover = float(lc.get("fill_cover_threshold", p["dedup_sentence_threshold"]))
        self.fill_skip_titles = bool(lc.get("fill_skip_titles", False))
        self.fill_number_bonus = float(lc.get("fill_number_bonus", 0.0))
        self.fill_skip_questions = bool(lc.get("fill_skip_questions", False))
        self.fill_fix_quotes = bool(lc.get("fill_fix_quotes", False))
        self.fill_strip_leads = bool(lc.get("fill_strip_leads", False))
        self.faith = dict(p.get("faithfulness") or {})

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
                     for (u, t), o in zip(units, outs)]          # raw model outputs
            sentences_out, merge_calls = self._aggregate(outs, target, merge=len(units) > 1)
            calls += merge_calls
        sentences_out, rejected = self._check(sentences_out, sents, strategy)
        t2 = time.perf_counter()

        final, filled = self._finish(sentences_out, sents, strategy, target, lo, hi)
        summary = join(final)
        n_out = count_words(summary)
        t3 = time.perf_counter()
        return SummaryResult(
            summary=summary, strategy=strategy, ratio=ratio, input_words=n_in, target_words=target,
            bounds=(lo, hi), output_words=n_out, length_ok=lo <= n_out <= hi,
            n_parts=max(1, len(parts)), n_model_calls=calls, filled_words=filled, rejected=rejected,
            seconds={"preprocess": round(t1 - t0, 4), "generate": round(t2 - t1, 4),
                     "postprocess": round(t3 - t2, 4), "total": round(t3 - t0, 4)},
            parts=parts,
        )

    def refinish(self, text: str, outputs: Sequence[str], ratio: float | None = None,
                 strategy: str | None = None) -> SummaryResult:
        """Re-run only the post-processing (unfinished-fragment drop, dedup, faithfulness filter, trim, length fill) on
        stored raw model outputs (`parts[i]["output"]` of an earlier run), so post-processing settings
        can be compared on a dev set without regenerating. No abstractive merge (needs the model)."""
        strategy = strategy or self.p["strategy"]
        ratio = self.policy.clamp_ratio(self.p["default_ratio"] if ratio is None else ratio)
        clean = normalize(text, self.norm)
        sents = split_sentences(clean)
        n_in = count_words(clean)
        target = self.policy.target_words(n_in, ratio)
        lo, hi = self.policy.bounds(n_in, ratio, self.p["length_tolerance"])
        sentences_out, _ = self._aggregate(list(outputs), target, merge=False)
        sentences_out, rejected = self._check(sentences_out, sents, strategy)
        final, filled = self._finish(sentences_out, sents, strategy, target, lo, hi)
        summary = join(final)
        n_out = count_words(summary)
        return SummaryResult(summary=summary, strategy=strategy, ratio=ratio, input_words=n_in, target_words=target,
                             bounds=(lo, hi), output_words=n_out, length_ok=lo <= n_out <= hi,
                             n_parts=max(1, len(outputs)), n_model_calls=0, filled_words=filled, rejected=rejected)

    # ------------------------------------------------------------------ internals
    def _aggregate(self, outs: Sequence[str], target: int, merge: bool) -> tuple[list[str], int]:
        """Partial model outputs -> deduplicated sentences (optionally re-summarized). Returns
        (sentences, extra model calls)."""
        if self.drop_unfinished:
            outs = [drop_unfinished(o) for o in outs]
        sentences_out = dedup_sentences(to_sentences(outs), self.p["dedup_sentence_threshold"])
        if merge and self.p["merge"] == "abstractive":
            mn, mx = token_bounds(target, self.tokens_per_word, self.min_factor, self.max_factor)
            merged = self.backend.generate([GenerationRequest(join(sentences_out), mn, mx)], self.decoding)[0]
            return dedup_sentences(to_sentences([merged]), self.p["dedup_sentence_threshold"]), 1
        return sentences_out, 0

    def _check(self, sentences_out: Sequence[str], sents: Sequence[Sentence], strategy: str) -> tuple[list[str], list[dict]]:
        """Faithfulness filter of model sentences (model strategies only). Returns (kept, rejected)."""
        if strategy not in ("direct", "segmentwise", "hybrid") or not self.faith.get("enabled", False):
            return list(sentences_out), []
        return filter_sentences(sentences_out, [s.text for s in sents], self.faith)

    def _finish(self, sentences_out: Sequence[str], sents: Sequence[Sentence], strategy: str, target: int,
                lo: int, hi: int) -> tuple[list[str], int]:
        """Length enforcement: trim at a sentence boundary, then the extractive fill for model strategies."""
        final = trim_to_words(sentences_out, hi, lo)
        if strategy in ("direct", "segmentwise", "hybrid") and self.fill == "extractive":
            return self._fill(final, sents, lo if self.fill_to == "lo" else target, hi)
        return final, 0

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

    def _fill(self, summary: Sequence[str], sents: Sequence[Sentence], goal: int, hi: int) -> tuple[list[str], int]:
        """Top up a too-short model summary with the best-ranked (TextRank) source sentences it doesn't
        already cover, never past `hi` (05, 2026-10-18: all 32 length failures were too short and
        nothing could add words). Optionally sentences with numbers rank higher (`fill_number_bonus`)
        and headings without end punctuation are skipped (`fill_skip_titles`; 05 review: page titles
        glued on), as are rhetorical questions (`fill_skip_questions`); datelines / quoted crossheads
        at the start of a sentence are removed (`fill_strip_leads`) and pieces of multi-sentence quotes
        repaired or skipped (`fill_fix_quotes`) first (PROGRESS 37). If no whole sentence fits, the first words of the best uncovered one are added, as
        `trim_to_words` does. Returns (sentences, words added)."""
        out = list(summary)
        n0 = n = count_words(join(out))
        goal = min(goal, hi)
        if n >= goal or not sents:
            return out, 0
        thr = self.fill_cover
        texts = [s.text for s in sents]
        scores = extractive.textrank_scores(texts, position_weight=self.p["hybrid"]["position_weight"])
        if self.fill_number_bonus:          # on the original sentence: a dateline's date counts
            scores = scores * [1 + self.fill_number_bonus * any(ch.isdigit() for ch in t) for t in texts]
        if self.fill_strip_leads:
            texts = [strip_lead(t, dateline=k < 3) for k, t in enumerate(texts)]
        if self.fill_fix_quotes:
            texts = [fix_quotes(t) or "" for t in texts]      # "" = skipped (no words, never fits)
        order = sorted(range(len(texts)), key=lambda i: -scores[i])
        for i in order:
            if n >= goal:
                break
            if not self._fill_ok(texts[i]):
                continue
            w = count_words(texts[i])
            if n + w <= hi and not covered(texts[i], out, thr):
                out.append(texts[i])
                n += w
        if n < goal:
            rest = [texts[i] for i in order if not covered(texts[i], out, thr)]
            rest = [t for t in rest if self._fill_ok(t)] or rest
            if rest:
                out = trim_to_words(out + rest[:1], hi, goal)
        return out, count_words(join(out)) - n0

    def _fill_ok(self, text: str) -> bool:
        return bool(text) and not ((self.fill_skip_titles and not is_complete(text))
                                   or (self.fill_skip_questions and is_question(text)))
