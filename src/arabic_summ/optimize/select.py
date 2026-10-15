"""Proposal model-selection rule. Stage 1 (gates): weights <= 1 GB, process memory <= 8 GB,
median latency <= 15 s, and for quantized variants a ROUGE-1 drop <= 0.02 vs the FP32 reference.
Stage 2: composite score = 50 % quality & factual accuracy + 30 % CPU time + 10 % memory + 10 % size,
each min-max normalized over the variants that passed the gates."""

from __future__ import annotations

from typing import Mapping, Sequence


def _norm(vals: Sequence[float], higher_better: bool) -> list[float]:
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [1.0] * len(vals)
    return [((v - lo) / (hi - lo)) if higher_better else ((hi - v) / (hi - lo)) for v in vals]


def gate(row: Mapping, ref: Mapping | None, g: Mapping) -> list[str]:
    fails = []
    if row["weights_mb"] > g["max_model_mb"]:
        fails.append(f"weights {row['weights_mb']} MB > {g['max_model_mb']}")
    if row["peak_rss_mb"] > g["max_rss_mb"]:
        fails.append(f"peak RSS {row['peak_rss_mb']} MB > {g['max_rss_mb']}")
    if row["latency_median_s"] > g["max_median_seconds"]:
        fails.append(f"median latency {row['latency_median_s']} s > {g['max_median_seconds']}")
    if ref is not None and ref is not row and ref["rouge1"] - row["rouge1"] > g["max_rouge1_drop"]:
        fails.append(f"ROUGE-1 drop {ref['rouge1'] - row['rouge1']:.4f} > {g['max_rouge1_drop']}")
    return fails


def composite(rows: Sequence[Mapping], gates: Mapping, weights: Mapping, reference: str = "torch_fp32") -> list[dict]:
    ref = next((r for r in rows if r["variant"] == reference), None)
    out = [dict(r, gate_failures=gate(r, ref, gates)) for r in rows]
    ok = [r for r in out if not r["gate_failures"]]
    if ok:
        quality = [(a + b) / 2 for a, b in zip(_norm([r["rouge1"] for r in ok], True),
                                                _norm([r["val_number_precision"] for r in ok], True))]
        parts = {"quality": quality, "time": _norm([r["latency_median_s"] for r in ok], False),
                 "memory": _norm([r["peak_rss_mb"] for r in ok], False), "size": _norm([r["weights_mb"] for r in ok], False)}
        for i, r in enumerate(ok):
            r["score_parts"] = {k: round(v[i], 4) for k, v in parts.items()}
            r["score"] = round(sum(float(weights[k]) * v[i] for k, v in parts.items()), 4)
    for r in out:
        r.setdefault("score", None)
    return sorted(out, key=lambda r: (r["score"] is None, -(r["score"] or 0)))
