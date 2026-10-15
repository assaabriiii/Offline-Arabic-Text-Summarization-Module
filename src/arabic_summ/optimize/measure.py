"""Measure one variant in a fresh process (so peak RSS is that variant's own process memory).

Spec JSON: {variant, model_dir, config_paths, root, threads, decoding, max_new_tokens,
            val: [{text, summary}], bench: [{sample_id, domain, text}], ratio, strategy, tokens_per_word}
Writes a result JSON. Batch size is 1 and the model is preloaded, as the proposal's reference test.
"""

from __future__ import annotations

import os
import platform
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Mapping

from ..io_utils import read_json, utc_now, write_json


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / 2**20 if sys.platform == "darwin" else r / 1024, 1)   # bytes on macOS, KiB on Linux


def _cpu_model() -> str:
    from ..env import _cpu_model as cm

    return cm()


def _versions() -> dict:
    from importlib import metadata as md

    out = {}
    for pkg in ("torch", "transformers", "onnxruntime", "optimum", "optimum-onnx"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return out


def _q(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


def measure(spec: Mapping) -> dict:
    from ..backends.base import GenerationRequest
    from ..config import load_config
    from ..eval import facts, rouge
    from ..summarize.pipeline import Summarizer
    from ..text.normalize import NormalizationPolicy, normalize
    from ..text.words import count_words
    from .variants import load_backend, weights_size_mb

    cfg = load_config(*spec["config_paths"], overrides={"paths": {"root": spec["root"]}})
    pol = NormalizationPolicy.from_dict(cfg["normalization"]["model"])
    rss0 = peak_rss_mb()
    t0 = time.perf_counter()
    be = load_backend(spec["variant"], spec["model_dir"], spec.get("threads"))
    load_s = time.perf_counter() - t0
    dec = dict(spec["decoding"])
    t0 = time.perf_counter()
    be.generate([GenerationRequest(normalize(spec["bench"][0]["text"], pol), 8, 32)], dec)
    warm_s = time.perf_counter() - t0

    # quality on XL-Sum validation (direct generation, same as 03 decoding tuning)
    srcs = [normalize(r["text"], pol) for r in spec["val"]]
    t0 = time.perf_counter()
    hyps = be.generate([GenerationRequest(s, 0, int(spec["max_new_tokens"])) for s in srcs], dec)
    val_s = time.perf_counter() - t0
    q = rouge.corpus([r["summary"] for r in spec["val"]], hyps)
    f = facts.aggregate(facts.check_many(hyps, srcs))

    # latency through the full pipeline on benchmark texts (no ROUGE: references under review)
    sm = Summarizer(cfg, backend=be, decoding=dec, tokens_per_word=spec.get("tokens_per_word"))
    per = []
    for r in spec["bench"]:
        res = sm.summarize(r["text"], ratio=spec.get("ratio"), strategy=spec.get("strategy"))
        fr = facts.check(res.summary, r["text"])
        per.append({"sample_id": r["sample_id"], "domain": r["domain"], "input_words": res.input_words,
                    "output_words": res.output_words, "length_ok": res.length_ok, "seconds": res.seconds["total"],
                    "model_calls": res.n_model_calls, "numbers_ok": fr.numbers_ok, "novel_rate": fr.novel_rate})
    secs = [p["seconds"] for p in per]
    return {
        "variant": spec["variant"], "model_dir": spec["model_dir"], "measured_utc": utc_now(),
        "weights_mb": weights_size_mb(spec["model_dir"]),
        "load_seconds": round(load_s, 2), "warmup_seconds": round(warm_s, 2),
        "peak_rss_mb": peak_rss_mb(), "rss_before_load_mb": rss0,
        "val_n": len(srcs), "val_seconds_per_text": round(val_s / max(1, len(srcs)), 3),
        **{k: round(v, 5) for k, v in q.items()},
        **{f"val_{k}": round(v, 5) for k, v in f.items() if k != "n"},
        "val_gen_words": round(sum(count_words(h) for h in hyps) / max(1, len(hyps)), 1),
        "bench_n": len(per), "latency_median_s": round(statistics.median(secs), 3) if secs else None,
        "latency_p90_s": _q(secs, 0.9), "latency_max_s": max(secs) if secs else None,
        "bench_length_ok": round(sum(p["length_ok"] for p in per) / max(1, len(per)), 4),
        "bench_numbers_ok": round(sum(p["numbers_ok"] for p in per) / max(1, len(per)), 4),
        "bench_novel_rate": round(sum(p["novel_rate"] for p in per) / max(1, len(per)), 4),
        "per_sample": per, "val_examples": hyps[:3],
        "machine": {"cpu_model": _cpu_model(), "cpu_count": os.cpu_count(), "runtime_versions": _versions(), "threads": spec.get("threads"), "python": platform.python_version(),
                    "os": platform.platform(), "processor": platform.processor() or platform.machine()},
        "decoding": dec, "strategy": spec.get("strategy"), "ratio": spec.get("ratio"),
    }


def main(spec_path: str, out_path: str) -> int:
    res = measure(read_json(spec_path))
    write_json(out_path, res)
    print({k: res[k] for k in ("variant", "rouge1", "val_number_precision", "latency_median_s", "peak_rss_mb", "weights_mb")})
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
