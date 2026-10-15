#!/usr/bin/env python3
"""Offline CPU benchmark for the acceptance test (proposal §7 reference environment).

Run on the x86-64 machine (>= 8 cores, 16 GB RAM, SSD) with the network disabled:

    python bench/run_bench.py --model-dir models/arabart_ft_cpu --texts bench_texts.jsonl --out bench_out

`--texts` is a JSONL or CSV with `sample_id`, `text` (and optionally `domain`), e.g. the selected rows of
benchmark_items.csv. Batch size 1, model preloaded, one warm-up call. Records CPU model, thread count,
OS, warm-up time, per-sample input length and latency, peak process memory, model size, and an
outbound-network probe. Exit code 0 only if every hard criterion passes.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arabic_summ.app.service import SummarizerService, force_offline  # noqa: E402
from arabic_summ.config import load_config  # noqa: E402
from arabic_summ.optimize.measure import peak_rss_mb  # noqa: E402
from arabic_summ.optimize.variants import weights_size_mb  # noqa: E402

LIMITS = {"median_s_300_700": 15.0, "median_s_300_1000": 25.0, "rss_mb": 8192.0, "model_mb": 1024.0}


def network_probe(host: str = "1.1.1.1", port: int = 443, timeout: float = 2.0) -> str:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return "reachable"
    except OSError:
        return "blocked"


def read_texts(p: Path) -> list[dict]:
    if p.suffix == ".csv":
        import csv

        with p.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if rows and "status" in rows[0]:
            rows = [r for r in rows if r["status"] == "selected"] or rows
    else:
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [{"sample_id": r.get("sample_id", str(i)), "domain": r.get("domain", ""), "text": r["text"]} for i, r in enumerate(rows)]


def cpu_model() -> str:
    from arabic_summ.env import _cpu_model

    return _cpu_model()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--texts", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("bench_out"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--ratio", type=float)
    ap.add_argument("--strategy")
    ap.add_argument("--limit", type=int, help="first N texts only (smoke test)")
    a = ap.parse_args(argv)

    force_offline()
    net = network_probe()
    cfg = load_config(a.config, overrides={"paths": {"root": str(a.out / "_root")}})
    texts = read_texts(a.texts)[: a.limit]
    rss0 = peak_rss_mb()
    svc = SummarizerService(cfg, a.model_dir, threads=a.threads)
    warm = svc.warm_up()
    per = []
    for r in texts:
        t0 = time.perf_counter()
        res = svc.summarize(r["text"], ratio=a.ratio, strategy=a.strategy)
        wall = time.perf_counter() - t0
        per.append({"sample_id": r["sample_id"], "domain": r["domain"], "input_words": res["input_words"],
                    "output_words": res["output_words"], "bounds": res["bounds"], "length_ok": res["length_ok"],
                    "seconds": round(wall, 3), "model_calls": res["n_model_calls"],
                    "unsupported_numbers": res["unsupported_numbers"], "summary": res["summary"]})
        print(f"{r['sample_id']:>10} {res['input_words']:5d} w -> {res['output_words']:4d} w  {wall:6.2f} s", flush=True)

    def med(lo, hi):
        xs = [p["seconds"] for p in per if lo <= p["input_words"] <= hi]
        return (round(statistics.median(xs), 3), len(xs)) if xs else (None, 0)

    m1, n1 = med(300, 700)
    m2, n2 = med(300, 1000)
    rss = peak_rss_mb()
    size = weights_size_mb(a.model_dir)
    checks = {
        "median_300_700_words": {"value": m1, "n": n1, "limit": LIMITS["median_s_300_700"], "pass": m1 is not None and m1 <= LIMITS["median_s_300_700"]},
        "median_300_1000_words": {"value": m2, "n": n2, "limit": LIMITS["median_s_300_1000"], "pass": m2 is not None and m2 <= LIMITS["median_s_300_1000"]},
        "peak_process_memory_mb": {"value": rss, "limit": LIMITS["rss_mb"], "pass": rss <= LIMITS["rss_mb"]},
        "model_weights_mb": {"value": size, "limit": LIMITS["model_mb"], "pass": size <= LIMITS["model_mb"]},
        "length_10_30_percent": {"value": round(sum(p["length_ok"] for p in per) / max(1, len(per)), 4), "pass": all(p["length_ok"] for p in per)},
        "network_during_run": {"value": net, "pass": None, "note": "should be 'blocked' for the acceptance test"},
    }
    report = {
        "machine": {"cpu_model": cpu_model(), "logical_cpus": os.cpu_count(), "threads": a.threads,
                    "os": platform.platform(), "python": platform.python_version()},
        "model": {**svc.info.__dict__, "weights_mb": size},
        "warmup_seconds": warm, "rss_before_load_mb": rss0, "n_texts": len(per),
        "latency": {"median_s": round(statistics.median([p["seconds"] for p in per]), 3) if per else None,
                    "p90_s": sorted(p["seconds"] for p in per)[int(0.9 * (len(per) - 1))] if per else None,
                    "max_s": max((p["seconds"] for p in per), default=None)},
        "checks": checks, "per_sample": per,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "bench_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_sample",)}, ensure_ascii=False, indent=1))
    hard = [c["pass"] for c in checks.values() if c["pass"] is not None]
    return 0 if all(hard) else 1


if __name__ == "__main__":
    sys.exit(main())
