"""Summarize benchmark texts with one *system* (model variant + strategy) and append to a JSONL file.

Runs in its own process (the ONNX variants need the ONNX environment). Resumable: sample_ids already
in the output file are skipped. Spec JSON:
  {system, variant (or null for lead/extractive), model_dir, device, strategy, ratio, decoding,
   tokens_per_word, threads, config_paths, root, items: [{sample_id, domain, text}]}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Mapping

from ..io_utils import read_json, utc_now


def run(spec: Mapping, out_path: str | Path) -> int:
    from ..config import load_config
    from ..summarize.pipeline import Summarizer

    out = Path(out_path)
    done = set()
    if out.exists():
        done = {json.loads(l)["sample_id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [r for r in spec["items"] if r["sample_id"] not in done]
    if not todo:
        return 0
    cfg = load_config(*spec["config_paths"], overrides={"paths": {"root": spec["root"]}})
    backend = None
    if spec.get("variant"):
        from ..optimize.variants import load_backend

        backend = load_backend(spec["variant"], spec["model_dir"], spec.get("threads"), device=spec.get("device", "cpu"))
    sm = Summarizer(cfg, backend=backend, decoding=spec.get("decoding"), tokens_per_word=spec.get("tokens_per_word"))
    if backend is not None:   # warm-up, excluded from latency (model preloaded, as in the proposal's test)
        sm.summarize(todo[0]["text"], ratio=spec.get("ratio"), strategy=spec["strategy"])
    with out.open("a", encoding="utf-8") as f:
        for r in todo:
            res = sm.summarize(r["text"], ratio=spec.get("ratio"), strategy=spec["strategy"])
            rec = {"system": spec["system"], "sample_id": r["sample_id"], "domain": r["domain"],
                   **{k: v for k, v in res.to_dict().items() if k != "parts"},
                   "device": getattr(backend, "device", "cpu") if backend else "cpu", "utc": utc_now()}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    return 0


if __name__ == "__main__":
    sys.exit(run(read_json(sys.argv[1]), sys.argv[2]))
