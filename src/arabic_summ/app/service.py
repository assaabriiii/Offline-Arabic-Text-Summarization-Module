"""Load an exported model folder and summarize Arabic text fully offline.

A model folder is self-contained: weights (PyTorch or ONNX), tokenizer, `decoding.json` and an
optional `runtime.json` ({variant, tokens_per_word, strategy, max_input_tokens}). No network access is
attempted: the Hugging Face offline switches are set before transformers is imported.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
               "GRADIO_ANALYTICS_ENABLED": "False", "DO_NOT_TRACK": "1"}


def force_offline() -> None:
    for k, v in OFFLINE_ENV.items():
        os.environ[k] = v


@dataclass
class ServiceInfo:
    model_dir: str
    variant: str
    strategy: str
    tokens_per_word: float | None
    decoding: dict
    load_seconds: float
    threads: int | None


class SummarizerService:
    def __init__(self, cfg: Mapping, model_dir: str | Path, threads: int | None = None, variant: str | None = None):
        force_offline()
        from ..io_utils import read_json
        from ..optimize.variants import load_backend
        from ..summarize.pipeline import Summarizer

        d = Path(model_dir)
        if not (d / "config.json").exists():
            raise FileNotFoundError(f"{d} is not a model folder (config.json missing)")
        rt = read_json(d / "runtime.json") if (d / "runtime.json").exists() else {}
        dec = {**cfg.get("decoding", {}), **(read_json(d / "decoding.json") if (d / "decoding.json").exists() else {})}
        self.variant = variant or rt.get("variant") or ("onnx_fp32" if (d / "encoder_model.onnx").exists() else "torch_fp32")
        t0 = time.perf_counter()
        self.backend = load_backend(self.variant, d, threads, max_input_tokens=int(rt.get("max_input_tokens", 1024)))
        load_s = time.perf_counter() - t0
        tpw = rt.get("tokens_per_word")
        self.summarizer = Summarizer(cfg, backend=self.backend, decoding=dec, tokens_per_word=tpw)
        self.info = ServiceInfo(str(d), self.variant, rt.get("strategy") or cfg["pipeline"]["strategy"],
                                self.summarizer.tokens_per_word, dec, round(load_s, 2), threads)

    def warm_up(self) -> float:
        t0 = time.perf_counter()
        self.summarize("هذا نص قصير لتهيئة النموذج قبل الاستخدام. " * 40)
        return round(time.perf_counter() - t0, 2)

    def summarize(self, text: str, ratio: float | None = None, strategy: str | None = None) -> dict:
        from ..eval import facts

        if not text or not text.strip():
            raise ValueError("empty input")
        res = self.summarizer.summarize(text, ratio=ratio, strategy=strategy or self.info.strategy)
        fr = facts.check(res.summary, text)
        out = {k: v for k, v in res.to_dict().items() if k != "parts"}
        out.update({"variant": self.variant, "unsupported_numbers": fr.unsupported_numbers,
                    "novel_rate": round(fr.novel_rate, 3)})
        return out
