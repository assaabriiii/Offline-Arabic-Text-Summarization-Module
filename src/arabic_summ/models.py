"""Model registry lookups, tokenizer calibration and backend construction from config."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

from . import config as C
from .io_utils import read_json, utc_now, write_json
from .text.normalize import NormalizationPolicy, normalize
from .text.words import count_words


def model_entry(cfg: Mapping, key: str) -> dict:
    """Look up `key` in models.* or models.baselines.*"""
    m = cfg["models"]
    if key in m and key != "baselines":
        return dict(m[key])
    if key in m.get("baselines", {}):
        return dict(m["baselines"][key])
    raise KeyError(f"unknown model key {key!r}")


def all_model_keys(cfg: Mapping) -> list[str]:
    m = cfg["models"]
    return [k for k in m if k != "baselines"] + list(m.get("baselines", {}))


def calibration_path(cfg: Mapping) -> Path:
    return C.path(cfg, "processed", "tokenizer_calibration.json")


def calibrate_tokens_per_word(tokenizer, texts: Iterable[str], policy: NormalizationPolicy) -> dict:
    """Subword tokens per word on normalized text (used to turn word targets into token bounds)."""
    n_tok = n_words = 0
    for t in texts:
        t = normalize(t, policy)
        n_tok += len(tokenizer(t, add_special_tokens=False)["input_ids"])
        n_words += count_words(t)
    return {"tokens": n_tok, "words": n_words, "tokens_per_word": round(n_tok / max(1, n_words), 4)}


def save_calibration(cfg: Mapping, key: str, record: dict) -> dict:
    p = calibration_path(cfg)
    data = read_json(p) if p.exists() else {}
    data[key] = {**record, "updated_utc": utc_now()}
    write_json(p, data)
    return data


def tokens_per_word(cfg: Mapping, key: str) -> float | None:
    p = calibration_path(cfg)
    if p.exists():
        rec = read_json(p).get(key)
        if rec:
            return float(rec["summary"]["tokens_per_word"])
    return None


def load_hf_backend(cfg: Mapping, key: str, **kw):
    from .backends.hf import HFBackend

    e = model_entry(cfg, key)
    return HFBackend(e.get("local_path") or e["repo_id"], revision=None if e.get("local_path") else e.get("revision"), **kw)
