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


def load_tokenizer(path_or_repo: str, revision: str | None = None, **kw):
    """AutoTokenizer, with a fallback for repos whose tokenizer.json disagrees with the declared class.

    UBC-NLP/AraT5v2 declares `T5Tokenizer` (Unigram) but ships a BPE `tokenizer.json`; transformers 5
    then fails with "'dict' object is not an instance of 'Sequence'". Loading tokenizer.json directly
    gives ids identical to the repo's spiece.model (verified on sample Arabic sentences, 2026-10-15).
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    try:
        return ensure_special_tokens(AutoTokenizer.from_pretrained(path_or_repo, revision=revision, **kw))
    except (TypeError, ValueError):
        # also: a tokenizer saved by transformers 5 declares `TokenizersBackend`, unknown to 4.x (ONNX env)
        tok = PreTrainedTokenizerFast.from_pretrained(path_or_repo, revision=revision, **kw)
        tok.model_input_names = ["input_ids", "attention_mask"]   # seq2seq models take no token_type_ids
        return ensure_special_tokens(tok)


def ensure_special_tokens(tok):
    """Make encodings end with EOS (and start with BOS where the tokenizer has one).

    transformers 5 builds AraBART's sentencepiece `BarthezTokenizer` without the post-processor
    that adds `<s> ... </s>`, so sources and *labels* came out with no special tokens: the model
    never saw where a summary ends and fine-tuning collapsed to empty outputs (03 run, 2026-10-16).
    Installs the BART-style template (`<s> $A </s>`, or `$A </s>` for tokenizers without BOS,
    e.g. T5) only when the tokenizer doesn't already append EOS.
    """
    eos, eos_t = tok.eos_token_id, tok.eos_token
    if eos is None:
        return tok
    ids = tok("ا")["input_ids"]
    if ids and ids[-1] == eos:
        return tok
    backend = getattr(tok, "backend_tokenizer", None) or getattr(tok, "_tokenizer", None)
    if backend is None:
        raise RuntimeError(f"{type(tok).__name__} does not append EOS and has no fast backend to fix")
    from tokenizers.processors import TemplateProcessing

    bos, bos_t = tok.bos_token_id, tok.bos_token
    if bos is not None:
        single, pair = f"{bos_t} $A {eos_t}", f"{bos_t} $A {eos_t} {eos_t} $B {eos_t}"
        special = [(bos_t, bos), (eos_t, eos)]
    else:
        single, pair, special = f"$A {eos_t}", f"$A {eos_t} $B {eos_t}", [(eos_t, eos)]
    backend.post_processor = TemplateProcessing(single=single, pair=pair, special_tokens=special)
    ids = tok("ا")["input_ids"]
    if not ids or ids[-1] != eos:
        raise RuntimeError(f"could not make {type(tok).__name__} append EOS")
    return tok


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
        rec = read_json(p).get(calibration_key(cfg, key) if key in all_model_keys(cfg) else key)
        if rec:
            return float(rec["summary"]["tokens_per_word"])
    return None


def local_path(cfg: Mapping, key: str) -> Path | None:
    """Resolved local directory of a model entry (relative paths live under paths.root)."""
    lp = model_entry(cfg, key).get("local_path")
    if not lp:
        return None
    p = Path(lp)
    return p if p.is_absolute() else C.root(cfg) / p


def decoding_for(cfg: Mapping, key: str) -> dict:
    """Tuned decoding.json shipped with a local model, else the config defaults."""
    lp = local_path(cfg, key)
    if lp and (lp / "decoding.json").exists():
        return {**cfg.get("decoding", {}), **read_json(lp / "decoding.json")}
    return dict(cfg.get("decoding", {}))


def calibration_key(cfg: Mapping, key: str) -> str:
    """Fine-tuned models share their base model's tokenizer calibration."""
    return model_entry(cfg, key).get("base", key)


def load_hf_backend(cfg: Mapping, key: str, **kw):
    from .backends.hf import HFBackend

    e = model_entry(cfg, key)
    lp = local_path(cfg, key)
    return HFBackend(str(lp) if lp else e["repo_id"], revision=None if lp else e.get("revision"), **kw)
