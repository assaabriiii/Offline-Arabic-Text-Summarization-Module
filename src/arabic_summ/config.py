"""YAML config loading. All tunables live in configs/*.yaml; code reads them from here."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


def _deep_merge(base: dict, override: Mapping) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(*paths: str | Path, overrides: Mapping | None = None) -> dict:
    """Load one or more YAML files (later ones deep-merge over earlier ones)."""
    if not paths:
        raise ValueError("at least one config path is required")
    cfg: dict = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def root(cfg: Mapping) -> Path:
    return Path(cfg["paths"]["root"])


def path(cfg: Mapping, key: str, *parts: str) -> Path:
    """Resolve `paths.<key>` under the root, append `parts`, and create the parent dir."""
    base = Path(cfg["paths"][key])
    if not base.is_absolute():
        base = root(cfg) / base
    p = base.joinpath(*parts)
    (p if not parts else p.parent).mkdir(parents=True, exist_ok=True)
    return p


def config_hash(cfg: Mapping) -> str:
    """Stable short hash of a config (or sub-config) for provenance records."""
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
