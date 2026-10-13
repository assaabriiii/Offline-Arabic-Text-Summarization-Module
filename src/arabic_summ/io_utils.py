"""File helpers: hashing, atomic writes, and step markers that make notebooks resumable."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import config as _config


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(p: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def write_json(p: str | Path, obj: Any) -> Path:
    """Write JSON atomically (temp file + rename) so a Colab disconnect never leaves half a file."""
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return p


def read_json(p: str | Path) -> Any:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ step markers
# A step is "done" only if its marker exists AND every output it lists still exists,
# so deleting an output on Drive forces that step to rerun.

def _marker_path(cfg: Mapping, step: str) -> Path:
    return _config.path(cfg, "markers", f"{step}.json")


def step_done(cfg: Mapping, step: str) -> dict | None:
    mp = _marker_path(cfg, step)
    if not mp.exists():
        return None
    meta = read_json(mp)
    if all(Path(o).exists() for o in meta.get("outputs", [])):
        return meta
    return None


def mark_step_done(cfg: Mapping, step: str, outputs: Iterable[str | Path] = (), **meta: Any) -> dict:
    record = {"step": step, "finished_utc": utc_now(), "outputs": [str(o) for o in outputs], **meta}
    write_json(_marker_path(cfg, step), record)
    return record


def reset_step(cfg: Mapping, step: str) -> None:
    _marker_path(cfg, step).unlink(missing_ok=True)
