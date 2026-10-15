"""Export the selected model with provenance: file hashes, base checkpoint, data and filter
records, hyperparameters, validation metrics, decoding config, code commit and environment."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Mapping

from .. import config as C
from ..env import environment_report
from ..io_utils import sha256_file, utc_now, write_json

DECODING_FILE = "decoding.json"
PROVENANCE_FILE = "provenance.json"


def export_model(cfg: Mapping, src_dir: str | Path, dest_name: str, decoding: Mapping, record: Mapping) -> dict:
    dest = C.path(cfg, "models", dest_name, "x").parent
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src_dir, dest)
    write_json(dest / DECODING_FILE, dict(decoding))
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
             for p in sorted(dest.iterdir()) if p.is_file() and p.name != PROVENANCE_FILE}
    env = environment_report()
    prov = {
        "model_name": dest_name, "exported_utc": utc_now(), "source_run_dir": str(src_dir),
        "size_mb": round(sum(f["bytes"] for f in files.values()) / 2**20, 1),
        "files": files, **dict(record),
        "code": {"git_commit": env.get("git_commit"), "arabic_summ_version": env.get("arabic_summ_version")},
        "config_hash": C.config_hash(cfg),
        "environment": {k: env.get(k) for k in ("python", "os", "gpu", "packages", "timestamp_utc")},
    }
    write_json(dest / PROVENANCE_FILE, prov)
    return prov
