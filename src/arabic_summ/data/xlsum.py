"""XL-Sum (Arabic) acquisition.

The HF dataset `csebuetnlp/xlsum` ships a loading script, which recent `datasets`
releases no longer execute. We therefore download the raw archive for the pinned
dataset commit with `huggingface_hub`, verify its SHA256 and read the JSONL files
directly. This also gives exact file hashes for the data manifest.
"""

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Mapping

import pandas as pd

from ..io_utils import sha256_file

SPLIT_PATTERNS = {"train": "_train.jsonl", "validation": "_val.jsonl", "test": "_test.jsonl"}
FIELDS = ["id", "url", "title", "summary", "text"]


def download_archive(xcfg: Mapping, dest_dir: Path) -> tuple[Path, str]:
    from huggingface_hub import hf_hub_download

    p = Path(
        hf_hub_download(
            repo_id=xcfg["repo_id"],
            filename=xcfg["archive"],
            repo_type="dataset",
            revision=xcfg["revision"],
            local_dir=str(dest_dir),
        )
    )
    digest = sha256_file(p)
    expected = xcfg.get("archive_sha256")
    if expected and digest != expected:
        raise RuntimeError(f"XL-Sum archive SHA256 mismatch: got {digest}, expected {expected}")
    return p, digest


def extract(archive: Path, out_dir: Path) -> dict[str, Path]:
    """Extract (idempotently) and return {split: jsonl path}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    found = _find_jsonl(out_dir)
    if len(found) < len(SPLIT_PATTERNS):
        with tarfile.open(archive, "r:bz2") as tf:
            try:
                tf.extractall(out_dir, filter="data")  # Python >= 3.12: blocks path traversal
            except TypeError:
                tf.extractall(out_dir)
        found = _find_jsonl(out_dir)
    missing = set(SPLIT_PATTERNS) - set(found)
    if missing:
        raise RuntimeError(f"archive lacks split files for {sorted(missing)} under {out_dir}")
    return found


def _find_jsonl(d: Path) -> dict[str, Path]:
    out = {}
    for split, suffix in SPLIT_PATTERNS.items():
        hits = sorted(d.rglob(f"*{suffix}"))
        if hits:
            out[split] = hits[0]
    return out


def load_split(p: Path) -> pd.DataFrame:
    df = pd.read_json(p, lines=True, dtype=False)
    missing = set(FIELDS) - set(df.columns)
    if missing:
        raise RuntimeError(f"{p.name} lacks fields {sorted(missing)}")
    df = df[FIELDS].copy()
    for c in FIELDS:
        df[c] = df[c].fillna("").astype(str)
    return df
