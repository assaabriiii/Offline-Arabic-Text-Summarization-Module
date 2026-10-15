"""Resumable data-preparation steps used by notebooks/01_data.ipynb and the CLI.

Every step is idempotent: it checks its marker (see io_utils.step_done) and skips if
its outputs already exist on Drive. Pass force=True to recompute.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .. import config as C
from ..env import environment_report
from ..io_utils import mark_step_done, read_json, sha256_file, step_done, utc_now, write_json
from ..text.normalize import normalize_for_matching, policies_from_config
from ..text.words import count_words
from . import benchmark as B
from . import xlsum
from .dedup import Doc, NearDupIndex, deduplicate

SPLITS = ("train", "validation", "test")
DEDUP_PRIORITY = ("test", "validation", "train")


def _xlsum_dir(cfg) -> Path:
    return C.path(cfg, "raw", "xlsum")


def _processed(cfg, *parts) -> Path:
    return C.path(cfg, "processed", "xlsum", *parts)


def manifest_path(cfg) -> Path:
    return C.path(cfg, "processed", "data_manifest.json")


def update_manifest(cfg, section: str, payload) -> dict:
    p = manifest_path(cfg)
    man = read_json(p) if p.exists() else {"schema": 1, "created_utc": utc_now()}
    man[section] = payload
    man["updated_utc"] = utc_now()
    rep = environment_report()
    man["code"] = {"arabic_summ_version": rep["arabic_summ_version"], "git_commit": rep["git_commit"]}
    man.setdefault("environment", {})[section] = {
        k: rep[k] for k in ("python", "os", "cpu_model", "timestamp_utc")
    } | {"packages": rep["packages"]}
    write_json(p, man)
    return man


# ------------------------------------------------------------------ step 1

def step_download(cfg, force: bool = False) -> dict:
    name = "01_xlsum_download"
    if not force and (m := step_done(cfg, name)):
        print(f"[skip] {name}: done at {m['finished_utc']}")
        return m
    xcfg = cfg["xlsum"]
    archive, digest = xlsum.download_archive(xcfg, _xlsum_dir(cfg))
    info = {
        "repo_id": xcfg["repo_id"], "revision": xcfg["revision"], "archive": str(archive),
        "archive_sha256": digest, "archive_bytes": archive.stat().st_size,
        "license": xcfg["license"], "license_url": xcfg["license_url"],
        "download_date_utc": utc_now(),
    }
    update_manifest(cfg, "dataset", info)
    return mark_step_done(cfg, name, outputs=[archive], **info)


# ------------------------------------------------------------------ step 2

def _scratch(cfg) -> Path:
    p = Path(cfg["paths"].get("scratch", "/tmp/arabic_summ_scratch"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_raw_splits(cfg) -> tuple[dict[str, pd.DataFrame], dict]:
    m = step_done(cfg, "01_xlsum_download")
    if not m:
        raise RuntimeError("run step_download first")
    files = xlsum.extract(Path(m["archive"]), _scratch(cfg) / "xlsum_extracted")
    frames, file_info = {}, {}
    for split, p in files.items():
        df = xlsum.load_split(p)
        frames[split] = df
        file_info[p.name] = {"split": split, "sha256": sha256_file(p), "rows": int(len(df))}
    return frames, file_info


def step_dedup(cfg, force: bool = False) -> dict:
    name = "01_xlsum_dedup"
    if not force and (m := step_done(cfg, name)):
        print(f"[skip] {name}: done at {m['finished_utc']}")
        return m
    dcfg = cfg["dedup"]
    _, match_policy = policies_from_config(cfg)
    frames, file_info = load_raw_splits(cfg)

    # quality filter (train only; eval splits are never modified, only flagged)
    quality_dropped = {}
    for split, df in frames.items():
        df["text_words"] = df.text.map(count_words)
        df["summary_words"] = df.summary.map(count_words)
        bad = (df.text_words < dcfg["min_text_words"]) | (df.summary_words < dcfg["min_summary_words"])
        df["quality_flag"] = np.where(bad, "too_short", "")
        if split == "train" or dcfg["modify_eval_splits"]:
            quality_dropped[split] = int(bad.sum())
            frames[split] = df[~bad].reset_index(drop=True)
        else:
            quality_dropped[split] = 0

    docs = []
    for split in SPLITS:
        for rid, text in zip(frames[split].id, frames[split].text):
            docs.append(Doc(f"{split}:{rid}", split, normalize_for_matching(text, match_policy)))
    dup_keys = [d.key for d in docs]
    if len(set(dup_keys)) != len(dup_keys):
        raise RuntimeError("non-unique (split, id) pairs in XL-Sum")
    removable = set(SPLITS) if dcfg["modify_eval_splits"] else {"train"}
    res = deduplicate(docs, DEDUP_PRIORITY, removable, dcfg["shingle_size"], dcfg["num_perm"],
                      dcfg["near_dup_threshold"])

    # every document that duplicates, or is duplicated by, another one (benchmark avoids these)
    involved = set(res.flagged) | {m.other for m in (*res.removed.values(), *res.flagged.values())}

    outputs, processed_info, split_stats = [], {}, {}
    removed_rows = []
    for split in SPLITS:
        df = frames[split]
        keys = split + ":" + df.id
        df["match_hash"] = keys.map(res.exact_hash)
        flag = keys.map(lambda k: res.flagged.get(k))
        df["dup_reason"] = flag.map(lambda m: m.reason if m else "")
        df["dup_of"] = flag.map(lambda m: m.other if m else "")
        df["dup_similarity"] = flag.map(lambda m: m.similarity if m else np.nan)
        df["has_duplicates"] = keys.isin(involved)
        keep = ~keys.isin(res.removed.keys())
        for k in keys[~keep]:
            mt = res.removed[k]
            removed_rows.append(dataclasses.asdict(mt))
        out = df[keep].reset_index(drop=True)
        p = _processed(cfg, f"{split}.parquet")
        out.to_parquet(p, index=False)
        outputs.append(p)
        processed_info[split] = {"path": str(p), "sha256": sha256_file(p), "rows": int(len(out))}
        raw_rows = next(v["rows"] for v in file_info.values() if v["split"] == split)
        exp = cfg["xlsum"]["expected_counts"].get(split)
        split_stats[split] = {
            "raw_rows": raw_rows, "expected_rows": exp, "matches_expected": raw_rows == exp,
            "quality_dropped": quality_dropped[split], "dedup_removed": int((~keep).sum()),
            "flagged_in_place": int((out.dup_reason != "").sum()), "kept_rows": int(len(out)),
        }

    removed_csv = _processed(cfg, "dedup_removed.csv")
    pd.DataFrame(removed_rows, columns=["key", "reason", "other", "similarity"]).to_csv(removed_csv, index=False)
    outputs.append(removed_csv)

    # cache MinHash signatures of kept docs for later leakage checks
    kept_keys = [k for k in res.signatures if k not in res.removed]
    idx_path = _processed(cfg, "minhash_index.npz")
    np.savez_compressed(
        idx_path, keys=np.array(kept_keys), hashvalues=np.stack([res.signatures[k] for k in kept_keys]),
        exact=np.array([res.exact_hash[k] for k in kept_keys]),
    )
    outputs.append(idx_path)

    dedup_info = {
        "method": "sha256(matching-normalized text) + MinHash/LSH over word shingles, greedy by split priority",
        "priority": list(DEDUP_PRIORITY), "removable_splits": sorted(removable),
        "params": dict(dcfg), "minhash_seed": 1,
        "normalization_policy": dataclasses.asdict(match_policy),
        "stats": res.stats(), "removed_list": str(removed_csv),
    }
    report = {"splits": split_stats, "dedup": dedup_info, "source_files": file_info}
    write_json(_processed(cfg, "dedup_report.json"), report)
    update_manifest(cfg, "source_files", file_info)
    update_manifest(cfg, "splits", split_stats)
    update_manifest(cfg, "dedup", dedup_info)
    update_manifest(cfg, "processed_files", processed_info)
    return mark_step_done(cfg, name, outputs=outputs, splits=split_stats)


def load_processed(cfg, split: str) -> pd.DataFrame:
    return pd.read_parquet(_processed(cfg, f"{split}.parquet"))


# ------------------------------------------------------------------ step 3

def step_benchmark_scaffold(cfg, force: bool = False) -> dict:
    name = "01_benchmark_scaffold"
    if not force and (m := step_done(cfg, name)):
        print(f"[skip] {name}: done at {m['finished_utc']}")
        return m
    bcfg = cfg["benchmark"]
    test = load_processed(cfg, "test")
    items = B.select_news_candidates(test, bcfg, cfg["project"]["seed"])
    bdir = C.path(cfg, "benchmark")
    bdir.mkdir(parents=True, exist_ok=True)
    res = B.write_scaffold(bdir, items, bcfg)
    pool = test[(test.text_words >= bcfg["input_min_words"]) & (test.text_words <= bcfg["input_max_words"])]
    info = {**res, "news_candidates": int(len(items)),
            "news_by_length_bin": items.length_bin.value_counts().sort_index().to_dict(),
            "test_articles_in_range": int(len(pool)), "seed": cfg["project"]["seed"]}
    return mark_step_done(cfg, name, outputs=[bdir / B.ITEMS_FILE, bdir / B.REFS_FILE], **info)


# ------------------------------------------------------------------ step 4 (always rerunnable)

def load_training_index(cfg) -> tuple[NearDupIndex, dict, dict]:
    """Index over XL-Sum train + validation (everything that may influence the model)."""
    z = np.load(_processed(cfg, "minhash_index.npz"))
    keys = z["keys"].astype(str)
    mask = np.array([not k.startswith("test:") for k in keys])
    dcfg = cfg["dedup"]
    index = NearDupIndex(keys[mask].tolist(), z["hashvalues"][mask], dcfg["near_dup_threshold"], dcfg["shingle_size"])
    hashes = {h: k for k, h in zip(keys[mask], z["exact"].astype(str)[mask])}
    urls = {}
    for split in ("train", "validation"):
        df = load_processed(cfg, split)
        urls.update({u: f"{split}:{i}" for u, i in zip(df.url, df.id) if u})
    return index, hashes, urls


def validate_benchmark(cfg, strict: bool = False, index_cache: tuple | None = None) -> tuple[B.ValidationReport, dict]:
    bdir = C.path(cfg, "benchmark")
    items, refs = B.read_csv(bdir / B.ITEMS_FILE), B.read_csv(bdir / B.REFS_FILE)
    d = cfg["dedup"]
    rep = B.validate(items, refs, cfg["benchmark"], strict, d["shingle_size"], d["near_dup_threshold"], d["num_perm"])
    index, hashes, urls = index_cache or load_training_index(cfg)
    leak = B.leakage_check(items, index, hashes, urls)
    if not leak["ok"]:
        rep.errors.append(f"leakage: {leak['n_leaks']} benchmark texts overlap XL-Sum train/validation")
    write_json(bdir / "validation_report.json", rep.to_dict())
    write_json(bdir / "leakage_report.json", leak)
    update_manifest(cfg, "benchmark", {
        "items_csv": str(bdir / B.ITEMS_FILE), "items_sha256": sha256_file(bdir / B.ITEMS_FILE),
        "references_csv": str(bdir / B.REFS_FILE), "references_sha256": sha256_file(bdir / B.REFS_FILE),
        "guideline_version": cfg["benchmark"]["guideline_version"],
        "validation": {"ok": rep.ok, "strict": strict, "n_errors": len(rep.errors),
                       "n_warnings": len(rep.warnings), "stats": rep.stats},
        "leakage": {k: leak[k] for k in ("ok", "items_checked", "n_leaks", "checked_utc")},
    })
    return rep, leak


def import_reference_drafts(cfg, drafts_json: Path | str, writer_id: str,
                            reference_source: str = "llm_draft") -> dict:
    """Merge a drafts JSON into references.csv (empty rows only), then validate (non-strict)."""
    bdir = C.path(cfg, "benchmark")
    items, refs = B.read_csv(bdir / B.ITEMS_FILE), B.read_csv(bdir / B.REFS_FILE)
    drafts = read_json(Path(drafts_json))
    refs, info = B.import_reference_drafts(refs, items, drafts, writer_id, cfg["benchmark"], reference_source)
    if info["filled"]:
        B.write_csv(refs, bdir / B.REFS_FILE)
    rep = B.validate(items, refs, cfg["benchmark"], strict=False)
    update_manifest(cfg, "reference_drafts", {
        "drafts_json": str(drafts_json), "drafts_sha256": sha256_file(Path(drafts_json)),
        "writer_id": writer_id, "reference_source": reference_source,
        "filled": len(info["filled"]), "kept_existing": len(info["kept_existing"]),
        "imported_utc": utc_now(),
    })
    return {"filled": len(info["filled"]), "kept_existing": len(info["kept_existing"]),
            "validation_ok": rep.ok, "n_errors": len(rep.errors), "errors": rep.errors[:10],
            "stats": rep.stats}


# ------------------------------------------------------------------ organizational reports + selection

def fetch_org_reports(cfg, repo_root: Path | str = ".", delay: float = 1.0) -> dict:
    """Fetch the committed org-report URL list into benchmark/incoming_org/ and import it.

    Already-downloaded files are skipped (resumable). A text whose content hash differs from the
    recorded one (page edited since harvesting) is still imported but noted in `notes`.
    """
    import time

    from . import web_sources as W

    bcfg = cfg["benchmark"]
    urls = pd.read_csv(Path(repo_root) / bcfg["org_sources_csv"], dtype=str, keep_default_na=False)
    bdir = C.path(cfg, "benchmark")
    incoming = bdir / "incoming_org_web"   # manual uploads stay in incoming_org/
    incoming.mkdir(parents=True, exist_ok=True)
    meta, fetched, failed, changed = [], 0, [], 0
    for i, r in urls.iterrows():
        name = f"{r.source_key}_{i:03d}.txt"
        f = incoming / name
        note = ""
        if not f.exists():
            try:
                _, text = W.extract(W.fetch(W.iri_to_uri(r.url)))
            except RuntimeError as e:
                failed.append(r.url)
                print("  fetch failed:", e)
                continue
            f.write_text(text, encoding="utf-8")
            fetched += 1
            time.sleep(delay)
        if B.content_hash(f.read_text(encoding="utf-8")) != r.content_hash:
            note = "page content changed since harvesting (hash differs from org_reports.csv)"
            changed += 1
        meta.append({"file_name": name, "source": r.org, "source_url": r.url,
                     "license": f"{r.license_note} Terms: {r.terms_url}",
                     "retrieved_date": utc_now()[:10], "notes": note})
    src_csv = incoming / "sources.csv"
    B.write_csv(pd.DataFrame(meta, columns=B.SOURCE_COLUMNS), src_csv)
    added = B.import_text_folder(bdir, incoming, "org_report", bcfg, src_csv)
    return {"urls": int(len(urls)), "fetched_now": fetched, "failed": failed,
            "hash_changed": changed, "imported_new": int(len(added))}


def run_auto_select(cfg) -> dict:
    bdir = C.path(cfg, "benchmark")
    items = B.read_csv(bdir / B.ITEMS_FILE)
    items, report = B.auto_select(items, cfg["benchmark"], cfg["project"]["seed"])
    B.write_csv(items, bdir / B.ITEMS_FILE)
    write_json(bdir / "auto_select_report.json", {"run_utc": utc_now(), **report})
    return report
