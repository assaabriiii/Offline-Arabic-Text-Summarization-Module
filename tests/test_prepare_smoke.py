"""End-to-end run of the 01_data steps on a tiny fake XL-Sum archive (no network)."""

import io
import json
import tarfile

import numpy as np
import pandas as pd

from arabic_summ.config import load_config
from arabic_summ.data import benchmark as B
from arabic_summ.data import prepare, xlsum
from arabic_summ.io_utils import read_json, sha256_file

LETTERS = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"


def text(n, seed):
    rng = np.random.default_rng(seed)
    return " ".join("".join(rng.choice(list(LETTERS), size=rng.integers(3, 7))) for _ in range(n))


def fake_archive(path):
    splits = {"train": 40, "val": 15, "test": 30}
    data = {}
    for s, n in splits.items():
        data[s] = [{"id": f"{s}{i}", "url": f"https://x/{s}/{i}", "title": "عنوان",
                    "summary": text(20, 10_000 + i), "text": text(320 + 20 * i, 1000 * len(s) + i)}
                   for i in range(n)]
    data["train"][0]["text"] = data["test"][0]["text"]          # exact cross-split dup
    data["train"][1]["text"] = data["test"][1]["text"] + " زيادة"  # near dup
    data["train"][2]["text"] = "قصير"                               # quality drop
    with tarfile.open(path, "w:bz2") as tf:
        for s, rows in data.items():
            blob = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows).encode("utf-8")
            info = tarfile.TarInfo(f"arabic_{s}.jsonl")
            info.size = len(blob)
            tf.addfile(info, io.BytesIO(blob))
    return data


def test_full_data_pipeline(tmp_path, monkeypatch):
    archive = tmp_path / "fake.tar.bz2"
    fake_archive(archive)
    monkeypatch.setattr(xlsum, "download_archive", lambda xcfg, d: (archive, sha256_file(archive)))
    cfg = load_config("configs/default.yaml", overrides={
        "paths": {"root": str(tmp_path / "root"), "scratch": str(tmp_path / "scratch")},
        "benchmark": {"news_candidates": 9},
    })

    prepare.step_download(cfg)
    m = prepare.step_dedup(cfg)
    s = m["splits"]
    assert s["train"]["raw_rows"] == 40 and s["train"]["quality_dropped"] == 1
    assert s["train"]["dedup_removed"] == 2
    assert s["test"]["kept_rows"] == 30 and s["validation"]["kept_rows"] == 15
    assert not s["train"]["matches_expected"]  # fake sizes differ from the official ones

    test = prepare.load_processed(cfg, "test")
    assert test.set_index("id").loc[["test0", "test1"], "has_duplicates"].all()

    sc = prepare.step_benchmark_scaffold(cfg)
    assert sc["news_candidates"] == 9
    items = B.read_csv(tmp_path / "root/data/benchmark" / B.ITEMS_FILE)
    assert not {"test0", "test1"} & set(items.source_id)

    rep, leak = prepare.validate_benchmark(cfg)
    assert leak["ok"] and leak["items_checked"] == 9
    assert rep.ok, rep.summary()  # pending references are warnings while annotating

    man = read_json(tmp_path / "root/data/processed/data_manifest.json")
    for key in ("dataset", "source_files", "splits", "dedup", "processed_files", "benchmark", "code"):
        assert key in man
    assert man["dataset"]["license"] == "CC BY-NC-SA 4.0"

    # resumability: second run skips everything
    assert prepare.step_dedup(cfg)["finished_utc"] == m["finished_utc"]
