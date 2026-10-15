import json
import os
import sys
from pathlib import Path

import pytest

from arabic_summ.config import load_config

sys.path.insert(0, str(Path(__file__).parent))
CFG = load_config("configs/default.yaml")


@pytest.fixture
def tiny(tmp_path):
    pytest.importorskip("transformers")
    import test_eval_train as T

    _, d = T._tiny_seq2seq(tmp_path)
    (d / "decoding.json").write_text(json.dumps({"num_beams": 1}))
    (d / "runtime.json").write_text(json.dumps({"variant": "torch_fp32", "tokens_per_word": 1.3, "strategy": "segmentwise"}))
    return d


def test_service_is_offline_and_reports_fact_flags(tiny):
    from arabic_summ.app.service import SummarizerService

    svc = SummarizerService(CFG, tiny, threads=1)
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert svc.info.variant == "torch_fp32" and svc.info.tokens_per_word == 1.3 and svc.info.decoding["num_beams"] == 1
    r = svc.summarize("قالت المنظمة إن 959 شخصا لقوا حتفهم. " * 60, strategy="extractive")
    assert r["length_ok"] and r["unsupported_numbers"] == [] and r["variant"] == "torch_fp32"
    with pytest.raises(ValueError):
        svc.summarize("   ")
    with pytest.raises(FileNotFoundError):
        SummarizerService(CFG, tiny.parent / "nope")


def test_bench_runner_writes_report_and_fails_on_length(tiny, tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
    import run_bench

    texts = tmp_path / "t.jsonl"
    texts.write_text("\n".join(json.dumps({"sample_id": f"X{i}", "domain": "news",
                                           "text": "قالت المنظمة إن 959 شخصا لقوا حتفهم في المحافظات. " * 40}, ensure_ascii=False)
                               for i in range(2)), encoding="utf-8")
    rc = run_bench.main(["--model-dir", str(tiny), "--texts", str(texts), "--out", str(tmp_path / "out"), "--threads", "1"])
    rep = json.loads((tmp_path / "out" / "bench_report.json").read_text())
    assert rep["n_texts"] == 2 and rep["machine"]["threads"] == 1 and rep["warmup_seconds"] >= 0
    assert set(rep["checks"]) >= {"median_300_700_words", "peak_process_memory_mb", "model_weights_mb", "network_during_run"}
    # a random tiny model can't write 30+ words, so the length criterion must fail -> exit 1
    assert rep["checks"]["length_10_30_percent"]["pass"] is False and rc == 1
    rc2 = run_bench.main(["--model-dir", str(tiny), "--texts", str(texts), "--out", str(tmp_path / "out2"),
                          "--threads", "1", "--strategy", "extractive"])
    assert rc2 == 0


def test_web_ui_builds(tiny):
    pytest.importorskip("gradio")
    from arabic_summ.app.service import SummarizerService
    from arabic_summ.app.web import build

    demo = build(SummarizerService(CFG, tiny, threads=1), CFG)
    assert demo is not None
