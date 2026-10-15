import json
import subprocess
import sys
from pathlib import Path

import pytest

from arabic_summ.config import load_config
from arabic_summ.optimize import select as SEL
from arabic_summ.optimize.variants import VARIANTS, model_dir_for

CFG = load_config("configs/default.yaml")
OPT = CFG["optimize"]


def row(v, r1, np_, lat, rss, mb):
    return {"variant": v, "rouge1": r1, "val_number_precision": np_, "latency_median_s": lat,
            "peak_rss_mb": rss, "weights_mb": mb}


def test_gates_and_composite_ranking():
    rows = [row("torch_fp32", 0.40, 0.95, 9.0, 2500, 530),
            row("torch_int8", 0.37, 0.95, 5.0, 1500, 530),      # ROUGE-1 drop 0.03 > 0.02 -> gated out
            row("onnx_fp32", 0.40, 0.95, 6.0, 2000, 540),
            row("onnx_int8", 0.395, 0.95, 3.0, 900, 140),
            row("too_slow", 0.41, 0.96, 20.0, 1000, 100)]
    out = SEL.composite(rows, OPT["gates"], OPT["weights"])
    by = {r["variant"]: r for r in out}
    assert any("ROUGE-1 drop" in f for f in by["torch_int8"]["gate_failures"]) and by["torch_int8"]["score"] is None
    assert any("latency" in f for f in by["too_slow"]["gate_failures"])
    assert out[0]["variant"] == "onnx_int8"
    assert sum(OPT["weights"].values()) == pytest.approx(1.0)


def test_single_eligible_variant_scores_one():
    out = SEL.composite([row("torch_fp32", 0.4, 1.0, 9, 2000, 500)], OPT["gates"], OPT["weights"])
    assert out[0]["score"] == pytest.approx(1.0)


def test_variant_dirs():
    assert set(VARIANTS) == set(OPT["variants"])
    assert model_dir_for("onnx_int8", Path("/m"), Path("/o")) == Path("/o/int8")
    assert model_dir_for("torch_int8", Path("/m"), Path("/o")) == Path("/m")


def test_torch_int8_backend_and_measure_subprocess(tmp_path):
    pytest.importorskip("transformers")
    sys.path.insert(0, str(Path(__file__).parent))
    import test_eval_train as T

    _, model = T._tiny_seq2seq(tmp_path)
    from arabic_summ.backends.base import GenerationRequest
    from arabic_summ.backends.hf import HFBackend

    be = HFBackend(str(model), device="cpu", quantize="dynamic_int8", batch_size=1)
    out = be.generate([GenerationRequest(T.SRC, 2, 6)], {"num_beams": 1})
    assert be.device == "cpu" and isinstance(out[0], str) and be.name.endswith("dynamic_int8")
    spec = {"variant": "torch_int8", "model_dir": str(model), "config_paths": ["configs/default.yaml"],
            "root": str(tmp_path / "root"), "threads": 1, "decoding": {"num_beams": 1}, "max_new_tokens": 8,
            "val": [{"text": T.SRC, "summary": T.GOOD}], "bench": [{"sample_id": "X", "domain": "news", "text": T.SRC * 30}],
            "ratio": 0.2, "strategy": "segmentwise"}
    (tmp_path / "spec.json").write_text(json.dumps(spec, ensure_ascii=False))
    r = subprocess.run([sys.executable, "-m", "arabic_summ", "optimize-measure", str(tmp_path / "spec.json"),
                        str(tmp_path / "res.json")], capture_output=True, text=True,
                       env={**__import__("os").environ, "PYTHONPATH": "src"})
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads((tmp_path / "res.json").read_text())
    assert res["variant"] == "torch_int8" and res["bench_n"] == 1 and res["peak_rss_mb"] > 0
    assert {"rouge1", "val_number_precision", "latency_median_s", "weights_mb"} <= set(res)


@pytest.mark.skipif(not __import__("os").environ.get("ARABIC_SUMM_ORT_PYTHON"),
                    reason="set ARABIC_SUMM_ORT_PYTHON to a python with requirements/colab-onnx.txt installed")
def test_onnx_export_and_measure_in_separate_env(tmp_path):
    import os

    sys.path.insert(0, str(Path(__file__).parent))
    import test_eval_train as T

    _, model = T._tiny_seq2seq(tmp_path)
    py, env = os.environ["ARABIC_SUMM_ORT_PYTHON"], {**os.environ, "PYTHONPATH": "src"}
    r = subprocess.run([py, "-m", "arabic_summ", "onnx-export", str(model), str(tmp_path / "onnx"), "--arch", "avx2"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    for v in ("onnx_fp32", "onnx_int8"):
        d = tmp_path / "onnx" / v.split("_")[1]
        assert (d / "encoder_model.onnx").exists() and (d / "tokenizer.json").exists()
        spec = {"variant": v, "model_dir": str(d), "config_paths": ["configs/default.yaml"], "root": str(tmp_path / "root"),
                "threads": 1, "decoding": {"num_beams": 1}, "max_new_tokens": 8,
                "val": [{"text": T.SRC, "summary": T.GOOD}],
                "bench": [{"sample_id": "X", "domain": "news", "text": T.SRC * 30}], "ratio": 0.2, "strategy": "segmentwise"}
        (tmp_path / f"{v}.json").write_text(json.dumps(spec, ensure_ascii=False))
        r = subprocess.run([py, "-m", "arabic_summ", "optimize-measure", str(tmp_path / f"{v}.json"),
                            str(tmp_path / f"res_{v}.json")], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr[-3000:]
