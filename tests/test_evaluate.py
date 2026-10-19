import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from arabic_summ.config import load_config
from arabic_summ.eval import benchmark_eval as BE
from arabic_summ.eval import human as H
from arabic_summ.eval.stats import bootstrap_ci, paired_diff_ci

CFG = load_config("configs/default.yaml")
EV = CFG["evaluate"]
SRC = "قالت منظمة الصحة العالمية إن 959 شخصا لقوا حتفهم في 10 محافظات باليمن. وأضافت أن الاحتياجات ارتفعت إلى 31 مليون دولار. " * 25


def test_weighted_kappa_known_values():
    assert H.weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    # textbook example: quadratic weights penalize far disagreements more than linear
    a, b = [1, 2, 3, 4, 5, 3, 2], [2, 2, 3, 5, 5, 1, 2]
    q, l = H.weighted_kappa(a, b), H.weighted_kappa(a, b, weights="linear")
    assert 0 < l < q < 1
    assert H.weighted_kappa(a, b) == pytest.approx(H.weighted_kappa(b, a))
    assert H.weighted_kappa([5, 1, 5, 1], [1, 5, 1, 5]) < 0


def test_bootstrap_is_seeded_and_brackets_mean():
    x = np.linspace(0, 1, 101)
    m, lo, hi = bootstrap_ci(x)
    assert lo < m < hi and bootstrap_ci(x) == (m, lo, hi)
    d, lo, hi = paired_diff_ci(x + 0.1, x)
    assert d == pytest.approx(0.1) and lo == pytest.approx(0.1) and hi == pytest.approx(0.1)


def _outputs():
    rows = []
    for sys_ in ("sysA", "sysB"):
        for dom in ("news", "org_report"):
            for i in range(6):
                rows.append({"system": sys_, "sample_id": f"{dom[:3]}{i}", "domain": dom, "summary": f"ملخص {sys_} {i}"})
    return pd.DataFrame(rows)


def test_blind_sheets_balanced_and_keyed():
    out = _outputs()
    sheet, key = H.build_sheets(out, {s: "نص" for s in out.sample_id}, n_per_domain=4, n_calibration=5, seed=1, salt="x")
    assert len(sheet) == 16 and list(sheet.columns) == H.SHEET_COLUMNS
    assert "system" not in sheet.columns and "sample_id" not in sheet.columns
    assert key.groupby(["system", "domain"]).size().eq(4).all() and key.calibration.sum() == 5
    assert sheet.item_id.is_unique and set(sheet.item_id) == set(key.item_id)
    sheet2, _ = H.build_sheets(out, {s: "نص" for s in out.sample_id}, 4, 5, seed=1, salt="x")
    assert sheet.equals(sheet2)


def test_agreement_adjudication_and_final_scores():
    out = _outputs()
    sheet, key = H.build_sheets(out, {s: "نص" for s in out.sample_id}, 3, 4, seed=2, salt="y")
    a, b = sheet.copy(), sheet.copy()
    for c in H.CRITERIA:
        a[c], b[c] = 4, 4
    a["critical_error"], b["critical_error"] = "no", "no"
    first = a.item_id.iloc[0]
    b.loc[b.item_id == first, "accuracy"] = 1          # gap 3 -> adjudication
    b.loc[b.item_id == first, "critical_error"] = "yes"
    assert H.validate_sheet(a) == [] and H.validate_sheet(b) == []
    adj = H.adjudication_items(a, b, EV["human_eval"]["max_gap"])
    assert set(adj.criterion) == {"accuracy", "critical_error"} and set(adj.item_id) == {first}
    third = adj.assign(rater_c=["4", "no"])
    fin = H.final_scores(a, b, key, third)
    assert fin.at[first, "accuracy"] == 4 and not fin.at[first, "critical"]
    summ = H.summarize(fin)
    assert summ[summ.domain == "all"].meets_human_target.all() and summ.meets_factual_target.all()
    ag = H.agreement(a, b)
    assert ag["n"] == len(a) and ag["coverage"] == pytest.approx(1.0)
    bad = a.copy(); bad.loc[0, "fluency"] = 7
    assert H.validate_sheet(bad)


def test_benchmark_scoring_and_acceptance():
    refs = pd.DataFrame({"sample_id": ["n1", "o1"], "reference_summary": [SRC[:120], SRC[:120]],
                         "review_status": ["approved", "pending"], "reference_source": ["llm_draft"] * 2})
    outs = pd.DataFrame([{"system": "s", "strategy": "lead", "sample_id": sid, "domain": d, "summary": SRC[:120],
                          "input_words": 450, "output_words": 90, "length_ok": True, "seconds": {"total": 3.0}, "device": "cpu"}
                         for sid, d in (("n1", "news"), ("o1", "org_report"))])
    r_all = BE.reference_table(refs, approved_only=False)
    r_app = BE.reference_table(refs, approved_only=True)
    sc = BE.score_outputs(outs, r_all, {"n1": SRC, "o1": SRC})
    assert len(sc) == 2 and sc.rouge1.eq(1.0).all() and sc.numbers_ok.all()
    assert len(BE.score_outputs(outs, r_app, {"n1": SRC, "o1": SRC})) == 1
    summ = BE.summary_table(sc)
    lat = BE.latency_table(outs)
    acc = {r["criterion"]: r for r in BE.acceptance("s", summ, lat, {**EV["acceptance"], "rouge1_metric": "strict"})}
    assert acc["ROUGE-1 F1"]["pass"] and acc["median latency 300-700 words (cpu)"]["pass"]
    assert acc["human scores >= 4"]["pass"] is None
    xl = summ.assign(rouge1=0.3, rouge1_xlsum=0.45)
    acc = {r["criterion"]: r for r in BE.acceptance("s", xl, lat, EV["acceptance"])}
    assert acc["ROUGE-1 F1, XL-Sum-style Snowball stemming"]["pass"]
    assert acc["ROUGE-1 F1, unstemmed (reported)"]["pass"] is None and "ROUGE-1 F1" not in acc


def test_run_batch_lead_resumable(tmp_path):
    items = [{"sample_id": f"N{i}", "domain": "news", "text": SRC} for i in range(3)]
    spec = {"system": "lead", "variant": None, "strategy": "lead", "ratio": 0.2, "config_paths": ["configs/default.yaml"],
            "root": str(tmp_path), "items": items}
    sp, out = tmp_path / "spec.json", tmp_path / "out.jsonl"
    sp.write_text(json.dumps(spec, ensure_ascii=False))
    env = {**os.environ, "PYTHONPATH": "src"}
    for _ in range(2):
        r = subprocess.run([sys.executable, "-m", "arabic_summ", "summarize-batch", str(sp), str(out)],
                           capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr[-2000:]
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert [x["sample_id"] for x in recs] == ["N0", "N1", "N2"] and all(x["length_ok"] for x in recs)


def test_snowball_rouge_matches_clitic_forms_and_is_secondary():
    pytest.importorskip("nltk")
    from arabic_summ.eval import rouge

    ref, hyp = "المنظمة أعلنت عن الأطفال", "والمنظمة أعلنت للأطفال"
    strict, stemmed = rouge.score(ref, hyp)["rouge1"], rouge.score(ref, hyp, stem="snowball")["rouge1"]
    assert strict < stemmed and stemmed > 0.7
    assert rouge.corpus([ref], [hyp]) == {k: v for k, v in rouge.score(ref, hyp).items()}   # default stays strict
    with pytest.raises(ValueError):
        rouge.tokenize(ref, stem="porter")

    refs = pd.DataFrame({"sample_id": ["n1"], "reference_summary": [ref], "review_status": ["pending"],
                         "reference_source": ["llm_draft"]})
    outs = pd.DataFrame([{"system": "s", "strategy": "segmentwise", "sample_id": "n1", "domain": "news", "summary": hyp,
                          "input_words": 450, "output_words": 90, "length_ok": True, "seconds": {"total": 3.0}, "device": "cpu"}])
    sc = BE.score_outputs(outs, BE.reference_table(refs, False), {"n1": ref}, secondary_stem="snowball")
    assert sc.rouge1_xlsum.iloc[0] > sc.rouge1.iloc[0]
    summ = BE.summary_table(sc)
    strict = {**EV["acceptance"], "rouge1_metric": "strict"}
    acc = {r["criterion"]: r for r in BE.acceptance("s", summ, BE.latency_table(outs), strict)}
    sec = acc["ROUGE-1 F1, XL-Sum-style Snowball stemming (secondary)"]
    assert sec["pass"] is None and sec["value"] == summ[summ.subset == "all"].rouge1_xlsum.iloc[0]
    assert acc["ROUGE-1 F1"]["value"] == summ[summ.subset == "all"].rouge1.iloc[0]          # pass/fail on the strict score
    acc = {r["criterion"]: r for r in BE.acceptance("s", summ, BE.latency_table(outs), EV["acceptance"])}
    assert acc["ROUGE-1 F1, XL-Sum-style Snowball stemming"]["value"] == sec["value"]       # approved metric (decision 38)
    assert acc["ROUGE-1 F1, unstemmed (reported)"]["value"] == summ[summ.subset == "all"].rouge1.iloc[0]
    assert "rouge1_xlsum" not in BE.score_outputs(outs, BE.reference_table(refs, False), {"n1": ref})
