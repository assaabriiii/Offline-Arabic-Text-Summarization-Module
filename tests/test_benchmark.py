import numpy as np
import pandas as pd
import pytest

from arabic_summ.config import load_config
from arabic_summ.data import benchmark as B
from arabic_summ.data.dedup import Doc, NearDupIndex, deduplicate
from arabic_summ.text.normalize import normalize_for_matching
from arabic_summ.text.words import count_words

CFG = load_config("configs/default.yaml")
BCFG = {**CFG["benchmark"], "min_per_domain": 2}
LETTERS = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"


def arabic_text(n_words: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    return " ".join("".join(rng.choice(list(LETTERS), size=rng.integers(3, 7))) for _ in range(n_words))


def make_items(specs):
    rows = []
    for sid, dom, n, seed in specs:
        t = arabic_text(n, seed)
        rows.append({c: "" for c in B.ITEM_COLUMNS} | {
            "sample_id": sid, "domain": dom, "status": "selected", "source": "src", "license": "lic",
            "word_count": str(count_words(t)), "content_hash": B.content_hash(t), "text": t,
        })
    return pd.DataFrame(rows, columns=B.ITEM_COLUMNS)


def approved_refs(items, ratio=0.2):
    refs = B.reference_template(items, BCFG).astype(str)
    for i, r in refs.iterrows():
        n = int(r.input_words)
        refs.loc[i, "reference_summary"] = arabic_text(round(ratio * n), 1000 + i)
        refs.loc[i, "key_points"] = "نقطة اولى\nنقطة ثانية\nنقطة ثالثة"
        refs.loc[i, ["writer_id", "reviewer_id", "review_status"]] = ["w1", "r1", "approved"]
    return refs


@pytest.fixture
def items():
    return make_items([("NEWS-0001", "news", 400, 1), ("NEWS-0002", "news", 800, 2),
                       ("ORG-0001", "org_report", 350, 3), ("ORG-0002", "org_report", 950, 4)])


def test_reference_bounds():
    assert B.reference_bounds(300, BCFG) == (30, 90, 60)
    assert B.reference_bounds(1000, BCFG) == (100, 300, 200)
    lo, hi, t = B.reference_bounds(555, BCFG)
    assert (lo, hi, t) == (56, 166, 111)


def test_parse_key_points():
    assert B.parse_key_points("أ\n- ب | • ج\n\n") == ["أ", "ب", "ج"]


def test_valid_benchmark_passes_strict(items):
    rep = B.validate(items, approved_refs(items), BCFG, strict=True)
    assert rep.ok, rep.summary()
    assert rep.stats["references_approved"] == 4


def test_pending_refs_warn_when_not_strict_and_fail_when_strict(items):
    refs = B.reference_template(items, BCFG).astype(str)
    assert B.validate(items, refs, BCFG, strict=False).ok
    assert not B.validate(items, refs, BCFG, strict=True).ok


def test_reference_length_out_of_bounds(items):
    refs = approved_refs(items)
    refs.loc[0, "reference_summary"] = arabic_text(10, 77)
    rep = B.validate(items, refs, BCFG, strict=True)
    assert any("NEWS-0001" in e and "allowed" in e for e in rep.errors)


def test_writer_must_differ_from_reviewer(items):
    refs = approved_refs(items)
    refs.loc[1, "reviewer_id"] = "w1"
    rep = B.validate(items, refs, BCFG, strict=True)
    assert any("reviewer must differ" in e for e in rep.errors)


def test_input_length_and_duplicates(items):
    bad = pd.concat([items, make_items([("NEWS-0003", "news", 200, 9)])], ignore_index=True)
    dup = items.iloc[[0]].assign(sample_id="NEWS-0009")
    bad = pd.concat([bad, dup], ignore_index=True)
    rep = B.validate(bad, approved_refs(bad), BCFG, strict=True)
    assert any("NEWS-0003" in e and "outside 300-1000" in e for e in rep.errors)
    assert any("NEWS-0009" in e and "exact duplicate" in e for e in rep.errors)


def test_edited_text_detected_by_hash(items):
    items.loc[0, "text"] = items.loc[0, "text"] + " كلمة"
    rep = B.validate(items, approved_refs(items), BCFG, strict=True)
    assert any("content_hash mismatch" in e for e in rep.errors)


def test_domain_balance_enforced_in_strict(items):
    few = items[items.domain == "news"]
    rep = B.validate(few, approved_refs(few), BCFG, strict=True)
    assert any("domain org_report" in e for e in rep.errors)


def test_leakage_check(items):
    train_texts = {"train:1": items.text[0], "train:2": items.text[1] + " اضافة",
                   "train:3": arabic_text(500, 99)}
    docs = [Doc(k, "train", normalize_for_matching(t)) for k, t in train_texts.items()]
    res = deduplicate(docs, ("train",), set(), k=5, num_perm=128, threshold=0.8)
    keys = list(train_texts)
    idx = NearDupIndex(keys, np.stack([res.signatures[k] for k in keys]), 0.8, 5)
    hashes = {h: k for k, h in res.exact_hash.items()}
    leak = B.leakage_check(items, idx, hashes, {})
    kinds = {l["sample_id"]: l["type"] for l in leak["leaks"]}
    assert kinds == {"NEWS-0001": "exact", "NEWS-0002": "near"}


def test_scaffold_never_overwrites(tmp_path, items):
    first = B.write_scaffold(tmp_path, items, BCFG)
    assert set(first["written"]) == {B.ITEMS_FILE, B.REFS_FILE, B.SOURCES_TEMPLATE_FILE}
    df = B.read_csv(tmp_path / B.ITEMS_FILE)
    df.loc[0, "status"] = "rejected"
    B.write_csv(df, tmp_path / B.ITEMS_FILE)
    second = B.write_scaffold(tmp_path, items, BCFG)
    assert second["written"] == []
    assert B.read_csv(tmp_path / B.ITEMS_FILE).status[0] == "rejected"


def test_import_text_folder_is_idempotent(tmp_path, items):
    B.write_scaffold(tmp_path, items, BCFG)
    folder = tmp_path / "org"
    folder.mkdir()
    (folder / "r1.txt").write_text(arabic_text(500, 41), encoding="utf-8")
    (folder / "r2.txt").write_text(arabic_text(600, 42), encoding="utf-8")
    src = tmp_path / "sources.csv"
    pd.DataFrame([{"file_name": "r1.txt", "source": "وزارة", "source_url": "", "license": "public",
                   "retrieved_date": "2026-10-13", "notes": ""}]).to_csv(src, index=False)
    added = B.import_text_folder(tmp_path, folder, "org_report", BCFG, src)
    assert list(added.sample_id) == ["ORG-0003", "ORG-0004"]
    assert added.source.iloc[0] == "وزارة"
    again = B.import_text_folder(tmp_path, folder, "org_report", BCFG, src)
    assert len(again) == 0
    refs = B.read_csv(tmp_path / B.REFS_FILE)
    assert {"ORG-0003", "ORG-0004"} <= set(refs.sample_id)


def test_select_news_candidates_stratified():
    rows = []
    for i in range(300):
        n = 150 + 3 * i  # 150 .. 1047 words
        rows.append({"id": f"id{i}", "url": f"u{i}", "title": "", "summary": "ملخص",
                     "text": arabic_text(n, i), "text_words": n, "has_duplicates": i % 50 == 0})
    test = pd.DataFrame(rows)
    cfg = {**BCFG, "news_candidates": 30}
    items = B.select_news_candidates(test, cfg, seed=42)
    assert len(items) == 30
    assert items.length_bin.value_counts().to_dict() == {"300-499": 10, "500-699": 10, "700-1000": 10}
    assert items.word_count.astype(int).between(300, 1000).all()
    assert not set(items.source_id) & {f"id{i}" for i in range(0, 300, 50)}
    again = B.select_news_candidates(test, cfg, seed=42)
    assert list(again.source_id) == list(items.source_id)  # deterministic


def _candidates(n_per_bin, domain="news", sources=("A", "B")):
    specs, k = [], 0
    for lo in (320, 550, 800):
        for j in range(n_per_bin):
            specs.append((f"X-{k:04d}", domain, lo + j, 500 + k))
            k += 1
    df = make_items(specs)
    df["status"] = "candidate"
    df["length_bin"] = df.word_count.astype(int).map(lambda n: B.length_bin(n, BCFG["length_bins"]))
    df["source"] = [sources[i % len(sources)] for i in range(len(df))]
    return df


def test_screen_rules():
    assert B.screen(arabic_text(400, 1), BCFG) == ""
    assert "length" in B.screen(arabic_text(100, 1), BCFG)
    numeric = " ".join(["123"] * 100) + " " + arabic_text(300, 2)
    assert "number-heavy" in B.screen(numeric, BCFG)


def test_auto_select_balanced_and_idempotent():
    cfg = {**BCFG, "domains": ["news"], "auto_select": {**BCFG["auto_select"], "per_domain": 12}}
    items = _candidates(10)
    items.loc[0, "text"] = arabic_text(100, 9)  # fails screening
    out, rep = B.auto_select(items, cfg, seed=42)
    sel = out[out.status == "selected"]
    assert len(sel) == 12 and rep["news"]["screen_failed"] == 1
    assert sel.length_bin.value_counts().to_dict() == {"300-499": 4, "500-699": 4, "700-1000": 4}
    assert sel.source.value_counts().to_dict() == {"A": 6, "B": 6}
    assert (out.status != "candidate").all()
    assert out.loc[0, "notes"].startswith("auto-screen")
    again, rep2 = B.auto_select(out, cfg, seed=42)
    assert again.equals(out) and rep2["news"]["newly_selected"] == 0
    # deterministic
    out2, _ = B.auto_select(items, cfg, seed=42)
    assert list(out2.status) == list(out.status)


def test_auto_select_redistributes_short_bins():
    cfg = {**BCFG, "domains": ["news"], "auto_select": {**BCFG["auto_select"], "per_domain": 12}}
    items = _candidates(10)
    items = items[~items.length_bin.eq("700-1000")].reset_index(drop=True)  # one bin empty
    out, rep = B.auto_select(items, cfg, seed=1)
    assert rep["news"]["selected_total"] == 12 and rep["news"]["shortfall"] == 0


def test_web_clean_text_cuts_boilerplate():
    from arabic_summ.data.web_sources import clean_text
    body = "فقرة أولى عن التقرير.\nفقرة ثانية.\nنبذة عن اليونيسف\nتعمل اليونيسف في أكثر من 190 بلدا."
    assert clean_text(body) == "فقرة أولى عن التقرير.\nفقرة ثانية."


def test_import_reference_drafts_fills_only_empty_rows(items):
    refs = B.reference_template(items, BCFG).astype(str)
    refs.loc[0, "reference_summary"] = "ملخص بشري موجود"
    drafts = {sid: {"summary": arabic_text(round(0.2 * int(n)), 7 + i),
                    "key_points": ["نقطة اولى", "نقطة ثانية", "نقطة ثالثة"]}
              for i, (sid, n) in enumerate(zip(refs.sample_id, refs.input_words))}
    out, info = B.import_reference_drafts(refs, items, drafts, "model-x", BCFG)
    assert info["kept_existing"] == ["NEWS-0001"] and len(info["filled"]) == 3
    assert out.loc[0, "reference_summary"] == "ملخص بشري موجود"
    row = out.set_index("sample_id").loc["ORG-0001"]
    assert (row.reference_source, row.writer_id, row.reviewer_id, row.review_status) == \
        ("llm_draft", "model-x", "", "pending")
    assert row.reference_words == str(count_words(row.reference_summary))
    again, info2 = B.import_reference_drafts(out, items, drafts, "model-x", BCFG)
    assert info2["filled"] == [] and again.equals(out)
    with pytest.raises(ValueError):
        B.import_reference_drafts(refs, items, {"NEWS-9999": drafts["NEWS-0002"]}, "model-x", BCFG)


def test_llm_drafts_need_a_different_human_reviewer(items):
    refs = approved_refs(items)
    refs["reference_source"] = "llm_draft"
    refs["writer_id"] = "claude"
    refs.loc[0, ["reviewer_id", "review_status"]] = ["", "pending"]
    assert B.validate(items, refs, BCFG, strict=False).ok
    rep = B.validate(items, refs, BCFG, strict=True)
    assert not rep.ok and any("NEWS-0001" in e for e in rep.errors)
    refs.loc[0, ["reviewer_id", "review_status"]] = ["R2", "approved"]
    rep = B.validate(items, refs, BCFG, strict=True)
    assert rep.ok, rep.summary()
    assert rep.stats["approved_by_source"] == {"llm_draft": 4}
    refs.loc[1, "reviewer_id"] = "claude"
    assert not B.validate(items, refs, BCFG, strict=True).ok
