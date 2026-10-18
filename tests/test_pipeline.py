import numpy as np
import pytest

from arabic_summ.backends.base import EchoBackend
from arabic_summ.config import load_config
from arabic_summ.summarize import extractive
from arabic_summ.summarize.length import LengthPolicy, allocate, token_bounds
from arabic_summ.summarize.pipeline import Summarizer
from arabic_summ.summarize.postprocess import dedup_sentences, trim_to_words
from arabic_summ.text.words import count_words

CFG = load_config("configs/default.yaml")
LETTERS = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"


def doc(n_sentences, words_per_sentence=15, seed=0, paragraphs_every=5):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_sentences):
        w = " ".join("".join(rng.choice(list(LETTERS), size=rng.integers(3, 7))) for _ in range(words_per_sentence))
        out.append(w + ".")
        if (i + 1) % paragraphs_every == 0:
            out.append("\n\n")
    return " ".join(out)


# ---------------------------------------------------------------- length policy

@pytest.mark.parametrize("n, ratio, expected", [
    (300, 0.2, 60), (1000, 0.2, 200), (1000, 0.3, 300), (2000, 0.3, 300),
    (300, 0.05, 30), (500, 0.9, 150), (100, 0.2, 30),
])
def test_target_words(n, ratio, expected):
    assert LengthPolicy().target_words(n, ratio) == expected


def test_bounds_within_hard_limits():
    p = LengthPolicy()
    for n in range(300, 1001, 7):
        for r in (0.1, 0.2, 0.3):
            lo, hi = p.bounds(n, r)
            assert max(30, np.ceil(0.1 * n)) <= lo <= p.target_words(n, r) <= hi <= min(300, np.floor(0.3 * n))


def test_allocate():
    assert allocate(100, [1, 1, 2]) == [25, 25, 50]
    assert sum(allocate(97, [3, 5, 7])) == 97
    assert allocate(30, [100, 1, 1], min_each=12) == [12, 9, 9] or min(allocate(30, [100, 1, 1], 12)) >= 9
    assert allocate(40, [100, 1, 1], min_each=12)[1] >= 12
    assert allocate(5, []) == []


def test_token_bounds():
    assert token_bounds(50, 1.6, 0.8, 1.25) == (64, 100)


# ---------------------------------------------------------------- postprocess

def test_dedup_sentences():
    s = ["أعلنت الوزارة افتتاح المستشفى الجديد في الرياض.",
         "افتتاح المستشفى الجديد في الرياض.",
         "وتبلغ تكلفة المشروع خمسمئة مليون ريال."]
    assert dedup_sentences(s) == [s[0], s[2]]


def test_dedup_longer_replaces_contained_shorter():
    s = ["افتتاح المستشفى الجديد.", "أعلنت الوزارة افتتاح المستشفى الجديد اليوم."]
    assert dedup_sentences(s) == [s[1]]


def test_trim_to_words_keeps_whole_sentences():
    s = ["واحد اثنان ثلاثة.", "أربعة خمسة.", "ستة سبعة ثمانية تسعة."]
    assert trim_to_words(s, 6) == s[:2]
    assert trim_to_words(["كلمة " * 10], 4) == ["كلمة كلمة كلمة كلمة"]


def test_textrank_prefers_central_sentence():
    s = ["الحكومة تعلن خطة اقتصادية جديدة لدعم النمو.",
         "الخطة الاقتصادية الجديدة تدعم النمو والاستثمار.",
         "الطقس مشمس اليوم في المدينة.",
         "الحكومة تتوقع أن تعزز الخطة النمو الاقتصادي."]
    sc = extractive.textrank_scores(s, position_weight=0.0)
    assert sc.argmin() == 2


# ---------------------------------------------------------------- pipeline (no model)

@pytest.mark.parametrize("strategy", ["direct", "segmentwise", "hybrid", "extractive", "lead"])
def test_all_strategies_respect_length(strategy):
    text = doc(40, 15, seed=1)  # 600 words
    cfg = {**CFG, "pipeline": {**CFG["pipeline"], "length_control": {**CFG["pipeline"]["length_control"], "tokens_per_word": 1.0}}}
    s = Summarizer(cfg, backend=EchoBackend(max_input_tokens=1024))
    r = s.summarize(text, ratio=0.2, strategy=strategy)
    assert r.input_words == 600 and r.target_words == 120
    lo, hi = r.bounds
    assert r.output_words <= hi
    assert r.output_words == count_words(r.summary)
    if strategy in ("extractive", "lead"):
        assert r.n_model_calls == 0
        assert r.length_ok
    else:
        assert r.n_model_calls >= 1


def test_segmentwise_makes_more_calls_than_direct():
    text = doc(50, 15, seed=2)  # 750 words, fits 900 tokens with the echo backend
    s = Summarizer(CFG, backend=EchoBackend())
    assert s.summarize(text, strategy="direct").n_model_calls == 1
    assert s.summarize(text, strategy="segmentwise").n_model_calls >= 3


def test_direct_chunks_long_inputs_with_900_token_limit():
    text = doc(140, 15, seed=3)  # 2100 words
    be = EchoBackend()
    r = Summarizer(CFG, backend=be).summarize(text, strategy="direct")
    assert r.n_model_calls >= 3
    assert all(len(c.text.split()) <= 900 for c in be.calls)
    assert r.output_words <= 300


def test_ratio_is_clamped_and_model_needed():
    s = Summarizer(CFG, backend=None)
    r = s.summarize(doc(30, 15), ratio=0.9, strategy="lead")
    assert r.ratio == 0.3
    with pytest.raises(ValueError):
        s.summarize("نص.", strategy="segmentwise")


def test_lead_skips_overlong_sentence_instead_of_stopping_short():
    from arabic_summ.summarize import extractive

    wc = [30, 60, 20, 20, 20]          # target 70, hard max 80: sentence 1 (60) would overflow
    assert extractive.lead(wc, 70) == [0]                          # old behaviour without a hard cap
    assert extractive.lead(wc, 70, max_words=80) == [0, 2, 3]       # 70 words, inside bounds


def test_trim_to_words_fills_to_minimum_at_word_boundary():
    from arabic_summ.summarize.postprocess import trim_to_words
    from arabic_summ.text.words import count_words

    sents = ["أ ب ج د", "ه و ز ح ط ي ك ل م ن"]
    out = trim_to_words(sents, max_words=8, min_words=6)
    assert out[0] == "أ ب ج د" and count_words(" ".join(out)) == 6
    assert trim_to_words(sents, max_words=8) == ["أ ب ج د"]


def test_lead_and_extractive_respect_length_bounds_on_short_sentence_texts():
    from arabic_summ.config import load_config
    from arabic_summ.summarize.pipeline import Summarizer

    cfg = load_config("configs/default.yaml")
    long = " ".join(["كلمة"] * 70) + "."
    short = "جملة قصيرة هنا للتجربة فقط."
    text = " ".join([short, long] + [short] * 60)
    sm = Summarizer(cfg)
    for st in ("lead", "extractive"):
        r = sm.summarize(text, strategy=st)
        assert r.length_ok, (st, r.output_words, r.bounds)


# ---------------------------------------------------------------- length fill / unfinished outputs

class _TerseBackend:
    """Ignores min_new_tokens and writes one short sentence per call, like a model trained on
    one-sentence highlights; `cut` appends a fragment without final punctuation."""

    name = "terse"
    max_input_tokens = 1024

    def __init__(self, words=6, cut=False):
        self.words, self.cut = words, cut

    def count_tokens(self, text):
        return len(text.split())

    def generate(self, requests, decoding):
        out = []
        for r in requests:
            w = r.text.split()
            s = " ".join(w[: self.words]).rstrip(".") + "."
            out.append(s + (" " + " ".join(w[self.words : self.words + 4]).rstrip(".") if self.cut else ""))
        return out


def _cfg(**lc):
    return {**CFG, "pipeline": {**CFG["pipeline"], "length_control": {**CFG["pipeline"]["length_control"], **lc}}}


def test_drop_unfinished():
    from arabic_summ.summarize.postprocess import drop_unfinished

    assert drop_unfinished("جملة كاملة. وجملة لم") == "جملة كاملة."
    assert drop_unfinished("قال: «انتهى الأمر.» ثم") == "قال: «انتهى الأمر.»"
    assert drop_unfinished("جملة كاملة؟") == "جملة كاملة؟"
    assert drop_unfinished("لا توجد علامة نهاية") == "لا توجد علامة نهاية"   # nothing complete: keep
    assert drop_unfinished("") == ""


@pytest.mark.parametrize("strategy", ["direct", "segmentwise", "hybrid"])
def test_fill_reaches_minimum_when_model_writes_too_little(strategy):
    text = doc(40, 15, seed=4)  # 600 words -> bounds around 120
    off = Summarizer(_cfg(fill="none"), backend=_TerseBackend()).summarize(text, ratio=0.2, strategy=strategy)
    assert not off.length_ok and off.output_words < off.bounds[0] and off.filled_words == 0
    on = Summarizer(_cfg(fill="extractive", fill_to="lo"), backend=_TerseBackend()).summarize(text, ratio=0.2, strategy=strategy)
    assert on.length_ok and on.filled_words > 0
    assert on.summary.startswith(off.summary.split(".")[0])            # model sentences stay first
    tgt = Summarizer(_cfg(fill="extractive", fill_to="target"), backend=_TerseBackend()).summarize(text, ratio=0.2, strategy=strategy)
    assert tgt.length_ok and tgt.output_words >= min(tgt.target_words, on.output_words)


def test_fill_does_not_repeat_covered_sentences():
    text = doc(40, 15, seed=5)
    r = Summarizer(_cfg(fill="extractive", fill_to="target"), backend=EchoBackend()).summarize(text, strategy="segmentwise")
    sents = [s.strip() for s in r.summary.split(".") if s.strip()]
    assert len(sents) == len(set(sents))


def test_unfinished_fragments_are_dropped_before_aggregation():
    text = doc(40, 15, seed=6)
    keep = Summarizer(_cfg(drop_unfinished=False, fill="none"), backend=_TerseBackend(cut=True)).summarize(text, strategy="segmentwise")
    drop = Summarizer(_cfg(drop_unfinished=True, fill="none"), backend=_TerseBackend(cut=True)).summarize(text, strategy="segmentwise")
    assert not keep.summary.rstrip().endswith(".") or keep.output_words > drop.output_words
    assert drop.summary.rstrip().endswith(".")


def test_fill_settings_are_validated():
    with pytest.raises(ValueError):
        Summarizer(_cfg(fill="abstractive"), backend=EchoBackend())


def test_parts_keep_raw_model_outputs():
    r = Summarizer(_cfg(drop_unfinished=True, fill="none"), backend=_TerseBackend(cut=True)).summarize(
        doc(40, 15, seed=7), strategy="segmentwise")
    assert r.parts and all(not p["output"].endswith(".") for p in r.parts)   # fragment kept in parts
    assert r.summary.rstrip().endswith(".")                                  # but not in the summary


@pytest.mark.parametrize("strategy", ["segmentwise", "hybrid"])
def test_refinish_reproduces_summarize_from_stored_parts(strategy):
    text = doc(40, 15, seed=8)
    for lc in ({"fill": "none", "drop_unfinished": False}, {"fill": "extractive", "fill_to": "target", "drop_unfinished": True}):
        sm = Summarizer(_cfg(**lc), backend=_TerseBackend(cut=True))
        r = sm.summarize(text, strategy=strategy)
        again = sm.refinish(text, [p["output"] for p in r.parts], strategy=strategy)
        assert (again.summary, again.output_words, again.filled_words) == (r.summary, r.output_words, r.filled_words)


def test_dev_tune_scores_every_post_variant_from_stored_parts():
    from arabic_summ.eval.dev_tune import POST_VARIANTS, score_variants

    sm = Summarizer(_cfg(fill="none"), backend=_TerseBackend(cut=True))
    texts = {f"D{k}": doc(40, 15, seed=20 + k) for k in range(3)}
    recs = []
    for sid, t in texts.items():
        r = sm.summarize(t, strategy="segmentwise").to_dict()
        recs.append({**r, "sample_id": sid})
    refs = {sid: " ".join(t.split()[:120]) for sid, t in texts.items()}
    rows = score_variants(CFG, {"seg": recs}, texts, refs)
    by = {r["post"]: r for r in rows}
    assert set(by) == set(POST_VARIANTS) and all(r["n"] == 3 for r in rows)
    assert by["as_before"]["length_ok"] < 1 and by["fill_lo"]["length_ok"] == 1 and by["fill_target"]["length_ok"] == 1
    assert by["as_before"]["filled_share"] == 0 < by["fill_lo"]["filled_share"] <= by["fill_target"]["filled_share"]


def test_dev_tune_faith_variants_report_rejections():
    from arabic_summ.eval.dev_tune import FAITH_VARIANTS, score_variants

    sm = Summarizer(_cfg(fill="none"), backend=_TerseBackend(cut=True))
    texts = {f"D{k}": doc(40, 15, seed=30 + k) for k in range(2)}
    recs = [{**sm.summarize(t, strategy="segmentwise").to_dict(), "sample_id": sid} for sid, t in texts.items()]
    refs = {sid: " ".join(t.split()[:120]) for sid, t in texts.items()}
    rows = score_variants(CFG, {"seg": recs}, texts, refs, FAITH_VARIANTS)
    assert {r["post"] for r in rows} == set(FAITH_VARIANTS) and all("rejected_per_text" in r for r in rows)
    assert next(r for r in rows if r["post"] == "no_filter")["rejected_per_text"] == 0
