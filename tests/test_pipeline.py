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
