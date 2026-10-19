import numpy as np

from arabic_summ.config import load_config
from arabic_summ.summarize.faithful import FaithfulnessChecker, filter_sentences
from arabic_summ.summarize.pipeline import Summarizer

CFG = load_config("configs/default.yaml")
FC = {**CFG["pipeline"]["faithfulness"], "enabled": True}

SOURCE = [
    "أعلنت وزارة الصحة تسجيل 140 ألف حالة تطعيم للأطفال خلال أسبوعين في المحافظات الشمالية.",
    "وقال مدير البرنامج الوطني سامي الخطيب إن الفرق الطبية ستواصل العمل حتى نهاية الشهر.",
    "ويحتمي ما بين 500,000 و600,000 شخص في المخيمات القريبة من الحدود.",
    "وقال المتحدث باسم الجيش العقيد مازن رفعت إن القتلى ليسوا من الجنود.",
    "وقالت الطبيبة هالة يوسف: \"الوضع الصحي في المخيمات صعب للغاية\".",
    "وفاز المنتخب الوطني على منتخب الضيوف بهدف دون رد في المباراة الافتتاحية.",
]


def reasons(sentence):
    return FaithfulnessChecker(SOURCE, FC).reasons(sentence)


def test_copied_and_shortened_sentences_pass():
    assert reasons(SOURCE[0]) == []
    assert reasons("قال مدير البرنامج الوطني سامي الخطيب إن الفرق الطبية ستواصل العمل.") == []
    assert reasons("تقول الطبيبة هالة يوسف إن \"الوضع الصحي في المخيمات صعب للغاية\".") == []


def test_number_moved_to_another_fact_is_rejected():
    r = reasons("أعلنت وزارة الصحة تطعيم ما بين 500,000 و600,000 طفل خلال أسبوعين في المحافظات الشمالية.")
    assert "number_context" in r


def test_number_not_in_source_is_rejected():
    assert "number_unsupported" in reasons("أعلنت وزارة الصحة تسجيل 2015 حالة تطعيم للأطفال.")


def test_lost_negation_is_rejected():
    assert "negation" in reasons("قال المتحدث باسم الجيش العقيد مازن رفعت إن القتلى من الجنود.")
    assert reasons("قال المتحدث باسم الجيش العقيد مازن رفعت إن القتلى ليسوا من الجنود.") == []


def test_continuity_la_yazal_is_not_a_negation():
    src = ["لا يزال الوضع الصحي في المخيمات صعبا بحسب الأطباء."]
    assert FaithfulnessChecker(src, FC).reasons("لا يزال الوضع الصحي في المخيمات صعبا.") == []


def test_invented_quote_and_wrong_speaker_are_rejected():
    assert "quote" in reasons("وقالت الطبيبة هالة يوسف: \"المساعدات وصلت إلى الجميع\".")
    assert "attribution" in reasons("يقول مدير البرنامج الوطني سامي الخطيب إن الوضع الصحي في المخيمات صعب للغاية.")


def test_unsupported_content_is_rejected():
    assert "support" in reasons("تعادل المنتخب الوطني مع منتخب الضيوف في الشوط الثاني بعد مباراة مثيرة.")


def test_thresholds_can_be_switched_off():
    off = {**FC, "min_support": None, "number_context": None, "negation": False, "quotes": None, "attribution": None}
    s = "قال المتحدث باسم الجيش العقيد مازن رفعت إن القتلى من الجنود في 2015."
    assert FaithfulnessChecker(SOURCE, off).reasons(s) == []


def test_filter_sentences_splits_kept_and_rejected():
    kept, rej = filter_sentences([SOURCE[0], "أعلنت وزارة الصحة تسجيل 2015 حالة تطعيم للأطفال."], SOURCE, FC)
    assert kept == [SOURCE[0]] and len(rej) == 1 and rej[0]["reasons"] == ["number_unsupported"]


class _FlipBackend:
    """Writes one sentence per call that drops the negation of the source's 4th sentence."""

    name = "flip"
    max_input_tokens = 1024

    def count_tokens(self, text):
        return len(text.split())

    def generate(self, requests, decoding):
        return ["قال المتحدث باسم الجيش العقيد مازن رفعت إن القتلى من الجنود. " + SOURCE[1] for _ in requests]


def _filler(n, seed=0):
    rng = np.random.default_rng(seed)
    letters = list("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")
    return [" ".join("".join(rng.choice(letters, size=rng.integers(3, 7))) for _ in range(14)) + "." for _ in range(n)]


def _cfg(enabled, **lc):
    p = CFG["pipeline"]
    return {**CFG, "pipeline": {**p, "faithfulness": {**p["faithfulness"], "enabled": enabled},
                                "length_control": {**p["length_control"], **lc}}}


def test_pipeline_drops_rejected_sentences_and_fills_from_source():
    text = " ".join(SOURCE + _filler(30))
    off = Summarizer(_cfg(False), backend=_FlipBackend()).summarize(text, strategy="direct")
    on = Summarizer(_cfg(True), backend=_FlipBackend()).summarize(text, strategy="direct")
    assert "إن القتلى من الجنود." in off.summary and off.rejected == []
    assert "إن القتلى من الجنود." not in on.summary and on.rejected and on.rejected[0]["reasons"] == ["negation"]
    assert on.length_ok and on.filled_words > 0
    again = Summarizer(_cfg(True), backend=_FlipBackend()).refinish(text, [p["output"] for p in on.parts], strategy="direct")
    assert (again.summary, again.rejected) == (on.summary, on.rejected)


def test_fill_skips_titles_and_prefers_numbers():
    title = "تقرير جديد عن الصحة في المخيمات"
    text = "\n".join([title] + SOURCE + _filler(30, seed=1))
    sm = Summarizer(_cfg(False, fill_skip_titles=True, fill_number_bonus=5.0), backend=_FlipBackend())
    r = sm.summarize(text, strategy="direct")
    assert title not in r.summary
    filled = r.summary.split(SOURCE[1], 1)[1]
    assert any(ch.isdigit() for ch in filled)


def test_negation_window_stops_at_punctuation():
    src = ["وتقديمها دون تأخير لا داعي له، تدعم منظمة الصحة العالمية الخدمات الخاصة بسرطان الثدي في الخليل وجنين."]
    assert FaithfulnessChecker(src, FC).reasons("تدعم منظمة الصحة العالمية الخدمات الخاصة بسرطان الثدي في الخليل وجنين.") == []


def test_speaker_named_nearby_and_described_earlier_passes():
    src = ["قال الخبير الاقتصادي البريطاني جيرارد ليونز إن الأسواق مستقرة.",
           "وارتفعت أسعار الأسهم في بورصة لندن خلال الأسبوع الماضي.",
           "وتراجع الجنيه الإسترليني أمام الدولار بعد صدور البيانات.",
           "وتوقعت البنوك المركزية زيادة أسعار الفائدة خلال العام المقبل.",
           "وتأثرت أسواق الطاقة بقرارات الدول المنتجة للنفط.",
           "ومع ذلك، قال ليونز إن \"المدن الأوروبية ستجد صعوبة في إزاحة لندن كمركز مالي\".",
           "وقال آلان غرينسبان، الرئيس السابق للبنك المركزي الأمريكي، إن التأثير سيكون على القضايا المالية.",
           "وقال غرينسبان إن لندن تمثل مركزا ماليا أساسيا لا غنى عنه."]
    chk = FaithfulnessChecker(src, FC)
    assert chk.reasons("يقول الخبير الاقتصادي البريطاني جيرارد ليونز، إن \"المدن الأوروبية ستجد صعوبة في إزاحة لندن كمركز مالي\".") == []
    assert "attribution" in chk.reasons("يقول رئيس مجموعة وزراء مالية أوروبا إن لندن تمثل مركزا ماليا أساسيا لا غنى عنه.")
    off = FaithfulnessChecker(src, {**FC, "attribution_name_match": False})
    assert "attribution" in off.reasons("يقول الخبير الاقتصادي البريطاني جيرارد ليونز، إن \"المدن الأوروبية ستجد صعوبة في إزاحة لندن كمركز مالي\".")
