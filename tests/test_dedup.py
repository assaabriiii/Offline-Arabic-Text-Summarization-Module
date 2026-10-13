import numpy as np

from arabic_summ.data.dedup import Doc, NearDupIndex, deduplicate, shingles
from arabic_summ.text.normalize import normalize_for_matching

BASE = (
    "أعلنت وزارة الصحة اليوم عن افتتاح مستشفى جديد في مدينة الرياض بسعة ثلاثمئة سرير "
    "وقال الوزير إن المشروع يهدف إلى تحسين الخدمات الصحية في المنطقة الوسطى وتقليل فترات الانتظار "
    "وأضاف أن العمل في المستشفى سيبدأ مطلع العام المقبل بعد استكمال تجهيز الأقسام وتعيين الكوادر الطبية "
    "وأشار إلى أن تكلفة المشروع بلغت نحو خمسمئة مليون ريال تم تمويلها من ميزانية الدولة"
)
OTHER = (
    "شهدت أسواق الأسهم الخليجية ارتفاعا ملحوظا في تعاملات الأسبوع الماضي مدعومة بصعود أسعار النفط "
    "وقال محللون إن المستثمرين يترقبون نتائج الشركات الكبرى للربع الثالث وسط توقعات بنمو الأرباح "
    "في حين تراجعت بعض الأسهم القيادية في قطاع العقارات بسبب مخاوف من ارتفاع أسعار الفائدة"
)


def n(t):
    return normalize_for_matching(t)


def run(docs):
    return deduplicate(docs, ("test", "validation", "train"), {"train"}, k=5, num_perm=128, threshold=0.8)


def test_shingles():
    assert shingles("a b c", 5) == {"a b c"}
    assert shingles("a b c d e f", 5) == {"a b c d e", "b c d e f"}
    assert shingles("", 5) == set()


def test_exact_duplicate_after_normalization_is_removed_from_train():
    variant = BASE.replace("أعلنت", "اَعلنت").replace(" ", "  ") + " ."  # diacritic/space/punct noise
    res = run([Doc("train:1", "train", n(variant)), Doc("test:1", "test", n(BASE))])
    assert set(res.removed) == {"train:1"}
    assert res.removed["train:1"].reason == "exact"
    assert res.removed["train:1"].other == "test:1"  # eval doc wins regardless of input order


def test_near_duplicate_detected():
    near = BASE.replace("ثلاثمئة سرير", "ثلاثمئة وخمسين سريرا")
    res = run([Doc("test:1", "test", n(BASE)), Doc("train:7", "train", n(near))])
    assert "train:7" in res.removed
    assert res.removed["train:7"].reason == "near"
    assert res.removed["train:7"].similarity >= 0.8


def test_distinct_docs_kept():
    res = run([Doc("train:1", "train", n(BASE)), Doc("train:2", "train", n(OTHER))])
    assert not res.removed and not res.flagged


def test_within_train_keeps_first():
    res = run([Doc("train:1", "train", n(BASE)), Doc("train:2", "train", n(BASE))])
    assert set(res.removed) == {"train:2"}


def test_eval_duplicates_are_flagged_not_removed():
    res = run([Doc("test:1", "test", n(BASE)), Doc("validation:1", "validation", n(BASE))])
    assert not res.removed
    assert res.flagged["validation:1"].other == "test:1"


def test_near_dup_index_query():
    keys = ["train:1", "train:2"]
    res = run([Doc(k, "train", n(t)) for k, t in zip(keys, (BASE, OTHER))])
    idx = NearDupIndex(keys, np.stack([res.signatures[k] for k in keys]), threshold=0.8, k=5)
    hits = idx.query(n(BASE.replace("الرياض", "جدة")))
    assert hits and hits[0][0] == "train:1"
    assert idx.query(n("نص مختلف تماما عن كل ما سبق ولا علاقة له بالموضوع إطلاقا")) == []
