import pytest

from arabic_summ.text.chunk import chunk_sentences, whitespace_tokens
from arabic_summ.text.segment import Sentence, split_paragraphs, split_sentences


def texts(sents):
    return [s.text for s in sents]


def test_basic_arabic_punctuation():
    t = "وصل الوفد اليوم. هل سيبدأ الاجتماع؟ نعم! وقال المتحدث: «سنعلن النتائج قريبا»."
    assert texts(split_sentences(t)) == [
        "وصل الوفد اليوم.", "هل سيبدأ الاجتماع؟", "نعم!", "وقال المتحدث: «سنعلن النتائج قريبا».",
    ]


def test_semicolon_and_comma_do_not_split():
    t = "ارتفعت الأسعار، وتراجع الطلب؛ وقال الخبراء إن ذلك متوقع."
    assert len(split_sentences(t)) == 1


def test_decimals_abbreviations_and_list_numbers():
    t = "بلغ النمو 3.5 في المئة وفق د. أحمد. 1. البند الأول هنا. ٢. البند الثاني."
    assert texts(split_sentences(t)) == [
        "بلغ النمو 3.5 في المئة وفق د. أحمد.", "1. البند الأول هنا.", "٢. البند الثاني.",
    ]


def test_quote_closer_kept_with_sentence():
    t = "قال: «انتهى الأمر.» ثم غادر."
    assert texts(split_sentences(t)) == ["قال: «انتهى الأمر.»", "ثم غادر."]


def test_newlines_and_paragraphs():
    t = "عنوان التقرير\nالفقرة الأولى. جملة ثانية.\n\nالفقرة الثانية بدون نقطة"
    s = split_sentences(t)
    assert texts(s) == ["عنوان التقرير", "الفقرة الأولى.", "جملة ثانية.", "الفقرة الثانية بدون نقطة"]
    assert [x.paragraph for x in s] == [0, 0, 0, 1]
    assert [x.index for x in s] == [0, 1, 2, 3]
    assert len(split_paragraphs(t)) == 2


def test_punctuation_only_lines_dropped():
    assert texts(split_sentences("...\n- \nنص.")) == ["نص."]


# ---------------------------------------------------------------- chunking

def mk(sizes, paragraphs=None):
    paragraphs = paragraphs or [0] * len(sizes)
    return [Sentence(" ".join(f"w{i}_{j}" for j in range(n)), p, i) for i, (n, p) in enumerate(zip(sizes, paragraphs))]


def covered(chunks):
    return sorted({i for c in chunks for i in c.sentence_ids})


def test_single_chunk_when_fits():
    ch = chunk_sentences(mk([10, 20, 30]), whitespace_tokens, 900, 80)
    assert len(ch) == 1 and ch[0].n_tokens == 60 and ch[0].n_overlap_sentences == 0


def test_budget_respected_and_full_coverage():
    sents = mk([100] * 25)
    ch = chunk_sentences(sents, whitespace_tokens, 900, 80)
    assert all(c.n_tokens <= 900 for c in ch)
    assert covered(ch) == list(range(25))


def test_overlap_is_whole_sentences_within_limit():
    sents = mk([50] * 40)
    ch = chunk_sentences(sents, whitespace_tokens, 900, 80, paragraph_break_min_fill=1.1)
    for prev, nxt in zip(ch, ch[1:]):
        shared = set(prev.sentence_ids) & set(nxt.sentence_ids)
        assert len(shared) == nxt.n_overlap_sentences == 1          # one 50-token sentence <= 80
        assert sum(50 for _ in shared) <= 80
        assert nxt.sentence_ids[0] == prev.sentence_ids[-1]          # overlap = tail of previous


def test_no_overlap_when_sentence_bigger_than_overlap():
    ch = chunk_sentences(mk([300] * 6), whitespace_tokens, 900, 80)
    assert all(c.n_overlap_sentences == 0 for c in ch)
    assert covered(ch) == list(range(6))


def test_overlap_dropped_when_it_blocks_progress():
    # 60-token tail would overlap, but 60 + 880 > 900 -> overlap dropped, no overlap-only chunk
    ch = chunk_sentences(mk([840, 60, 880]), whitespace_tokens, 900, 80)
    assert [c.sentence_ids for c in ch] == [(0, 1), (2,)]


def test_long_sentence_split_at_commas_then_words():
    s = [Sentence("، ".join(" ".join(["كلمة"] * 30) for _ in range(5)), 0, 0)]
    ch = chunk_sentences(s, whitespace_tokens, 70, 0)
    assert all(c.n_tokens <= 70 for c in ch) and sum(c.n_tokens for c in ch) == 150
    huge = [Sentence(" ".join(["كلمة"] * 250), 0, 0)]
    ch = chunk_sentences(huge, whitespace_tokens, 100, 10)
    assert all(c.n_tokens <= 100 for c in ch)


def test_prefers_paragraph_boundary_when_full_enough():
    sents = mk([300, 300, 100, 100, 200], paragraphs=[0, 0, 1, 1, 1])  # 1000 > 900
    ch = chunk_sentences(sents, whitespace_tokens, 900, 0, paragraph_break_min_fill=0.6)
    assert ch[0].sentence_ids == (0, 1)


def test_invalid_args():
    with pytest.raises(ValueError):
        chunk_sentences(mk([1]), whitespace_tokens, 100, 100)
