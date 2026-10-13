import pytest

from arabic_summ.config import load_config
from arabic_summ.text.normalize import (
    MATCHING_POLICY, MODEL_POLICY, NormalizationPolicy, arabic_letter_ratio, normalize,
    normalize_for_matching, normalize_for_model, policies_from_config,
)
from arabic_summ.text.words import count_words


def test_config_policies_match_code_defaults():
    model, matching = policies_from_config(load_config("configs/default.yaml"))
    assert model == MODEL_POLICY
    assert matching == MATCHING_POLICY


def test_strips_diacritics_including_shadda_and_dagger_alef():
    assert normalize_for_model("كَتَبَ مُحَمَّدٌ هٰذا") == "كتب محمد هذا"


def test_keep_diacritics_policy():
    p = NormalizationPolicy(diacritics="keep")
    assert normalize("كَتَبَ", p) == "كَتَبَ"


def test_removes_tatweel():
    assert normalize_for_model("الســـــلام") == "السلام"


def test_model_policy_preserves_spelling():
    s = "أحمد إلى مدرسة آمنة"
    assert normalize_for_model(s) == s  # alef / maqsura / teh marbuta untouched


def test_matching_policy_folds_letters():
    assert normalize_for_matching("أحمد إلى مدرسة آمنة") == "احمد الي مدرسه امنه"


def test_presentation_forms_and_lam_alef_ligature():
    # U+FEFB is the isolated lam-alef ligature, U+FE8D isolated alef
    assert normalize_for_model("ﻻﺍ") == "لاا"


def test_persian_keyboard_variants_folded():
    assert normalize_for_model("کتاب علی") == "كتاب علي"


def test_control_and_bidi_chars_removed():
    s = "‏مرحبا‎ ﻿بكم‍‫"
    assert normalize_for_model(s) == "مرحبا بكم"


def test_whitespace_keeps_paragraphs():
    s = "سطر  أول\t  \r\n\r\n\r\n\r\nسطر   ثان  "
    assert normalize_for_model(s) == "سطر أول\n\nسطر ثان"


def test_digits_policy():
    assert normalize_for_model("عام ٢٠٢٤") == "عام ٢٠٢٤"
    assert normalize_for_matching("عام ٢٠٢٤ و۱۵") == "عام 2024 و15"


def test_matching_removes_punctuation_without_merging_words():
    assert normalize_for_matching("قال،الوزير: «نعم»؟") == "قال الوزير نعم"


def test_matching_is_idempotent():
    s = "قَالَ الوزيرُ إنَّ نسبةَ النموِّ بلغت ٣٫٥٪ في عام ٢٠٢٣."
    once = normalize_for_matching(s)
    assert normalize_for_matching(once) == once


def test_empty_input():
    assert normalize_for_model("") == ""
    assert normalize_for_matching("   ") == ""


def test_unknown_option_rejected():
    with pytest.raises(ValueError):
        NormalizationPolicy.from_dict({"strip_everything": True})


def test_arabic_ratio():
    assert arabic_letter_ratio("مرحبا") == 1.0
    assert arabic_letter_ratio("hello") == 0.0
    assert 0.4 < arabic_letter_ratio("مرحبا hello") < 0.6


def test_word_count_ignores_punctuation_tokens():
    assert count_words("قال الوزير - في بيان ، إن النسبة 10% .") == 7
