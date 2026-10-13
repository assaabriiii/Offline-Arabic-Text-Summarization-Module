"""Arabic text normalization.

Two policies are used across the project (both configured under `normalization:`):

* ``model``    – light, spelling-preserving cleanup applied to model input/output.
* ``matching`` – aggressive folding used ONLY for hashing, deduplication, ROUGE
  tokenization and fact matching, never for text shown to users.

Order of operations (fixed, documented so results are reproducible):
  1. Unicode normalization (NFKC folds presentation forms and lam-alef ligatures)
  2. newline unification (\\r\\n, \\r -> \\n)
  3. removal of control/format/private-use/unassigned chars (keeps \\n and \\t)
  4. tatweel removal
  5. diacritics removal (harakat, shadda, sukun, tanwin, superscript alef, Quranic marks)
  6. letter folding (Persian variants, alef, alef maqsura, teh marbuta)
  7. digit folding (Arabic-Indic / Extended Arabic-Indic -> ASCII)
  8. punctuation/symbol removal (matching only)
  9. Latin lowercasing (matching only)
 10. whitespace collapsing (paragraph breaks kept by the model policy)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, fields
from typing import Mapping

import pyarabic.araby as araby

# Marks not covered by pyarabic's TASHKEEL (fatha..sukun, tanwin, shadda).
_EXTRA_MARKS = (
    "ٰ"            # superscript (dagger) alef
    "ؐ-ؚ"     # Quranic honorific / small high marks
    "ۖ-ۜ"     # Quranic small high ligatures
    "۟-ۤ"     # Quranic small marks
    "ۧۨ"      # small high yeh / noon
    "۪-ۭ"     # Quranic small low marks
    "࣓-ࣣ࣡-ࣿ"  # Arabic Extended-A marks (U+08E2 is a format char)
)
_DIACRITICS_RE = re.compile("[" + "".join(araby.TASHKEEL) + _EXTRA_MARKS + "]")

_ALEF_VARIANTS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
_PERSIAN_VARIANTS = str.maketrans({"ی": "ي", "ک": "ك", "ۀ": "ة", "ە": "ه"})
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "0123456789" * 2)

_KEEP_CONTROL = {"\n", "\t"}
_DROP_CATEGORIES = {"Cc", "Cf", "Co", "Cn"}

_HSPACE_RE = re.compile(r"[^\S\n]+")      # any whitespace except newline
_MULTI_NL_RE = re.compile(r"\n{3,}")
_ANY_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class NormalizationPolicy:
    unicode_form: str | None = "NFKC"
    remove_tatweel: bool = True
    diacritics: str = "strip"            # strip | keep
    fold_persian_variants: bool = True   # ی ک -> ي ك (keyboard-encoding variants, not spelling)
    unify_alef: bool = False
    alef_maqsura_to_yeh: bool = False
    teh_marbuta_to_heh: bool = False
    digits: str = "keep"                 # keep | ascii
    remove_control_chars: bool = True
    collapse_whitespace: bool = True
    remove_punctuation: bool = False
    lowercase_latin: bool = False

    @classmethod
    def from_dict(cls, d: Mapping | None) -> "NormalizationPolicy":
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown normalization options: {sorted(unknown)}")
        if d.get("diacritics", "strip") not in ("strip", "keep"):
            raise ValueError("diacritics must be 'strip' or 'keep'")
        if d.get("digits", "keep") not in ("keep", "ascii"):
            raise ValueError("digits must be 'keep' or 'ascii'")
        return cls(**d)


MODEL_POLICY = NormalizationPolicy()
MATCHING_POLICY = NormalizationPolicy(
    unify_alef=True, alef_maqsura_to_yeh=True, teh_marbuta_to_heh=True, digits="ascii",
    remove_punctuation=True, lowercase_latin=True,
)


def policies_from_config(cfg: Mapping) -> tuple[NormalizationPolicy, NormalizationPolicy]:
    n = cfg.get("normalization", {})
    return NormalizationPolicy.from_dict(n.get("model")), NormalizationPolicy.from_dict(n.get("matching"))


def _strip_control(text: str) -> str:
    return "".join(
        ch for ch in text if ch in _KEEP_CONTROL or unicodedata.category(ch) not in _DROP_CATEGORIES
    )


def _strip_punct(text: str) -> str:
    # Punctuation (P*) and symbols (S*) become spaces so adjacent words don't merge.
    return "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)


def normalize(text: str, policy: NormalizationPolicy = MODEL_POLICY) -> str:
    if not text:
        return ""
    if policy.unicode_form:
        text = unicodedata.normalize(policy.unicode_form, text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if policy.remove_control_chars:
        text = _strip_control(text)
    if policy.remove_tatweel:
        text = araby.strip_tatweel(text)
    if policy.diacritics == "strip":
        text = _DIACRITICS_RE.sub("", text)
    if policy.fold_persian_variants:
        text = text.translate(_PERSIAN_VARIANTS)
    if policy.unify_alef:
        text = text.translate(_ALEF_VARIANTS)
    if policy.alef_maqsura_to_yeh:
        text = text.replace("ى", "ي")
    if policy.teh_marbuta_to_heh:
        text = text.replace("ة", "ه")
    if policy.digits == "ascii":
        text = text.translate(_DIGITS)
    if policy.remove_punctuation:
        text = _strip_punct(text)
    if policy.lowercase_latin:
        text = text.lower()
    if policy.collapse_whitespace:
        if policy.remove_punctuation:
            # matching mode: one flat token stream
            text = _ANY_WS_RE.sub(" ", text)
        else:
            text = _HSPACE_RE.sub(" ", text)
            text = "\n".join(line.strip() for line in text.split("\n"))
            text = _MULTI_NL_RE.sub("\n\n", text)
        text = text.strip()
    return text


def normalize_for_model(text: str, policy: NormalizationPolicy = MODEL_POLICY) -> str:
    return normalize(text, policy)


def normalize_for_matching(text: str, policy: NormalizationPolicy = MATCHING_POLICY) -> str:
    return normalize(text, policy)


def arabic_letter_ratio(text: str) -> float:
    """Share of alphabetic characters that are Arabic script (0.0 if no letters)."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    arabic = sum(1 for ch in letters if "؀" <= ch <= "ۿ" or "ݐ" <= ch <= "ݿ"
                 or "ࢠ" <= ch <= "ࣿ" or "ﭐ" <= ch <= "﷿" or "ﹰ" <= ch <= "﻿")
    return arabic / len(letters)
