"""External (non-XL-Sum) training pairs for the style fine-tune (03b, PROGRESS 31).

Filters a public article/summary dataset to pairs that look like our task (Arabic, benchmark-like
input length, multi-sentence summary, every summary number in the article, limited unsupported
content) and drops any article that shares text with the benchmark or dev texts.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import pandas as pd

from ..eval import rouge
from ..text.normalize import arabic_letter_ratio
from ..text.words import count_words
from . import data as D


def ngram_overlap(texts: Iterable[str], protected: Mapping[str, str], n: int = 8, step: int = 4,
                  min_hits: int = 5) -> list[set[str]]:
    """For each text, the ids of `protected` texts it shares at least `min_hits` word `n`-grams with
    (sampled every `step` positions of the text; matching-normalized tokens)."""
    index: dict[tuple, set[str]] = {}
    for pid, t in protected.items():
        toks = rouge.tokenize(t)
        for i in range(len(toks) - n + 1):
            index.setdefault(tuple(toks[i:i + n]), set()).add(pid)
    out = []
    for t in texts:
        toks, hits = rouge.tokenize(t), {}
        for i in range(0, len(toks) - n + 1, step):
            for pid in index.get(tuple(toks[i:i + n]), ()):
                hits[pid] = hits.get(pid, 0) + 1
        out.append({p for p, c in hits.items() if c >= min_hits})
    return out


def prepare_pairs(df: pd.DataFrame, scfg: Mapping, protected: Mapping[str, str]) -> tuple[pd.DataFrame, dict]:
    """Columns `scfg.dataset.text_col` / `summary_col` -> filtered DataFrame[text, summary] + report."""
    ds = scfg["dataset"]
    x = pd.DataFrame({"text": df[ds["text_col"]].astype(str).str.strip(),
                      "summary": df[ds["summary_col"]].astype(str).str.strip()})
    rep = {"rows_in": int(len(x))}
    x = x[(x.text != "") & (x.summary != "") & (x.text != "nan") & (x.summary != "nan")]
    x = x.drop_duplicates("text")
    rep["after_empty_and_duplicate"] = int(len(x))
    sw, mw = x.text.map(count_words), x.summary.map(count_words)
    x = x[sw.between(scfg["min_source_words"], scfg["max_source_words"]) & (mw >= scfg["min_summary_words"])]
    rep["after_length"] = int(len(x))
    ar = x.text.map(arabic_letter_ratio).ge(scfg["min_arabic_ratio"]) & x.summary.map(arabic_letter_ratio).ge(scfg["min_arabic_ratio"])
    x = x[ar]
    rep["after_arabic_ratio"] = int(len(x))
    leak = ngram_overlap(x.text, protected)
    leaked = [bool(s) for s in leak]
    rep["leaked_rows"] = int(sum(leaked))
    rep["leaked_protected_ids"] = sorted(set().union(*leak)) if leak else []
    x = x[[not b for b in leaked]]
    kept, frep = D.filter_pairs(x, scfg["filter"])
    rep["faithfulness"] = frep
    rep["rows_kept"] = int(len(kept))
    return kept.reset_index(drop=True), rep
