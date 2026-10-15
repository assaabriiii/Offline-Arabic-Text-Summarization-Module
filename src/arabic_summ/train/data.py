"""Faithfulness filtering of XL-Sum training pairs and tokenization for seq2seq training.

XL-Sum highlights are written by BBC editors and often contain facts that are not in the article
body (02_preprocess finding). Training on them teaches the model to add facts, so pairs whose
highlight has a number missing from the article, or too many content words with no support in
the article, are removed before fine-tuning. Validation and test stay untouched (comparability).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from ..eval.facts import SourceIndex
from ..text.normalize import NormalizationPolicy, normalize


def faithfulness_table(df: pd.DataFrame, text_col: str = "text", summary_col: str = "summary") -> pd.DataFrame:
    rows = []
    for text, summ in zip(df[text_col], df[summary_col]):
        r = SourceIndex(text).check(summ)
        rows.append({"n_numbers": r.n_numbers, "numbers_ok": r.numbers_ok,
                     "n_content": r.n_content, "novel_rate": r.novel_rate})
    return pd.DataFrame(rows, index=df.index)


def filter_pairs(df: pd.DataFrame, fcfg: Mapping, table: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Keep pairs whose summary numbers all occur in the article and whose novel-content rate is
    <= max_novel_rate. If that keeps fewer than min_keep_fraction of the pairs, the novel-rate
    threshold is raised to the quantile that keeps exactly that fraction (among numbers_ok pairs)
    and the relaxation is reported."""
    t = faithfulness_table(df) if table is None else table
    ok_num = t.numbers_ok if fcfg.get("require_numbers_supported", True) else pd.Series(True, index=t.index)
    thr = float(fcfg["max_novel_rate"])
    min_keep = int(np.ceil(float(fcfg.get("min_keep_fraction", 0.0)) * len(df)))
    keep = ok_num & (t.novel_rate <= thr)
    relaxed = False
    if keep.sum() < min_keep and ok_num.sum() >= min_keep:
        thr = float(np.sort(t.novel_rate[ok_num].to_numpy())[min_keep - 1])
        keep = ok_num & (t.novel_rate <= thr)
        relaxed = True
    report = {
        "rows_in": int(len(df)), "rows_kept": int(keep.sum()), "kept_share": round(float(keep.mean()), 4),
        "dropped_numbers": int((~ok_num).sum()),
        "dropped_novel_only": int((ok_num & ~keep).sum()),
        "max_novel_rate_config": float(fcfg["max_novel_rate"]), "max_novel_rate_used": round(thr, 4),
        "relaxed_to_min_keep": relaxed,
        "novel_rate_quantiles": {str(q): round(float(t.novel_rate.quantile(q)), 4) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "numbers_ok_share": round(float(t.numbers_ok.mean()), 4),
    }
    return df[keep], report


def tokenize_pairs(df: pd.DataFrame, tokenizer, policy: NormalizationPolicy, max_source: int, max_target: int,
                   text_col: str = "text", summary_col: str = "summary") -> list[dict]:
    """List of {input_ids, attention_mask, labels} (no padding; the collator pads per batch)."""
    src = [normalize(t, policy) for t in df[text_col]]
    tgt = [normalize(t, policy) for t in df[summary_col]]
    enc = tokenizer(src, max_length=max_source, truncation=True)
    lab = tokenizer(text_target=tgt, max_length=max_target, truncation=True)
    return [{"input_ids": i, "attention_mask": a, "labels": l}
            for i, a, l in zip(enc["input_ids"], enc["attention_mask"], lab["input_ids"])]


def sample_fraction(df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if fraction >= 1:
        return df
    return df.sample(frac=fraction, random_state=seed).sort_index()
