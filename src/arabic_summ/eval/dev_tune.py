"""Dev-set tuning of the pipeline's post-processing (never the 240 benchmark texts).

Generation runs (`summarize-batch` with `keep_parts: true`) are made once per strategy / generation
setting; `score_variants` then re-applies each post-processing variant to the stored raw outputs
via `Summarizer.refinish` and scores it against the dev references, so length-fill settings are
compared without regenerating.
"""

from __future__ import annotations

import statistics as st
from typing import Iterable, Mapping, Sequence

from ..config import load_config  # noqa: F401  (re-exported for notebook use)
from ..summarize.pipeline import Summarizer
from . import facts, rouge

POST_VARIANTS = {
    "as_before": {"drop_unfinished": False, "fill": "none"},
    "drop_only": {"drop_unfinished": True, "fill": "none"},
    "fill_lo": {"drop_unfinished": True, "fill": "extractive", "fill_to": "lo"},
    "fill_target": {"drop_unfinished": True, "fill": "extractive", "fill_to": "target"},
}


def _with_length_control(cfg: Mapping, lc: Mapping) -> dict:
    p = dict(cfg["pipeline"])
    p["length_control"] = {**p["length_control"], **lc}
    return {**cfg, "pipeline": p}


def score_variants(cfg: Mapping, runs: Mapping[str, Sequence[Mapping]], texts: Mapping[str, str],
                   refs: Mapping[str, str], variants: Mapping[str, Mapping] = POST_VARIANTS) -> list[dict]:
    """`runs`: run name -> records from summarize-batch (with `parts`, `strategy`, `ratio`).
    Returns one row per (run, post-processing variant)."""
    rows = []
    for run, recs in runs.items():
        recs = [r for r in recs if r["sample_id"] in refs]
        for vname, lc in variants.items():
            sm = Summarizer(_with_length_control(cfg, lc))
            outs = [sm.refinish(texts[r["sample_id"]], [p["output"] for p in r["parts"]], r["ratio"], r["strategy"])
                    for r in recs]
            rows.append(score_row(run, vname, [r["sample_id"] for r in recs], [o.summary for o in outs],
                                  [o.length_ok for o in outs], [o.filled_words / max(1, o.output_words) for o in outs],
                                  texts, refs))
    return rows


def score_row(run: str, variant: str, ids: Sequence[str], hyps: Sequence[str], length_ok: Iterable[bool],
              filled_share: Iterable[float], texts: Mapping[str, str], refs: Mapping[str, str]) -> dict:
    r = [rouge.score(refs[i], h) for i, h in zip(ids, hyps)]
    xl = [rouge.score(refs[i], h, stem="snowball")["rouge1"] for i, h in zip(ids, hyps)]
    f = facts.aggregate(facts.check(h, texts[i]) for i, h in zip(ids, hyps))
    return {"run": run, "post": variant, "n": len(ids),
            "rouge1": round(st.mean(x["rouge1"] for x in r), 4), "rouge2": round(st.mean(x["rouge2"] for x in r), 4),
            "rougeL": round(st.mean(x["rougeL"] for x in r), 4), "rouge1_xlsum": round(st.mean(xl), 4),
            "length_ok": round(st.mean(length_ok), 3), "filled_share": round(st.mean(filled_share), 3),
            "numbers_ok": round(f["numbers_ok_share"], 3), "novel_rate": round(f["novel_rate"], 3)}
