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
# faithfulness filter + fill options (PROGRESS 36); on top of the fill_target post-processing
_FT = POST_VARIANTS["fill_target"]
_ON = {"enabled": True}
FAITH_VARIANTS = {
    "no_filter": {**_FT, "faithfulness": {"enabled": False}},
    "filter": {**_FT, "faithfulness": _ON},
    "filter_s0.7": {**_FT, "faithfulness": {**_ON, "min_support": 0.7}},
    "filter_s0.9": {**_FT, "faithfulness": {**_ON, "min_support": 0.9}},
    "filter_no_attr": {**_FT, "faithfulness": {**_ON, "attribution": None}},
    "filter+fill": {**_FT, "fill_cover_threshold": 0.5, "fill_skip_titles": True, "faithfulness": _ON},
    "filter+fill+num0.3": {**_FT, "fill_cover_threshold": 0.5, "fill_skip_titles": True, "fill_number_bonus": 0.3,
                           "faithfulness": _ON},
    "filter+fill+num0.3+single": {**_FT, "fill_cover_threshold": 0.5, "fill_skip_titles": True,
                                  "fill_number_bonus": 0.3,
                                  "faithfulness": {**_ON, "attribution_single_source": True}},
    "filter+fill+num0.3+single+support1": {**_FT, "fill_cover_threshold": 0.5, "fill_skip_titles": True,
                                           "fill_number_bonus": 0.3,
                                           "faithfulness": {**_ON, "attribution_single_source": True,
                                                            "max_support_sentences": 1}},
    "filter+fill+num0.6": {**_FT, "fill_cover_threshold": 0.5, "fill_skip_titles": True, "fill_number_bonus": 0.6,
                           "faithfulness": _ON},
}
# second round (PROGRESS 37): speaker window, fill cleanup; on top of the PROGRESS 36 setting
_D36 = {**_FT, "fill_cover_threshold": 0.5, "fill_skip_titles": True, "fill_number_bonus": 0.3,
        "fill_skip_questions": False, "fill_fix_quotes": False, "fill_strip_leads": False,
        "fill_strip_leads_extended": False, "fill_skip_glued_headings": False, "fill_skip_first_person": "none",
        "fill_skip_pronoun_speaker": False, "order": "as_written",
        "faithfulness": {**_ON, "max_support_sentences": 1, "attribution_single_source": True,
                         "attribution_window": 2, "attribution_name_match": False,
                         "attribution_latest_speaker": False, "attribution_pronoun": False,
                         "attribution_latest_share": False}}
_CLEAN = {"fill_skip_questions": True, "fill_fix_quotes": True, "fill_strip_leads": True}
ROUND2_VARIANTS = {
    "d36": _D36,
    "win3": {**_D36, "faithfulness": {**_D36["faithfulness"], "attribution_window": 3}},
    "win4": {**_D36, "faithfulness": {**_D36["faithfulness"], "attribution_window": 4}},
    "win_all": {**_D36, "faithfulness": {**_D36["faithfulness"], "attribution_window": None}},
    "no_questions": {**_D36, "fill_skip_questions": True},
    "no_questions+quotes": {**_D36, "fill_skip_questions": True, "fill_fix_quotes": True},
    "clean+name+win3": {**_D36, **_CLEAN, "faithfulness": {**_D36["faithfulness"], "attribution_name_match": True,
                                                         "attribution_window": 3}},
    "clean": {**_D36, **_CLEAN},
    "clean+win3": {**_D36, **_CLEAN, "faithfulness": {**_D36["faithfulness"], "attribution_window": 3}},
    "clean+win4": {**_D36, **_CLEAN, "faithfulness": {**_D36["faithfulness"], "attribution_window": 4}},
    "clean+win_all": {**_D36, **_CLEAN, "faithfulness": {**_D36["faithfulness"], "attribution_window": None}},
    "clean+name+win3+latest": {**_D36, **_CLEAN, "faithfulness": {**_D36["faithfulness"], "attribution_name_match": True,
                                                                "attribution_window": 3,
                                                                "attribution_latest_speaker": True}},
    "name": {**_D36, "faithfulness": {**_D36["faithfulness"], "attribution_name_match": True}},
    "name+win3": {**_D36, "faithfulness": {**_D36["faithfulness"], "attribution_name_match": True,
                                           "attribution_window": 3}},
}


def _with_length_control(cfg: Mapping, lc: Mapping) -> dict:
    """Variant = length_control keys, plus an optional `faithfulness` dict merged into that section."""
    lc = dict(lc)
    p = dict(cfg["pipeline"])
    if "faithfulness" in lc:
        p["faithfulness"] = {**p.get("faithfulness", {}), **lc.pop("faithfulness")}
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
            row = score_row(run, vname, [r["sample_id"] for r in recs], [o.summary for o in outs],
                            [o.length_ok for o in outs], [o.filled_words / max(1, o.output_words) for o in outs],
                            texts, refs)
            row["rejected_per_text"] = round(st.mean(len(o.rejected) for o in outs), 3)
            rows.append(row)
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
# third round (PROGRESS 38): pronoun speakers, first-person / glued-heading fill sentences, more lead cleanup;
# on top of the PROGRESS 37 setting
_D37 = ROUND2_VARIANTS["clean+name+win3+latest"]
_EXT = {"fill_strip_leads_extended": True, "fill_skip_glued_headings": True}
_PRON = {"faithfulness": {**_D37["faithfulness"], "attribution_pronoun": True}}
ROUND3_VARIANTS = {
    "d37": _D37,
    "pronoun": {**_D37, **_PRON},
    "first_person": {**_D37, "fill_skip_first_person": "singular"},
    "first_person_all": {**_D37, "fill_skip_first_person": "all"},
    "leads": {**_D37, **_EXT},
    "all": {**_D37, **_EXT, **_PRON, "fill_skip_first_person": "singular"},
    "all+plural": {**_D37, **_EXT, **_PRON, "fill_skip_first_person": "all"},
    "fill_pronoun": {**_D37, "fill_skip_pronoun_speaker": True},
    "all+fill_pronoun": {**_D37, **_EXT, **_PRON, "fill_skip_first_person": "singular", "fill_skip_pronoun_speaker": True},
    "all+fill_pronoun+share": {**_D37, **_EXT, "fill_skip_first_person": "singular", "fill_skip_pronoun_speaker": True,
                               "faithfulness": {**_PRON["faithfulness"], "attribution_latest_share": True}},
    "all+plural+fill_pronoun": {**_D37, **_EXT, **_PRON, "fill_skip_first_person": "all",
                                "fill_skip_pronoun_speaker": True},
}
# fourth round (PROGRESS 40): sentence order; on top of the PROGRESS 39 setting
_D39 = ROUND3_VARIANTS["all+fill_pronoun"]
ROUND4_VARIANTS = {"d39": _D39, "source_order": {**_D39, "order": "source"}}
