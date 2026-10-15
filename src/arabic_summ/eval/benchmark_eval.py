"""Score system outputs on the internal benchmark and check the proposal's acceptance criteria.

References: rows of references.csv with a summary. With `approved_only` only review_status=approved
rows count (final report); otherwise every drafted reference is used and the report says how many
were approved, so interim numbers are never presented as final.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from . import facts, rouge
from .stats import bootstrap_ci

LATENCY_BANDS = {"300-700": (300, 700), "300-1000": (300, 1000)}


def reference_table(refs: pd.DataFrame, approved_only: bool) -> pd.DataFrame:
    r = refs[refs.reference_summary.astype(str).str.strip() != ""]
    if approved_only:
        r = r[r.review_status == "approved"]
    return r.set_index("sample_id")


def score_outputs(outputs: pd.DataFrame, refs: pd.DataFrame, sources: Mapping[str, str]) -> pd.DataFrame:
    """Per (system, sample): ROUGE vs reference, fact check vs source, length compliance."""
    rows = []
    for _, o in outputs.iterrows():
        if o.sample_id not in refs.index:
            continue
        sc = rouge.score(refs.at[o.sample_id, "reference_summary"], o.summary)
        fr = facts.check(o.summary, sources[o.sample_id])
        rows.append({"system": o.system, "strategy": o.strategy, "sample_id": o.sample_id, "domain": o.domain,
                     "input_words": o.input_words, "output_words": o.output_words,
                     "ratio_out": o.output_words / max(1, o.input_words), "length_ok": bool(o.length_ok),
                     **sc, "numbers_ok": fr.numbers_ok, "novel_rate": fr.novel_rate,
                     "unsupported_numbers": " ".join(fr.unsupported_numbers),
                     "seconds": (o.seconds or {}).get("total") if isinstance(o.seconds, dict) else None,
                     "device": o.get("device", ""), "review_status": refs.at[o.sample_id, "review_status"],
                     "reference_source": refs.at[o.sample_id, "reference_source"]})
    return pd.DataFrame(rows)


def table_md(df: pd.DataFrame) -> str:
    """Markdown table (falls back to plain text when `tabulate` isn't installed)."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```\n" + df.to_string(index=False) + "\n```"


def length_band(n: int) -> str:
    return "300-499" if n < 500 else "500-699" if n < 700 else "700-1000"


def summary_table(scored: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rows = []
    for sys_, g in scored.groupby("system"):
        for name, h in [("all", g), *[(f"domain={d}", x) for d, x in g.groupby("domain")],
                        *[(f"len={b}", x) for b, x in g.groupby(g.input_words.map(length_band))]]:
            m, lo, hi = bootstrap_ci(h.rouge1, seed=seed)
            rows.append({"system": sys_, "subset": name, "n": len(h), "rouge1": round(m, 4),
                         "rouge1_ci95": f"{lo:.3f}-{hi:.3f}", "rouge2": round(h.rouge2.mean(), 4),
                         "rougeL": round(h.rougeL.mean(), 4), "length_ok": round(h.length_ok.mean(), 4),
                         "ratio_median": round(h.ratio_out.median(), 3), "numbers_ok": round(h.numbers_ok.mean(), 4),
                         "novel_rate": round(h.novel_rate.mean(), 4)})
    return pd.DataFrame(rows)


def latency_table(outputs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sys_, g in outputs.groupby("system"):
        secs = g.seconds.map(lambda s: s.get("total") if isinstance(s, dict) else None)
        for band, (lo, hi) in LATENCY_BANDS.items():
            m = g.input_words.between(lo, hi)
            if m.any():
                rows.append({"system": sys_, "band": band, "n": int(m.sum()), "device": ",".join(sorted(set(g.device))),
                             "median_s": round(float(secs[m].median()), 3), "p90_s": round(float(secs[m].quantile(0.9)), 3),
                             "max_s": round(float(secs[m].max()), 3)})
    return pd.DataFrame(rows)


def acceptance(system: str, summ: pd.DataFrame, lat: pd.DataFrame, crit: Mapping, human: pd.DataFrame | None = None,
               cpu: Mapping | None = None) -> list[dict]:
    """One row per proposal criterion: value, target, pass / pending."""
    a = summ[(summ.system == system) & (summ.subset == "all")].iloc[0]
    L = lat[lat.system == system].set_index("band") if len(lat) else pd.DataFrame()
    rows = [
        {"criterion": "ROUGE-1 F1", "value": a.rouge1, "target": f"> {crit['min_rouge1']}", "pass": a.rouge1 > crit["min_rouge1"]},
        {"criterion": "ROUGE-2 / ROUGE-L F1", "value": f"{a.rouge2} / {a.rougeL}", "target": "reported", "pass": True},
        {"criterion": "output length 10-30 % and 30-300 words", "value": a.length_ok, "target": "all texts",
         "pass": a.length_ok >= crit["min_length_ok"]},
    ]
    for band, key in (("300-700", "max_median_s_300_700"), ("300-1000", "max_median_s_all")):
        v = L.loc[band, "median_s"] if band in L.index else None
        dev = L.loc[band, "device"] if band in L.index else ""
        rows.append({"criterion": f"median latency {band} words ({dev})", "value": v, "target": f"<= {crit[key]} s",
                     "pass": None if v is None or "cuda" in dev else v <= crit[key]})
    if human is not None and len(human):
        h = human[(human.system == system) & (human.domain == "all")]
        if len(h):
            h = h.iloc[0]
            rows.append({"criterion": "no critical factual error (human)", "value": h.no_critical_error_share,
                         "target": ">= 0.8", "pass": bool(h.meets_factual_target)})
            rows.append({"criterion": "human scores (coverage/accuracy/fluency/coherence)",
                         "value": f"{h.coverage}/{h.accuracy}/{h.fluency}/{h.coherence}", "target": ">= 4 each",
                         "pass": bool(h.meets_human_target)})
    else:
        rows.append({"criterion": "no critical factual error (human)", "value": f"auto pre-screen numbers_ok={a.numbers_ok}",
                     "target": ">= 0.8", "pass": None})
        rows.append({"criterion": "human scores >= 4", "value": None, "target": ">= 4 each", "pass": None})
    if cpu:
        rows.append({"criterion": "process RAM", "value": cpu.get("peak_rss_mb"), "target": "<= 8192 MB",
                     "pass": cpu.get("peak_rss_mb") is not None and cpu["peak_rss_mb"] <= 8192})
        rows.append({"criterion": "model + tokenizer size", "value": cpu.get("weights_mb"), "target": "<= 1024 MB",
                     "pass": cpu.get("weights_mb") is not None and cpu["weights_mb"] <= 1024})
    return rows
