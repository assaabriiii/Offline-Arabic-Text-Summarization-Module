"""Blind human evaluation (proposal §7).

* >= 100 outputs sampled at random, balanced across the two domains; two fluent Arabic raters score
  each summary 1-5 for coverage, accuracy, fluency and coherence, and flag critical errors in names,
  numbers, dates or event attribution. Raters never see the system name (codes only; the key file is
  kept separately).
* Agreement: weighted Cohen's kappa per criterion. Items where the raters differ by more than one
  point go to an independent third rater, whose score replaces the pair's mean for that criterion.
* Rater qualification: weighted kappa >= 0.7 against a reference rater on >= 20 calibration items.
"""

from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

CRITERIA = ("coverage", "accuracy", "fluency", "coherence")
SHEET_COLUMNS = ["item_id", "domain", "source_text", "summary", "auto_flags", *CRITERIA,
                 "critical_error", "critical_error_type", "notes"]
ERROR_TYPES = ("name", "number", "date", "attribution", "other")


def weighted_kappa(a: Sequence[int], b: Sequence[int], labels: Sequence[int] = (1, 2, 3, 4, 5),
                   weights: str = "quadratic") -> float:
    """Cohen's kappa with linear or quadratic disagreement weights for ordinal ratings."""
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[idx[int(x)], idx[int(y)]] += 1
    n = obs.sum()
    if n == 0:
        return float("nan")
    i, j = np.meshgrid(range(k), range(k), indexing="ij")
    w = np.abs(i - j) / (k - 1) if weights == "linear" else ((i - j) / (k - 1)) ** 2
    exp = np.outer(obs.sum(1), obs.sum(0)) / n
    denom = (w * exp).sum()
    return 1.0 if denom == 0 else float(1 - (w * obs).sum() / denom)


def item_id(system: str, sample_id: str, salt: str) -> str:
    return "H" + hashlib.sha256(f"{salt}|{system}|{sample_id}".encode()).hexdigest()[:8].upper()


def build_sheets(outputs: pd.DataFrame, sources: Mapping[str, str], n_per_domain: int, n_calibration: int,
                 seed: int, salt: str, auto_flags: Mapping[tuple, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """outputs: rows with system, sample_id, domain, summary. Returns (sheet, key).

    The sample is drawn per system and domain (balanced). Calibration items are the first
    `n_calibration` rows of a separate seeded draw and are marked in the key only."""
    rng = np.random.default_rng(seed)
    picked = []
    for (_, dom), g in outputs.groupby(["system", "domain"]):
        take = g.sample(min(n_per_domain, len(g)), random_state=int(rng.integers(1 << 31)))
        picked.append(take)
    sel = pd.concat(picked).sample(frac=1, random_state=seed).reset_index(drop=True)
    sel["item_id"] = [item_id(s, i, salt) for s, i in zip(sel.system, sel.sample_id)]
    cal_ids = set(sel.sample(min(n_calibration, len(sel)), random_state=seed + 1).item_id)
    flags = auto_flags or {}
    sheet = pd.DataFrame({
        "item_id": sel.item_id, "domain": sel.domain, "source_text": [sources[i] for i in sel.sample_id],
        "summary": sel.summary, "auto_flags": [flags.get((s, i), "") for s, i in zip(sel.system, sel.sample_id)],
        **{c: "" for c in CRITERIA}, "critical_error": "", "critical_error_type": "", "notes": "",
    })[SHEET_COLUMNS]
    key = pd.DataFrame({"item_id": sel.item_id, "system": sel.system, "sample_id": sel.sample_id,
                        "domain": sel.domain, "calibration": sel.item_id.isin(cal_ids)})
    return sheet, key


def _scores(sheet: pd.DataFrame) -> pd.DataFrame:
    s = sheet.set_index("item_id")
    out = s[list(CRITERIA)].apply(pd.to_numeric, errors="coerce")
    out["critical"] = s["critical_error"].astype(str).str.strip().str.lower().isin({"yes", "y", "1", "true", "نعم"})
    return out


def validate_sheet(sheet: pd.DataFrame) -> list[str]:
    errs = []
    sc = _scores(sheet)
    for c in CRITERIA:
        bad = sc.index[~sc[c].isin([1, 2, 3, 4, 5])].tolist()
        if bad:
            errs.append(f"{c}: {len(bad)} items not scored 1-5 (e.g. {bad[:3]})")
    ce = sheet.critical_error.astype(str).str.strip().str.lower()
    bad = sheet.item_id[~ce.isin({"yes", "no", "y", "n", "1", "0", "true", "false", "نعم", "لا"})].tolist()
    if bad:
        errs.append(f"critical_error: {len(bad)} items not yes/no (e.g. {bad[:3]})")
    return errs


def agreement(a: pd.DataFrame, b: pd.DataFrame, weights: str = "quadratic", ids: Sequence[str] | None = None) -> dict:
    sa, sb = _scores(a), _scores(b)
    common = sa.index.intersection(sb.index) if ids is None else pd.Index(ids)
    out = {c: round(weighted_kappa(sa.loc[common, c], sb.loc[common, c], weights=weights), 4) for c in CRITERIA}
    out["critical_error_agreement"] = round(float((sa.loc[common, "critical"] == sb.loc[common, "critical"]).mean()), 4)
    out["n"] = int(len(common))
    return out


def adjudication_items(a: pd.DataFrame, b: pd.DataFrame, max_gap: int = 1) -> pd.DataFrame:
    """Rows (item_id, criterion) where the two raters differ by more than `max_gap` points,
    plus items where they disagree on a critical error."""
    sa, sb = _scores(a), _scores(b)
    rows = []
    for c in CRITERIA:
        gap = (sa[c] - sb[c]).abs()
        rows += [{"item_id": i, "criterion": c, "rater_a": sa.at[i, c], "rater_b": sb.at[i, c]} for i in gap.index[gap > max_gap]]
    dis = sa.index[sa.critical != sb.critical]
    rows += [{"item_id": i, "criterion": "critical_error", "rater_a": sa.at[i, "critical"], "rater_b": sb.at[i, "critical"]} for i in dis]
    return pd.DataFrame(rows, columns=["item_id", "criterion", "rater_a", "rater_b"])


def final_scores(a: pd.DataFrame, b: pd.DataFrame, key: pd.DataFrame, third: pd.DataFrame | None = None,
                 max_gap: int = 1) -> pd.DataFrame:
    """Per item: mean of the two raters, replaced by the third rater where the gap > max_gap.
    Critical error: OR of the two raters unless a third rater adjudicated it."""
    sa, sb = _scores(a), _scores(b)
    fin = (sa[list(CRITERIA)] + sb[list(CRITERIA)]) / 2
    fin["critical"] = sa.critical | sb.critical
    if third is not None and len(third):
        t = third.copy()
        for _, r in t.iterrows():
            v = str(r["rater_c"]).strip().lower()
            if not v:
                continue
            if r["criterion"] == "critical_error":
                fin.at[r["item_id"], "critical"] = v in {"yes", "y", "1", "true", "نعم"}
            elif (sa.at[r["item_id"], r["criterion"]] - sb.at[r["item_id"], r["criterion"]]).__abs__() > max_gap:
                fin.at[r["item_id"], r["criterion"]] = float(v)
    fin["mean_score"] = fin[list(CRITERIA)].mean(axis=1)
    return fin.join(key.set_index("item_id"))


def summarize(fin: pd.DataFrame, min_mean: float = 4.0, max_critical_share: float = 0.2) -> pd.DataFrame:
    rows = []
    for sys_, g in fin.groupby("system"):
        for dom, h in [("all", g), *g.groupby("domain")]:
            rows.append({"system": sys_, "domain": dom, "n": len(h), **{c: round(h[c].mean(), 3) for c in CRITERIA},
                         "mean_score": round(h.mean_score.mean(), 3),
                         "no_critical_error_share": round(1 - h.critical.mean(), 3)})
    df = pd.DataFrame(rows)
    df["meets_human_target"] = df[list(CRITERIA)].min(axis=1) >= min_mean
    df["meets_factual_target"] = df.no_critical_error_share >= 1 - max_critical_share
    return df
