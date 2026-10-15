"""Model / decoding selection that uses the fact checker (02 finding: ROUGE alone rewards
fluent hallucination). Among candidates whose faithfulness is within `faith_tolerance` of the
best candidate, pick the highest primary metric (ROUGE-1 by default)."""

from __future__ import annotations

from typing import Mapping, Sequence


def select_best(rows: Sequence[Mapping], primary: str = "rouge1", faith_key: str = "number_precision",
                faith_tolerance: float = 0.02, lower_is_better_faith: bool = False) -> dict:
    if not rows:
        raise ValueError("no candidates")
    faith = [(-r[faith_key] if lower_is_better_faith else r[faith_key]) for r in rows]
    best_faith = max(faith)
    eligible = [r for r, f in zip(rows, faith) if f >= best_faith - faith_tolerance]
    return dict(max(eligible, key=lambda r: r[primary]))
