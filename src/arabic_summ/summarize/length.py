"""Output-length policy (proposal §7): 10-30 % of input words, clamped to 30-300 words."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class LengthPolicy:
    min_ratio: float = 0.10
    max_ratio: float = 0.30
    min_words: int = 30
    max_words: int = 300

    @classmethod
    def from_config(cls, pcfg: Mapping) -> "LengthPolicy":
        return cls(pcfg["min_ratio"], pcfg["max_ratio"], pcfg["min_words"], pcfg["max_words"])

    def clamp_ratio(self, ratio: float) -> float:
        return min(self.max_ratio, max(self.min_ratio, ratio))

    def target_words(self, input_words: int, ratio: float) -> int:
        """Target summary length. The absolute 30-300 clamp wins over the ratio for very short
        or very long inputs (e.g. a 200-word input still gets 30 words)."""
        t = round(self.clamp_ratio(ratio) * input_words)
        t = max(t, math.ceil(self.min_ratio * input_words))     # rounding must not leave the
        t = min(t, math.floor(self.max_ratio * input_words))    # 10-30 % band
        return int(min(self.max_words, max(self.min_words, t)))

    def bounds(self, input_words: int, ratio: float, tolerance: float = 0.2) -> tuple[int, int]:
        """Acceptable [lo, hi] word range around the target, never outside the hard limits."""
        t = self.target_words(input_words, ratio)
        hard_lo = max(self.min_words, math.ceil(self.min_ratio * input_words))
        hard_hi = min(self.max_words, math.floor(self.max_ratio * input_words))
        if hard_hi < hard_lo:  # inputs < 100 words: ratio cap below the 30-word floor
            hard_hi = hard_lo
        lo = max(hard_lo, math.floor(t * (1 - tolerance)))
        hi = min(hard_hi, math.ceil(t * (1 + tolerance)))
        return min(lo, hi), hi


def allocate(budget: int, weights: Sequence[float], min_each: int = 0) -> list[int]:
    """Split an integer `budget` proportionally to `weights` (largest-remainder method),
    giving each part at least `min_each` when the budget allows it."""
    n = len(weights)
    if n == 0:
        return []
    floor_each = min_each if budget >= min_each * n else budget // n
    rest = budget - floor_each * n
    total = float(sum(weights)) or 1.0
    raw = [rest * w / total for w in weights]
    parts = [floor_each + int(math.floor(r)) for r in raw]
    leftover = budget - sum(parts)
    order = sorted(range(n), key=lambda i: raw[i] - math.floor(raw[i]), reverse=True)
    for i in order[:leftover]:
        parts[i] += 1
    return parts


def token_bounds(target_words: int, tokens_per_word: float, min_factor: float, max_factor: float) -> tuple[int, int]:
    """min_new_tokens / max_new_tokens for a decoder given a word target."""
    t = target_words * tokens_per_word
    return max(1, int(math.floor(t * min_factor))), max(2, int(math.ceil(t * max_factor)))
