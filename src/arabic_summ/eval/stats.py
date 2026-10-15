"""Bootstrap confidence intervals (seeded) for per-sample metric means and paired differences."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_diff_ci(a: Sequence[float], b: Sequence[float], **kw) -> tuple[float, float, float]:
    """CI of mean(a - b) over the same samples (system comparison on one benchmark)."""
    return bootstrap_ci(np.asarray(a, float) - np.asarray(b, float), **kw)
