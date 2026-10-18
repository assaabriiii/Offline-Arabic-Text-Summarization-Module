"""Human-readable strategy names for the CLI / UI."""

STRATEGY_LABELS = {
    "segmentwise": "segment-wise abstractive",
    "hybrid": "extract then rewrite",
    "direct": "chunked direct (900/80, default)",
    "extractive": "extractive only (no model)",
    "lead": "first sentences (no model)",
}
