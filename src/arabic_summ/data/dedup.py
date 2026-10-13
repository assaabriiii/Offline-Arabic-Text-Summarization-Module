"""Exact + near-duplicate detection.

* Exact: SHA256 of the *matching*-normalized text (diacritics, alef/yeh/teh-marbuta,
  digits, punctuation and whitespace folded), so trivially re-encoded copies collide.
* Near: MinHash over word k-shingles of the same normalized text with LSH candidate
  search; a pair is a near-duplicate when the estimated Jaccard >= threshold.

Policy (greedy, deterministic): documents are visited in priority order (e.g. test,
validation, train; original order within a split). A document that duplicates an
already-kept document is *removed* if its split is removable, otherwise it is kept and
only *flagged*. Hence train copies of eval documents are removed, eval splits stay intact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from datasketch import LeanMinHash, MinHash, MinHashLSH

from ..io_utils import sha256_text

MINHASH_SEED = 1  # datasketch default; recorded in the manifest


def shingles(norm_text: str, k: int) -> set[str]:
    toks = norm_text.split()
    if len(toks) <= k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


def minhash(norm_text: str, k: int, num_perm: int) -> LeanMinHash:
    m = MinHash(num_perm=num_perm, seed=MINHASH_SEED)
    sh = shingles(norm_text, k)
    if sh:
        m.update_batch([s.encode("utf-8") for s in sh])
    return LeanMinHash(m)


def from_hashvalues(hv: np.ndarray) -> LeanMinHash:
    return LeanMinHash(seed=MINHASH_SEED, hashvalues=hv)


@dataclass(frozen=True)
class Doc:
    key: str          # globally unique, e.g. "train:<id>"
    split: str
    norm_text: str


@dataclass
class Match:
    key: str
    reason: str       # "exact" | "near"
    other: str        # key of the kept document it duplicates
    similarity: float


@dataclass
class DedupResult:
    removed: dict[str, Match] = field(default_factory=dict)
    flagged: dict[str, Match] = field(default_factory=dict)
    exact_hash: dict[str, str] = field(default_factory=dict)
    signatures: dict[str, np.ndarray] = field(default_factory=dict)

    def stats(self) -> dict:
        def by(d: dict[str, Match]) -> dict:
            out: dict = {}
            for m in d.values():
                split = m.key.split(":", 1)[0]
                out.setdefault(split, {"exact": 0, "near": 0})[m.reason] += 1
            return out

        return {"removed": by(self.removed), "flagged": by(self.flagged)}


def deduplicate(
    docs: Sequence[Doc],
    priority: Sequence[str],
    removable: Iterable[str],
    k: int,
    num_perm: int,
    threshold: float,
) -> DedupResult:
    removable = set(removable)
    rank = {s: i for i, s in enumerate(priority)}
    order = sorted(range(len(docs)), key=lambda i: (rank.get(docs[i].split, len(rank)), i))

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept_sigs: dict[str, LeanMinHash] = {}
    kept_by_hash: dict[str, str] = {}
    res = DedupResult()

    for i in order:
        d = docs[i]
        h = sha256_text(d.norm_text)
        sig = minhash(d.norm_text, k, num_perm)
        res.exact_hash[d.key] = h
        res.signatures[d.key] = sig.hashvalues

        match: Match | None = None
        if h in kept_by_hash:
            match = Match(d.key, "exact", kept_by_hash[h], 1.0)
        else:
            best = (None, 0.0)
            for cand in lsh.query(sig):
                sim = sig.jaccard(kept_sigs[cand])
                if sim > best[1]:
                    best = (cand, sim)
            if best[0] is not None and best[1] >= threshold:
                match = Match(d.key, "near", best[0], round(float(best[1]), 4))

        if match and d.split in removable:
            res.removed[d.key] = match
            continue
        if match:
            res.flagged[d.key] = match
        lsh.insert(d.key, sig)
        kept_sigs[d.key] = sig
        kept_by_hash.setdefault(h, d.key)
    return res


class NearDupIndex:
    """Query-only index over a fixed corpus (used for benchmark leakage checks)."""

    def __init__(self, keys: Sequence[str], hashvalues: np.ndarray, threshold: float, k: int):
        self.k = k
        self.num_perm = hashvalues.shape[1]
        self.threshold = threshold
        self.lsh = MinHashLSH(threshold=threshold, num_perm=self.num_perm)
        self.sigs: dict[str, LeanMinHash] = {}
        with self.lsh.insertion_session() as sess:
            for key, hv in zip(keys, hashvalues):
                sig = from_hashvalues(hv)
                self.sigs[key] = sig
                sess.insert(key, sig)

    def query(self, norm_text: str) -> list[tuple[str, float]]:
        sig = minhash(norm_text, self.k, self.num_perm)
        hits = [(c, float(sig.jaccard(self.sigs[c]))) for c in self.lsh.query(sig)]
        return sorted([h for h in hits if h[1] >= self.threshold], key=lambda x: -x[1])
