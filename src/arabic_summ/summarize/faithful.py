"""Sentence-level faithfulness filter for model-written summary sentences (PROGRESS 35-36).

The 05 review found the model's copied and shortened sentences almost always faithful, but about half
of its rewritten/fused ones not: a number moved to another fact, a negation lost, a quote or speaker
taken from elsewhere, words that are not in the article. Each model sentence is checked against the
source sentences; a rejected sentence is dropped and the length fill replaces it with source
sentences. Lexical and conservative like `eval.facts`: it cannot see a changed quantifier ("less than
a third of" -> "about") or a wrong job title; those stay with the human evaluation.

Checks (`pipeline.faithfulness`; a threshold of null switches a check off):
* support      - share of the claim's content stems found in its best `max_support_sentences` source
                 sentences (greedy cover) >= `min_support`; the claim is the sentence without its
                 speaker ("يقول X إن ...")
* numbers      - every number occurs in a source sentence that shares >= `number_context` of the
                 claim's content stems
* negation     - a negation (ليس/لم/لن/لا) in the support sentences whose following words the sentence
                 uses must be kept, and the sentence may not negate what its support sentences don't
* quotes       - >= `quotes` of each quoted span's word bigrams (light stems) occur in the source
* attribution  - the speaker's words occur in the claim's support sentences or the
                 `attribution_window` sentences before them (null: any earlier sentence; share >=
                 `attribution`); with
                 `attribution_single_source` the claim of an attributed sentence must come from one source
                 sentence (dev review: two facts fused under one speaker); a quote whose source sentence
                 names another speaker ("تقول هيفاء: ...") is rejected; with `attribution_name_match` a
                 speaker named in the speaker phrase of a nearby support sentence and described earlier
                 passes; with `attribution_latest_speaker` the latest speaker phrase at or before the
                 support sentence (within the window) must share a word with the speaker
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

from ..eval.facts import STOPWORDS, stem
from ..text.normalize import NormalizationPolicy, normalize, normalize_for_matching

# matching folds without punctuation removal, so quotes / colons stay visible as tokens
_PRE = NormalizationPolicy(unify_alef=True, alef_maqsura_to_yeh=True, teh_marbuta_to_heh=True, digits="ascii",
                           lowercase_latin=True)
_TOKEN = re.compile(r"\d+(?:[.,،٫٬]\d+)*|[^\W\d]+|[^\w\s]")   # numbers, words (no digits), punctuation
_DIGITS = re.compile(r"^\d")
_QUOTED = re.compile(r"«([^»]+)»|\"([^\"]+)\"|“([^”]+)”")
_QUOTE_CHARS = frozenset("«»\"“”:")


def _norm(words: str) -> frozenset[str]:
    return frozenset(normalize_for_matching(w) for w in words.split())


_NEG = _norm("ليس ليست ليسوا ليسا لسن لست لسنا لم لن لا")
# "لا يزال" (still), "لا سيما" (especially), "لا بد" (must) are not negations of the following content
_NOT_NEG_NEXT = _norm("يزال تزال يزالون تزالان يزلن زال زالت سيما بد")
_SPEECH = _norm("""يقول تقول قال قالت أضاف أضافت يضيف تضيف أكد أكدت يؤكد تؤكد ذكر ذكرت أوضح أوضحت
                   أشار أشارت صرح صرحت يرى ترى حذر حذرت يحذر تحذر""")
_THAT = _norm("إن أن")
_SPEAKER_MAX = 8


_ARTICLE = tuple(normalize_for_matching(w) for w in "وال فال بال كال لل ال و ف".split())


def _key(t: str) -> str:
    """Light stem after removing a conjunction/article prefix first, so "الكراهية" and "كراهية" get the
    same key (`facts.stem` alone strips at most one prefix and may read the first letter as one)."""
    for p in _ARTICLE:
        if t.startswith(p) and len(t) - len(p) >= 3:
            t = t[len(p):]
            break
    return stem(t)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(normalize(text, _PRE))


def _is_word(t: str) -> bool:
    return t[:1].isalnum()


def _strip_conj(t: str, words: frozenset[str]) -> str | None:
    if t in words:
        return t
    if len(t) > 2 and t[0] in "وف" and t[1:] in words:
        return t[1:]
    return None


def _content(tokens: Sequence[str]) -> set[str]:
    return {_key(t) for t in tokens if _is_word(t) and not _DIGITS.match(t) and len(t) >= 3
            and t not in STOPWORDS and _key(t) not in STOPWORDS}


def _number(t: str) -> str:
    return re.sub(r"[.,،٫٬]", "", t).lstrip("0") or "0"


def _negations(tokens: Sequence[str]) -> list[set[str]]:
    """Content stems governed by each negation (its next three word tokens, up to the next punctuation:
    05 review, "دون تأخير لا داعي له، تدعم ..." does not negate "تدعم")."""
    out = []
    for i, t in enumerate(tokens):
        if _strip_conj(t, _NEG) is None:
            continue
        nxt = []
        for x in tokens[i + 1:i + 4]:
            if not _is_word(x):
                break
            nxt.append(x)
        if nxt and nxt[0] in _NOT_NEG_NEXT:
            continue
        out.append(_content(nxt))
    return out


def _speaker(tokens: Sequence[str]) -> tuple[list[str], list[str]]:
    """(speaker tokens, remaining tokens) for "يقول/قالت/أضاف X إن ..." sentences."""
    for i, t in enumerate(tokens):
        if _strip_conj(t, _SPEECH) is None:
            continue
        j = i + 1
        while j < len(tokens) and j - i <= _SPEAKER_MAX and tokens[j] not in _QUOTE_CHARS \
                and tokens[j] not in _THAT and tokens[j] != ".":
            j += 1
        spk = [x for x in tokens[i + 1:j] if _is_word(x)]
        if spk:
            return spk, list(tokens[:i]) + list(tokens[j:])
    return [], list(tokens)


class FaithfulnessChecker:
    def __init__(self, source_sentences: Sequence[str], cfg: Mapping):
        self.cfg = cfg
        toks = [_tokens(s) for s in source_sentences]
        self.sets = [{_key(t) for t in ts if _is_word(t)} for ts in toks]
        self.nums = [{_number(t) for t in ts if _DIGITS.match(t)} for ts in toks]
        self.negs = [_negations(ts) for ts in toks]
        self.speakers = [_content(_speaker(ts)[0]) for ts in toks]
        self.df: dict[str, int] = {}
        for ss in self.sets:
            for t in ss:
                self.df[t] = self.df.get(t, 0) + 1
        words = [[_key(t) for t in ts if _is_word(t)] for ts in toks]
        self.words = {t for ws in words for t in ws}
        self.bigrams = {(a, b) for ws in words for a, b in zip(ws, ws[1:])}

    def _cover(self, stems: set[str]) -> tuple[list[int], set[str]]:
        need, chosen = set(stems), []
        for _ in range(int(self.cfg.get("max_support_sentences", 2))):
            if not need:
                break
            best = max((i for i in range(len(self.sets)) if i not in chosen),
                       key=lambda i: len(need & self.sets[i]), default=None)
            if best is None or not need & self.sets[best]:
                break
            chosen.append(best)
            need -= self.sets[best]
        return chosen, need

    def _latest_speaker(self, i: int, w) -> set[str]:
        """Speaker phrase of the latest source sentence at or before `i` (within `w` before it) that has one."""
        lo = 0 if w is None else max(0, i - int(w))
        return next((self.speakers[j] for j in range(i, lo - 1, -1) if self.speakers[j]), set())

    def _named_nearby(self, ss: set[str], chosen: list[int], w) -> bool:
        """`attribution_name_match`: the speaker is named by a rare word (a name, in <= 20% of the source
        sentences) in the latest speaker phrase at or before a support sentence, within the window
        ("قال ليونز إن ..."), and the rest of the description occurs earlier ("الخبير الاقتصادي البريطاني
        جيرارد ليونز", first mention)."""
        if not self.cfg.get("attribution_name_match"):
            return False
        rare = max(2, 0.2 * len(self.sets))
        if not any(self.df.get(t, 0) <= rare for i in chosen for t in ss & self._latest_speaker(i, w)):
            return False
        before = set().union(*self.sets[:max(chosen) + 1])
        return len(ss & before) / len(ss) >= self.cfg["attribution"]

    def reasons(self, sentence: str) -> list[str]:
        c = self.cfg
        toks = _tokens(sentence)
        spk, rest = _speaker(toks)
        claim = _content(rest)
        chosen, missing = self._cover(claim)
        out = []
        if c.get("min_support") is not None and len(claim) >= 2 and 1 - len(missing) / len(claim) < c["min_support"]:
            out.append("support")
        if c.get("number_context") is not None:
            for n in {_number(t) for t in toks if _DIGITS.match(t)}:
                cand = [i for i, ns in enumerate(self.nums) if n in ns]
                if not cand:
                    out.append("number_unsupported")
                    break
                if claim and max(len(claim & self.sets[i]) for i in cand) / len(claim) < c["number_context"]:
                    out.append("number_context")
                    break
        if c.get("negation", True) and chosen:
            mine = _negations(toks)
            src = [g for i in chosen for g in self.negs[i]]
            used = _content(toks)
            if (any(g & used for g in src) and not mine) or (mine and not src):
                out.append("negation")
        if c.get("quotes") is not None:
            for m in _QUOTED.finditer(normalize(sentence, _PRE)):
                ws = [_key(t) for t in _TOKEN.findall(next(g for g in m.groups() if g)) if _is_word(t)]
                if spk and c.get("attribution") is not None and ws:
                    q = max(range(len(self.sets)), key=lambda i: len(set(ws) & self.sets[i]))
                    if self.speakers[q] and not self.speakers[q] & _content(spk):
                        out.append("attribution")
                if len(ws) == 1 and ws[0] not in self.words:
                    out.append("quote")
                    break
                bg = list(zip(ws, ws[1:]))
                if bg and sum(b in self.bigrams for b in bg) / len(bg) < c["quotes"]:
                    out.append("quote")
                    break
        if c.get("attribution") is not None and spk and chosen:
            ss = _content(spk)
            w = c.get("attribution_window", 2)
            near = set().union(*(self.sets[j] for i in chosen
                                 for j in range(0 if w is None else max(0, i - int(w)), i + 1)))
            latest = self._latest_speaker(chosen[0], w) if c.get("attribution_latest_speaker") else set()
            if ss and len(ss & near) / len(ss) < c["attribution"] and not self._named_nearby(ss, chosen, w):
                out.append("attribution")
            elif ss and latest and not ss & latest:      # the source gives the claim to someone else
                out.append("attribution")
            elif ss and c.get("attribution_single_source") and len(claim) >= 2 \
                    and len(claim & self.sets[chosen[0]]) / len(claim) < c["min_support"]:
                out.append("attribution")
        return sorted(set(out), key=out.index)


def filter_sentences(sentences: Sequence[str], source_sentences: Sequence[str],
                     cfg: Mapping) -> tuple[list[str], list[dict]]:
    """(kept sentences, rejected [{sentence, reasons}])."""
    chk = FaithfulnessChecker(source_sentences, cfg)
    kept, rejected = [], []
    for s in sentences:
        r = chk.reasons(s)
        if r:
            rejected.append({"sentence": s, "reasons": r})
        else:
            kept.append(s)
    return kept, rejected
