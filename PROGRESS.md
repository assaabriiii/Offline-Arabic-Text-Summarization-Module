# Progress

Source of truth for requirements: `docs/proposal.pdf`. It's kept out of git because it holds
personal contact details.

## Status
| # | Deliverable | Status |
|---|---|---|
| 0 | Package skeleton, config, env report, resumable steps, tests | done |
| 1 | `01_data.ipynb`: XL-Sum + dedup + benchmark scaffold + leakage check + manifest | **done**, run 2026-10-15 (see results below) |
| 2 | `02_preprocess.ipynb`: sentence split, chunking, length policy, pipeline (5 strategies), tokenizer calibration | **built, awaiting Colab run** |
| 3 | `03_finetune.ipynb` | — |
| 4 | `04_optimize.ipynb` | — |
| 5 | `05_evaluate.ipynb` | — |
| 6 | MVP: CLI + Gradio, `bench/` offline CPU runner | — |
| — | Benchmark items (240 texts) | automated: 120 news (XL-Sum test) + 120 org reports (WHO EMRO / UNICEF MENA / UNFPA / UN Women AS), seeded selection |
| — | Benchmark reference summaries | **human work outstanding** (writer + reviewer per guideline) |

## Decisions (2026-10-13)
1. **Acceptance ROUGE** is measured on the internal benchmark (≥ 240 texts with long human
   references). XL-Sum test ROUGE is reported for comparison only. Nothing is tuned on either.
2. **Length strategy**: (B) segment-wise abstractive summarization. Sentence-group segments
   are well under the 900-token cap with ≤ 80 tokens of overlap. Each segment gets a share of
   the length budget, segments are summarized with length-controlled decoding, duplicates are
   removed, the ratio is enforced, and a merge pass follows. This competes against (C), a
   hybrid where extractive sentence selection is followed by abstractive rewriting, as the
   proposal requires "abstractive vs hybrid". (D), longer-reference training data, is optional
   and only used if a properly licensed set exists (SumArabic license unverified). An
   extractive lead/TextRank baseline is also reported.
3. **AraT5v2** gets a single comparison run (no full LR grid) because T5 fp16 on a T4 is
   unstable. The full grid {1e-5, 3e-5, 5e-5} runs on AraBART.
4. **Model-size limit (≤ 1 GB)** applies to the optimized model plus tokenizer. AraT5v2 FP32
   (about 1.47 GB) can only pass after INT8.
5. **Benchmark news half** is drawn from XL-Sum *test* articles of 300–1000 words, stratified
   by length (300–499 / 500–699 / 700–1000), excluding any article in a duplicate pair, and
   oversampled to 150 candidates. Each gets a new human reference. The 1-sentence XL-Sum
   summary is kept only as `source_highlight`. The organizational-report half comes from the
   team via `incoming_org/`.
6. **References**: never model-generated. All references, including source-provided ones, need
   a second-reviewer approval (stricter than the proposal). Key points (≥ 3) are recorded for the
   proposal's 80% coverage check. Guideline: `docs/reference_guideline.md` v1.0.
7. **XL-Sum loading**: the raw archive is downloaded at pinned commit `30fece4…` and SHA256
   verified (`371a07df…`) instead of using `datasets.load_dataset`, because recent `datasets`
   releases don't run loading scripts.
8. **Dedup**: SHA256 of matching-normalized text plus MinHash/LSH (5-word shingles, 128 perms,
   Jaccard ≥ 0.8). Greedy by split priority test > validation > train. Only train rows are
   removed; eval splits are flagged, never modified.
9. **Normalization**: two policies. *model*: NFKC, strip tatweel and diacritics, fold Persian
   ی/ک, keep alef/ى/ة spelling and digits. *matching*: additionally folds alef, ى→ي, ة→ه,
   digits to ASCII, and removes punctuation. The matching policy is used only for hashing,
   dedup, ROUGE tokenization and fact matching.
10. **Colab versions**: we pin the packages we install. Colab's preinstalled stack (torch,
    pandas, …) isn't downgraded; its exact versions are saved per run in
    `logs/env_*.json` and `logs/pip_freeze_*.txt`. The final lock file is assembled from these.

## Decisions (2026-10-15)
11. **GPU**: the user has a Colab T4. Fine-tuning goes ahead on T4 (AraBART fp16; AraT5v2 single run).
12. **Benchmark items are automated** at the user's request: news items are picked from the 150
    XL-Sum test candidates. Organizational reports are harvested from public Arabic pages of four
    UN agencies (`tools/harvest_org_reports.py` → `benchmark_sources/org_reports.csv`, URLs plus
    content hashes only; texts are fetched in Colab and never committed). Selection rules v1:
    300–1000 words, ≥ 90% Arabic script, ≤ 15% numeric tokens, then 120 per domain spread
    evenly over length bins and round-robin over sources, seed 42. Unsampled items are kept as
    `reserve`.
    *Caveat*: these are public agency press releases, statements and reports, the closest
    openly available proxy for "organizational reports". If the client has real internal reports,
    they should be added via `incoming_org/` and replace some of these.
    **Reference summaries remain human-written** (proposal §11.1 and the original brief).
    Harvest 2026-10-15: 180 pages, 45 per organization (WHO EMRO, UNICEF regional and country
    offices, UNFPA, UN Women Arab States). By length bin: 52 / 70 / 58 (300–499 / 500–699 / 700–1000).
13. ReliefWeb was considered for org reports, but its API needs an approved appname and its RSS has no
    pagination or Arabic bodies. ESCWA is behind Cloudflare and the World Bank pages are rendered by JavaScript.
14. **Pipeline**: one `Summarizer` with strategies `direct` (900/80 hierarchical, proposal §11.3),
    `segmentwise` (320-token segments, strategy B), `hybrid` (TextRank extract 1.6× budget →
    abstractive rewrite, strategy C), plus `extractive` and `lead` baselines. Word targets become
    decoder token bounds via per-tokenizer tokens/word calibration (XL-Sum train sample).
    The "final pass" is sentence dedup plus trimming to the length bound. An abstractive merge pass is
    optional (`pipeline.merge`) and is evaluated in 05.
15. **Baselines** pinned in config: Omar10lfc/arabart-xlsum-arabic (Apache-2.0, card ROUGE-1
    34.99 on XL-Sum), kazed/AraBART-finetuned-xlsum-arabic, ahmeddbahaa/AraT5v2-base-1024-finetune-ar-xlsum,
    omarsabri8756/AraT5v2-XLSum-arabic-text-summarization. Three of these have no license on the Hub,
    so they are for comparison only.

## Benchmark report must also record (proposal §7)
CPU model, thread count, OS version, warm-up time, and per-sample input length. This goes into `bench/` later.

## 01_data results (Colab CPU runtime, 2026-10-15, commit 116aabd)
- Archive SHA256 verified (`371a07df…`). Raw split sizes match the proposal: 37519 / 4689 / 4689.
- Train: 28 near-empty rows dropped; 130 duplicates removed (95 exact, 35 near) → **37361**.
  Validation: 1 near-duplicate flagged and kept. Test: untouched.
- Lengths: median article is about 313 words (p90 882 train / 703 val+test). Median summary
  is about 23–24 words, a **median ratio of 0.072–0.074**, which is below our 10% floor.
  Only 18–26% of summaries have ≥ 30 words. About 50% of articles are 300–1000 words
  (2382 test articles in range). This confirms the length mismatch behind decision 2.
- Benchmark scaffold: 150 news candidates (50 per length bin), **0 leaks** vs train+validation.
- Colab image observed: Python 3.13, torch 2.11 (CPU build on a CPU runtime), transformers 5.18,
  datasets 4.8.5, huggingface_hub 1.33. Note that `03` must target transformers v5 APIs.

## Next
- Team (parallel, long lead time): write and review the 240 reference summaries per `docs/reference_guideline.md`.
- User re-runs `01_data.ipynb` (new cells 7–8: org reports + auto-selection), then runs `02_preprocess.ipynb` on T4.
- Then: `03_finetune.ipynb`.
