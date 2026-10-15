# Progress

Source of truth for requirements: `docs/proposal.pdf`. It's kept out of git because it holds
personal contact details.

## Status
| # | Deliverable | Status |
|---|---|---|
| 0 | Package skeleton, config, env report, resumable steps, tests | done |
| 1 | `01_data.ipynb`: XL-Sum + dedup + benchmark scaffold + leakage check + manifest | **done**, run 2026-10-15 (see results below) |
| 2 | `02_preprocess.ipynb`: sentence split, chunking, length policy, pipeline (5 strategies), tokenizer calibration | **built, awaiting Colab run** |
| 3 | `03_finetune.ipynb`: faithfulness filter, fact checker, LR grid, final run, AraT5v2 run, decoding tuning, export + provenance, GPU ledger | **built, awaiting Colab T4 run** |
| 4 | `04_optimize.ipynb`: PyTorch/ONNX × FP32/INT8, isolated CPU measurement, proposal gates + composite score, export | **built, awaiting Colab CPU run** (dry run possible before 03 finishes) |
| 5 | `05_evaluate.ipynb`: all systems on 240 texts, ROUGE with CIs, length, fact check, latency, BERTScore, blind human evaluation + kappa, acceptance table, report | **built**; runs after 03/04 (lead/extractive work now) |
| 6 | MVP: CLI + Gradio, `bench/` offline CPU runner | **built** (`python -m arabic_summ summarize|serve`, `bench/run_bench.py`); tested on a tiny model in the pinned runtime env; needs the exported model from 04 |
| — | Benchmark items (240 texts) | automated: 120 news (XL-Sum test) + 120 org reports (WHO EMRO / UNICEF MENA / UNFPA / UN Women AS), seeded selection |
| — | Benchmark reference summaries | **240 model drafts done** (`llm_draft`, pending); **human review outstanding** (decision 16) |

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

## Decisions (2026-10-15, references)
16. **References are model-drafted, human-verified** (deviates from proposal §11.1 and decision 6;
    the user chose this via "AI drafts + human review"; **project lead must approve**). Claude drafted
    all 240 (`reference_source=llm_draft`, `writer_id=claude-opus-5.5`, guideline v1.0 §2–§5). An
    automatic check confirmed hard length limits, ≥ 3 key points, one paragraph, and that every number
    in a draft occurs in its source. A manual pass removed details the sources didn't state. 20 drafts
    sit slightly outside the 15–25 % guideline band (all inside the hard 10–30 % limits); reviewers may
    lengthen the shortest (NEWS-0102/0105/0122/0125, ORG-0177 at 13–14 %). Every row needs a
    different, fluent Arabic human reviewer before `--strict` passes. Reports must say
    "model-drafted, human-verified". The drafts JSON is never committed. See guideline §1a.

17. **03 fine-tuning design**: XL-Sum train pairs are filtered with `eval.facts` (every highlight number must
    occur in the article; novel-content rate ≤ 0.35, relaxed to keep ≥ 50 % if needed, reported). The LR grid
    {1e-5, 3e-5, 5e-5} runs on a seeded 25 % of filtered train to fit the T4 budget; the selected LR is then
    trained on 100 %. Models are selected by validation ROUGE-1 among candidates within 0.02 of the best number
    precision. ROUGE is our own Arabic implementation (matching-normalized tokens, no stemming), so it is not
    directly comparable to model-card numbers. AraT5v2: one fp32 run at 5e-5 (top of the proposal grid) on the
    same 25 % subset. Decoding (beams 1–4 × LP 1/1.5/2, no-repeat 3) is tuned on 300 validation articles
    disjoint from the 500 used for epoch selection and saved as `models/arabart_ft/decoding.json`.

18. **04 optimization**: four variants (PyTorch FP32 reference, PyTorch dynamic INT8, ONNX Runtime FP32,
    ONNX Runtime dynamic INT8 with AVX2 kernels). optimum-onnx 0.1.0 needs transformers 4.57.6 and fails on
    Python 3.14, so ONNX export/inference use a separate `--system-site-packages` venv
    (`requirements/colab-onnx.txt`) driven by subprocess; tokenizers saved by transformers 5 load there via the
    `PreTrainedTokenizerFast` fallback. Each variant is measured in a fresh process (batch 1, preloaded model,
    peak RSS, warm-up, per-sample input length, CPU model/threads/OS/runtime versions). Selection follows the
    proposal: gates (weights ≤ 1 GB, RSS ≤ 8 GB, median ≤ 15 s, ROUGE-1 drop ≤ 0.02 vs FP32), then 50/30/10/10.
    PyTorch INT8 quantizes at load time, so its on-disk size is the FP32 file. Colab CPU (2 vCPU) latencies are
    only comparative; acceptance timing is done by `bench/` on the 8-core machine. Verified locally on a tiny
    BART: all four variants produce identical outputs.

19. **05 evaluation**: the primary system's strategy (`segmentwise`) is fixed in config before any benchmark
    scoring; `direct` and `hybrid` are reported as ablations, so the strategy isn't tuned on the test set.
    The primary system is the optimized CPU variant from 04 (else the fine-tuned model). Interim numbers use all
    drafted references and are labelled INTERIM; the final report sets `approved_only: true`. Human evaluation:
    100 outputs (50 per domain) in blind sheets for two raters, 20 calibration items for a reference rater
    (weighted kappa ≥ 0.7), quadratic-weighted Cohen's kappa, a third rater for gaps > 1 point or disagreement on
    critical errors. "Mean ≥ 4" is read strictly as every criterion ≥ 4.
    **Fix found by the dry run**: `lead` and `extractive` stopped at the target length even when that left the
    summary below the minimum (lead met the bounds on only 84 % of benchmark texts). They now skip sentences that
    would exceed the maximum and fill to the minimum; `trim_to_words` can complete the minimum at a word boundary.
    **Interim reference points** (240 texts vs model-drafted references, pending review): lead ROUGE-1 0.370
    (95 % CI 0.359–0.380), extractive 0.352; both 100 % length-compliant. The 0.40 target is therefore above
    simple extraction on this benchmark.

20. **MVP + bench**: one `SummarizerService` loads a self-contained model folder (weights, tokenizer,
    `decoding.json`, `runtime.json` with variant / tokens-per-word / strategy) and sets the Hugging Face offline
    switches before loading, so no network call is attempted. CLI: `python -m arabic_summ summarize`; local web UI:
    `python -m arabic_summ serve` (Gradio bound to 127.0.0.1, analytics off). Every result carries the
    fact-check flag (numbers not found in the input). `bench/run_bench.py` is the acceptance runner for the 8-core
    machine: batch 1, preloaded model, warm-up, per-sample input length and latency, CPU model / threads / OS,
    peak RSS, weight size, outbound-network probe; exit code 0 only if every hard criterion passes.
    **One offline runtime env** (`requirements/cpu-runtime.txt`, Python 3.13) runs both PyTorch and ONNX
    variants: transformers 4.57.6 + optimum-onnx + onnxruntime + Gradio 5.50 (Gradio 6 needs huggingface-hub ≥ 1,
    which transformers 4.57 rejects). Fixed while testing it: tokenizers saved by transformers 5 and loaded under
    4.x emitted `token_type_ids`, which BART's `generate` rejects.

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

## 02_preprocess results (Colab T4, 2026-10-15, commit a5eb9c4)
- Tokens per word (XL-Sum train sample, n=2000): AraBART 1.276 article / 1.237 summary;
  AraT5v2 1.299 / 1.257.
- News benchmark candidates (150): median 748 AraBART tokens, max 1384. **33% exceed 900 tokens and
  23% exceed 1024**, so chunking is needed for about a third of inputs.
- Chunking on NEWS-0135 (1000 words): direct = 2 chunks [897, 456]; segmentwise = 5 chunks of 249–317 tokens.
- Smoke run (Omar10lfc/arabart-xlsum-arabic, fp16, 9 news texts, ratio 0.2): length compliance
  direct 0.83, all others 1.0. Median time on T4: direct 1.33 s, segmentwise 1.08 s, hybrid 1.08 s.
- **Faithfulness problem (critical for the ≥ 80% factual-accuracy target):** on NEWS-0029 every
  abstractive strategy invented facts. Examples: Houthis "attacking the intelligence HQ" (the source
  says resistance committees attacked Houthi police), plus "Sanaa", "Ibb" and a "رفض" movement, none
  of which appear in the article. Two likely causes: XL-Sum highlights often contain information
  that isn't in the article, so the model learned to add facts; and forcing a model trained on
  ~25-word outputs to write 80+ words (min_new_tokens) pushes it to pad.
  Implications for 03:
  (a) filter XL-Sum training pairs for faithfulness (entity/number precision of the summary against
      its article) before fine-tuning;
  (b) build the factual checker now (moved forward from 05) and use it in model and decoding selection;
  (c) prefer strategies whose per-call compression is mild (hybrid rewrite) and keep extractive as
      a guaranteed-faithful fallback.

## Benchmark items frozen (01_data cells 7–9, Colab, 2026-10-15, commit 093fb97)
- 180 org-report pages fetched, 0 failures, 0 content-hash changes since harvesting.
- Auto-selection (rules v1, seed 42): **240 selected**, 120 news + 120 org_report, each split
  40/40/40 across the 300–499 / 500–699 / 700–1000 bins. Org sources: WHO EMRO 34, UN Women AS 34,
  UNICEF 27, UNFPA 25. 0 screening failures; 90 reserves (30 news, 60 org).
- Validation PASS (non-strict), **0 leaks** against XL-Sum train+validation (240 checked).
  Remaining warnings: 240 reference summaries are missing (human work).
- Colab note: a reattached runtime kept old `arabic_summ` modules in memory, which caused
  AttributeErrors after code updates. The setup cell now drops cached `arabic_summ` modules after
  pulling (commit 093fb97).

## Next
- Upload `reference_drafts.json` to `DRIVE_ROOT/data/benchmark/`, run 01_data cells 8b + 9.
- Team (parallel): human review of all 240 drafts per guideline §1a/§6, then `benchmark-validate --strict`.
- Run `03_finetune.ipynb` on a T4 (resumable; paste back cell 13 output).
- Then `04_optimize.ipynb` on a CPU runtime (can be dry-run on the public baseline now).
- Then `05_evaluate.ipynb` (T4); human raters fill the sheets it writes.
- Next: install-and-run guide (English + Persian summary) and architecture notes.
