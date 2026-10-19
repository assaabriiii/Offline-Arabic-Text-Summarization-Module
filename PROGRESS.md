# Progress

Source of truth for requirements: `docs/proposal.pdf`. It's kept out of git because it holds
personal contact details.

## Status
| # | Deliverable | Status |
|---|---|---|
| 0 | Package skeleton, config, env report, resumable steps, tests | done |
| 1 | `01_data.ipynb`: XL-Sum + dedup + benchmark scaffold + leakage check + manifest | **done**, run 2026-10-15 (see results below) |
| 2 | `02_preprocess.ipynb`: sentence split, chunking, length policy, pipeline (5 strategies), tokenizer calibration | **done**, run 2026-10-15 on T4 |
| 3 | `03_finetune.ipynb`: faithfulness filter, fact checker, LR grid, final run, AraT5v2 run, decoding tuning, export + provenance, GPU ledger | **done (2026-10-18)**: `models/arabart_ft` exported (534.6 MB), validation ROUGE-1 0.284 (greedy, 500 articles), 0.303 with the selected 4-beam decoding (300 other articles), number precision 0.96–1.00 (decision 24). AraT5v2 comparison not run yet (optional, off by default) |
| 4 | `04_optimize.ipynb`: PyTorch/ONNX × FP32/INT8, isolated CPU measurement, proposal gates + composite score, export | **done on Colab CPU (2026-10-18)**: selected **PyTorch dynamic INT8** → `models/arabart_ft_cpu` (R1 0.305, median 11.1 s on 2 vCPUs, 1.9 GB RSS). ONNX INT8 missed the latency gate by 0.11 s on 2 vCPUs: re-check both on the 8-core reference machine (decision 25) |
| 3b | `03b_style_finetune.ipynb`: continue `arabart_ft` on filtered oddadmix pairs | **done (2026-10-18)**: `models/arabart_ft_style`, dev R1 0.360 (decision 32) |
| 4b | 04 for the style model | **done (2026-10-19)**: torch INT8 → `models/arabart_ft_style_cpu`, 531 MB, median 12.8 s on 2 vCPUs (decision 34) |
| 5 | `05_evaluate.ipynb`: all systems on 240 texts, ROUGE with CIs, length, fact check, latency, BERTScore, blind human evaluation + kappa, acceptance table, report | **automatic part done (2026-10-19, interim references)**: primary `cpu_torch_int8_direct` with the faithfulness filter (decisions 36-37): R1 **0.439** XL-Sum-style (acceptance metric, decision 38; CI 0.430–0.448, passes) / 0.376 strict, BERTScore 0.751, all automatic criteria pass (decisions 36-39). INT8 output varies with the CPU type (decision 37). Human-evaluation sheets re-written 2026-10-20 (100 items, decision 39), waiting for raters |
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
    "model-drafted, human-verified". See guideline §1a. *Update 2026-10-16*: at the owner's request the
    drafts are committed (`benchmark_sources/reference_drafts.json`, public repo), so the benchmark
    answers are public; reports must say so.

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

21. **First 03 run (r1, Colab T4, 2026-10-16) collapsed and was stopped**: on the grid run lr 1e-5, ROUGE-1
    fell 0.044 → 0.004 → 0.0003 and generated length 102 → 9 → 0.8 words over 3 epochs. Cause: transformers 5
    builds AraBART's sentencepiece `BarthezTokenizer` without the `<s> … </s>` post-processor, so sources and
    labels had no special tokens and the model never learned where a summary ends. Fix: `models.ensure_special_tokens`
    installs the BART template when EOS is missing (regression test added); 03 cell 6 now asserts sources and
    labels end with EOS. Runs are versioned (`finetune.train.run_tag: r2`) so the broken r1 checkpoints on
    Drive are never resumed. The r1 GPU time stays in the ledger.
    Filter result from that run (valid, unaffected by the bug): 37,361 → 18,914 train pairs kept (50.6 %);
    17.3 % of XL-Sum highlights contain a number absent from their article; median novel-content rate 0.43,
    so the threshold was relaxed from 0.35 to 0.467 to keep half.

22. **r2 grid (Colab T4, 2026-10-16): reported ROUGE-1 ~0.055 for all three LRs, but the models were fine.**
    transformers 5 applies `group_by_length` (`train_sampling_strategy`) to the *evaluation* sampler as well,
    so generated summaries came back length-sorted while `compute_metrics` paired them with references and
    sources in dataset order. Every per-epoch score compared a summary with another article (hence novel
    rate ~0.90, number precision ~0.08), so early stopping, best-epoch choice and the LR selection were noise.
    Re-scored in order on the first 64 of the 500 eval articles (greedy, max 128 tokens):
    | model | ROUGE-1 fp32 | ROUGE-1 fp16 | novel rate | words |
    |---|---|---|---|---|
    | AraBART, no fine-tuning | 0.124 | 0.123 | 0.06 | 100 |
    | r2 lr 1e-5 `best/` (noise-selected epoch) | 0.245 | 0.244 | 0.23 | 21 |
    | r2 lr 3e-5 `best/` | 0.262 | 0.258 | 0.23 | 19 |
    | r2 lr 5e-5 epoch 1 checkpoint | 0.270 | 0.272 | 0.22 | 19 |
    | public baseline Omar10lfc (24 other val articles, local CPU) | 0.294 | – | 0.25 | 20 |
    fp16 does not hurt generation; the training-batch format is correct (`labels` `<s> … </s>`,
    `decoder_input_ids` `[</s>, <s>, …]`). Fix: `train.seq2seq.OrderedEvalTrainer` evaluates sequentially
    (length grouping stays on for training) and `make_compute_metrics` now checks every call that the
    returned label ids are in eval-set order, raising otherwise (regression tests added). `run_tag: r3`.
    The "training loss" logged by transformers 5 is ~4× the validation loss, i.e. apparently summed over the
    4 gradient-accumulation micro-batches; the validation loss and the in-order ROUGE above show training works.

23. **Epoch checkpoints moved off Drive (2026-10-17).** The r3 grid and final run trained fine on a
    contributor's Colab, but almost nothing reached the shared Drive: the upload stopped mid-checkpoint
    (`checkpoint-888` without `optimizer.pt`). Files a collaborator writes into a shared folder count against
    *their own* Drive quota, and every epoch checkpoint is ~1.5 GB (deleted ones sit in Trash), so their
    free 15 GB filled within about an hour and uploads failed silently. Now `finetune.train.checkpoint_dir`
    (`/content/arabic_summ_checkpoints`) keeps checkpoints on the runtime's local disk; Drive only gets
    results, logs and each run's `best/` model (~0.6 GB per run, ~3 GB for all of 03). Trade-off: an
    interrupted run resumes after *Restart session*, but on a fresh runtime it restarts from scratch (grid run
    ~15–25 min, final ~1–1.5 h). 03 now ends with `drive.flush_and_unmount()` (waits for pending uploads), and
    the AraT5v2 comparison is off by default (`RUN_ARAT5`), since fp32 batch 4 ran out of T4 memory and the
    reduced setting (batch 2 × 8, gradient checkpointing) takes hours.

24. **03 finished (r3, Colab T4, 2026-10-16 → 2026-10-18; ledger total 10.5 T4-hours incl. all failed runs).**
    - LR grid (25 % of filtered train, best epoch per run, 500 validation articles, greedy):
      1e-5 → R1 0.2821 / number precision 0.980; 3e-5 → 0.2825 / 0.948; 5e-5 → 0.2783 / 0.951 (stopped at 3).
      Selected **1e-5**: 3e-5 has the top R1 but its number precision is 0.032 below 1e-5's (tolerance 0.02).
    - Final run, 100 % of filtered train (18,914 pairs), lr 1e-5: R1 by epoch 0.2799 / 0.2821 / 0.2830 /
      **0.2838** / 0.2834 → best epoch 4 (R2 0.123, RL 0.242, number precision 0.963, novel rate 0.22,
      19 words). The run was split over two Colab sessions (free-tier quota): epochs 1–2, then the epoch-2
      checkpoint was copied to Drive and the next session resumed from it (`latest_checkpoint`, commit 2976dfd).
      The resumed trainer could not reload its pre-resume best checkpoint (old local path); harmless here because
      epochs 3–4 improved on it. Fixed afterwards: a Drive checkpoint is copied back to the local checkpoint
      folder before resuming.
    - Decoding grid (300 validation articles disjoint from the 500): greedy R1 0.310 but number precision 0.933;
      selected **4 beams, length penalty 1.5, no-repeat 3**: R1 0.303, R2 0.144, RL 0.259, number precision 1.000,
      novel rate 0.19, 0.08 s/text on the T4. Beams 2–4 all lie within R1 0.301–0.303.
    - Sanity check, 6 benchmark texts × 3 strategies vs the public baseline (fp16, T4): numbers supported in all
      outputs of both models; novel rate ours 0.16–0.19 vs baseline 0.20–0.21; length compliance ours 0.67–0.83,
      baseline 0.67–1.00; ~1.2 s/text. Example `NEWS-0055` (hybrid) reads fluently but merges the positions of
      two speakers (the mufti and his critics) into one quote: a coherence/attribution error that numbers and
      novel-word checks cannot detect. The human evaluation (05) must judge these.

25. **04 results (Colab CPU, 2 vCPU Xeon 2.20 GHz, 2 threads, batch 1, 4-beam decoding; 200 XL-Sum validation
    articles for ROUGE, 24 stratified benchmark texts through the segmentwise pipeline for latency):**
    | variant | weights | peak RSS | median / p90 latency | R1 | number precision | bench length ok / numbers ok | gates |
    |---|---|---|---|---|---|---|---|
    | torch FP32 | 531 MB | 1.45 GB | 19.7 / 31.2 s | 0.303 | 1.000 | 0.88 / 1.00 | latency |
    | **torch INT8 (dynamic)** | 531 MB on disk | 1.88 GB | **11.1 / 15.4 s** | **0.305** | 0.964 | 0.79 / 0.88 | pass |
    | ONNX FP32 | 1,310 MB | 2.32 GB | 14.7 / 24.8 s | 0.303 | 1.000 | 0.88 / 1.00 | size |
    | ONNX INT8 (AVX2) | 329 MB | 1.26 GB | 15.1 / 24.0 s | 0.296 | 1.000 | 0.83 / 0.96 | latency (by 0.11 s) |
    PyTorch INT8 was the only variant to pass all gates, so it was exported (its composite score of 1.0 is
    trivial with one candidate). **Caveat:** the latency gate is a reference-machine criterion; on 2 vCPUs it
    decided between torch INT8 and ONNX INT8 by 0.11 s, and ONNX INT8 is smaller (329 MB), lighter (1.26 GB)
    and had more benchmark outputs with all numbers supported (0.96 vs 0.88, 24 texts). Re-run both on the 8-core
    reference machine (`bench/`) before freezing the choice. ONNX FP32 duplicates decoder weights
    (decoder + decoder-with-past), hence 1.3 GB.
    Fixes found while running 04: the ONNX venv is now isolated and built with uv (Colab's Python has no
    ensurepip; the system diffusers broke optimum), and tokenizer files written by transformers 5
    (`extra_special_tokens` as a list) now load under transformers 4.57.

26. **05, CPU half (Colab CPU, 2 vCPU, 2026-10-18; INTERIM: 240 model-drafted references, none human-approved yet).**
    | system | ROUGE-1 (95 % CI) | R2 | RL | length ok | median length ratio | numbers ok | novel rate | median s, 300–700 / all |
    |---|---|---|---|---|---|---|---|---|
    | lead | **0.370** (0.359–0.380) | 0.196 | 0.253 | 1.000 | 0.216 | 1.000 | 0 | 0.005 / 0.006 |
    | extractive | 0.352 (0.343–0.361) | 0.176 | 0.243 | 1.000 | 0.217 | 1.000 | 0 | 0.014 / 0.017 |
    | cpu ONNX INT8, segmentwise | 0.308 (0.300–0.317) | 0.127 | 0.207 | 0.846 | 0.176 | 0.979 | 0.171 | 13.0 / 15.4 |
    | **cpu torch INT8, segmentwise (primary)** | 0.306 (0.298–0.314) | 0.123 | 0.204 | **0.867** | 0.177 | 0.975 | 0.182 | **8.6 / 10.9** |
    Primary by domain: news 0.299, org_report 0.313; by length 300–499 0.318, 500–699 0.306, 700–1000 0.294.
    Findings:
    - The fine-tuned model in the pre-registered `segmentwise` strategy is **below lead and extractive** on
      these references (−0.064 R1 vs lead) and far from 0.40. The references (model-drafted, close to the source
      wording) favour extraction; XL-Sum taught one rephrased sentence per segment.
    - **Length compliance fails** (0.87 vs required 1.00): summaries undershoot the target (median ratio 0.177 vs
      ~0.20). This is a hard acceptance criterion and the first thing to fix (pipeline length control), together
      with the strategy choice; any tuning must use texts outside the 240 (reserve texts need references first).
    - Latency on 2 vCPUs already meets the targets for torch INT8 (8.6 s median for 300–700 words, 10.9 s all);
      ONNX INT8 is ~1.5× slower here and no better in ROUGE, so torch INT8 stays selected.
    - Still to run on a T4: public baseline and fine-tuned direct / segmentwise / hybrid (FP32), then BERTScore,
      human-evaluation sheets and the acceptance table.

27. **Why the primary system is at 0.306 (diagnosis of the 05 CPU-half outputs, 2026-10-18).** Diagnostic only:
    measured on the 240 benchmark texts to size each problem; no setting is chosen from it.
    - The ceiling is not the problem: a greedy extractive oracle (best source sentences within the length bound)
      reaches R1 0.536. Reference content is spread over the text (only 23 % of reference-matching words come
      from the first 20 % of sentences), so lead's lead is word-level overlap, not position.
    - Gap to lead (−0.064) ≈ **length ~0.015** (lead cut to our word count still scores 0.356) + **content /
      wording ~0.05**. R1 precision is close to lead (0.320 vs 0.348); recall is the gap (0.298 vs 0.399).
    - **Length control had two defects** (all 32 failures too short, none too long): per-call `min_new_tokens` is
      only 0.8× the budget and nothing tops up a short result (`trim_to_words` fills only from leftover
      sentences); per-call `max_new_tokens` (≈56 tokens for a 36-word budget) cut 16 summaries mid-sentence.
    - **The model writes XL-Sum teasers, not 20 % summaries**: 0.8 numbers per summary (references 4.5; lead 2.6),
      13 % of reference numbers recalled (lead 38 %), 28 % of key points covered (lead 41 %), novel-content rate
      0.18 (references 0.12), formulaic openings ("قالت منظمة …", "حذرت منظمة …"); beats lead on 56/240 texts.
      Causes: one-sentence highlight targets; trained on whole articles but run on 320-token segments; no
      organisational-report text in training; 6/240 summaries contain a number absent from the source.
    - Scorer: with light stemming (`facts.stem`) every system gains ~0.05 (ours 0.356, lead 0.422). Our ROUGE stays
      unstemmed (decision 17); a stemmed score may only be added as a secondary number with project-lead approval.
    - References: median length ratio 0.188, 47/240 outside our own 20 % ± 20 % bounds (to check in human review).

28. **Length fix + dev set (2026-10-18).** `length_control.drop_unfinished` removes a fragment cut off by the
    token limit; `length_control.fill: extractive` tops up a model summary that is still below its goal with
    the best-ranked (TextRank) source sentences it doesn't cover, never past the upper bound (`fill_to: lo`
    by default; `target` to be decided on the dev set). Code defaults keep the old behaviour; the config turns
    both on. `Summarizer.refinish` re-applies the post-processing to stored raw outputs (`summarize-batch` with
    `keep_parts: true`), and `eval.dev_tune` scores the post-processing variants without regenerating.
    Simulated on the benchmark outputs (diagnostic): length compliance 0.867 → 1.000, R1 +0.001 (fill to lo) or
    +0.017 (fill to target).
    **Dev set**: the 90 reserve texts (30 news + 60 org, passed the same screening, never sampled; no overlap
    with the 240). References drafted by Claude Haiku (`writer_id = claude-haiku`, guideline v1.0 §2–§5, same
    validator checks: length, numbers in source, key points ≥ 3, single paragraph), then adjusted so their
    copy share (summary bigrams found in the source) matches the benchmark references' range (0.36–0.62;
    a first Haiku pass copied far more, 0.61 mean vs 0.49). Different writer model than the benchmark
    references (Opus), so dev numbers are for comparing settings, not for predicting benchmark scores.

29. **Dev-set tuning of strategy and length post-processing (Colab CPU, 2 vCPU, torch INT8, 2026-10-18).** 90 dev
    texts, Haiku-drafted dev references (decision 28); one generation run per row group, post-processing variants
    re-applied to the stored raw outputs (`eval.dev_tune`).
    | generation | post-processing | R1 | R1 stemmed | length ok | filled share | numbers ok | novel rate | median s 300–700 |
    |---|---|---|---|---|---|---|---|---|
    | segmentwise, min_factor 0.8 | as before | 0.289 | 0.338 | 0.878 | 0 | 0.978 | 0.162 | 8.9 |
    | segmentwise, min_factor 0.8 | fill to lo | 0.292 | 0.341 | 1.000 | 0.032 | 0.978 | 0.156 | 8.9 |
    | **segmentwise, min_factor 0.8** | **fill to target** | **0.307** | 0.356 | **1.000** | 0.206 | 0.978 | 0.127 | 8.9 |
    | segmentwise, min_factor 1.0 | as before / fill to target | 0.291 / 0.293 | 0.339 / 0.341 | 0.989 / 1.000 | 0 / 0.046 | 0.989 | 0.172 | 10.1 |
    | hybrid, min_factor 0.8 | as before / fill to target | 0.289 / 0.301 | 0.334 / 0.348 | 0.744 / 1.000 | 0 / 0.234 | 0.956 | 0.169 | 6.6 |
    | lead / extractive | – | 0.355 / 0.329 | 0.406 / 0.379 | 1.000 | – | 1.000 | 0 | – |
    - **Selected: segmentwise + `fill_to: target`** (config). Length compliance 1.000, +0.018 R1; about a fifth of
      the words are source sentences added by the fill, so reports must describe the output as model summary
      plus extractive completion.
    - Asking the model for more words (min_factor 1.0) fixes length but adds no ROUGE: the extra words are new
      wording (novel rate 0.17). Hybrid is worse: the model rewrites the selected sentences into its XL-Sum teaser
      style, and numbers_ok drops to 0.956. Conclusion: with this model, pipeline settings top out at ~0.30–0.31;
      the gap to lead is the model's training target (decision 27), so the next step is retraining on
      multi-sentence 20 % summaries.
    - Gemma 3 1B Q4_K_M zero-shot, for reference (20 benchmark texts, same 2 vCPUs, diagnostic): R1 0.318 vs ours
      0.306 on the same texts, length ok 0.45, numbers ok 0.85, median 91.5 s (ours 6.8 s), 1.3 GB RSS. Not viable.

30. **Secondary ROUGE: XL-Sum-style Snowball stemming (2026-10-18). Needs project-lead approval to count.**
    The proposal fixes "ROUGE-1 F1 > 0.40" without defining tokenization; decision 17 fixed our primary score as
    unstemmed. Published Arabic numbers (XL-Sum, model cards) use the XL-Sum authors' multilingual scorer, which
    stems Arabic with NLTK's Snowball stemmer (`multilingual_rouge_scoring/stemmers.py`). 05 now also reports
    `rouge*_xlsum` (`evaluate.secondary_rouge_stem: snowball`; words stemmed in their own spelling, then our
    matching folds applied to the stems; close to, not identical with, that scorer). The acceptance table keeps
    pass/fail on the unstemmed score and lists the stemmed one as a secondary row with pass left open. Because
    this is decided after seeing results, any report must show both scores; the stemmed one may replace the
    strict one as the acceptance metric only if the project lead approves.
    Measured on the 05 CPU-half outputs (240 texts, interim references): ours as run 0.306 strict / 0.367 xlsum;
    ours re-processed with the current fill (approximation) 0.319 / 0.382; lead 0.370 / 0.430; extractive
    0.352 / 0.415. Pinned `nltk==3.9.1` (requirements/colab-data.txt).

31. **03b style fine-tune on a public dataset (prepared 2026-10-18; not run yet, needs a T4).** The user chose
    public data over LLM-generated training summaries for now (own dataset in the next project). Open Arabic
    summarization sets checked (style vs our references: ratio 0.19, ~4 sentences, copy share 0.49):
    karimraouf/AsDs (2k, 214-word inputs, LLM meta-style), BounharAbdelaziz synthetic (3.5k, mostly WikiHow),
    Abdelkareem 30k (94-word inputs, 99 % copied), **oddadmix/arabic-news-summarization** (17.9k Arabic machine
    translation of CNN/DailyMail; 458-word inputs, ~4-sentence highlights, ratio 0.11, numbers_ok 0.83; no
    license stated: research use, deviation to report). Only the last fits; none overlaps the 330 benchmark/dev
    texts (8-gram check). Filter (`train.external.prepare_pairs`, same faithfulness rule as XL-Sum train):
    17,862 → 9,037 pairs. `notebooks/03b_style_finetune.ipynb` continues `arabart_ft` (lr 3e-5, ≤ 3 epochs,
    patience 1), selects the epoch on dev ROUGE-1, exports `models/arabart_ft_style`, and compares old vs new
    model on the dev set through the pipeline (segmentwise / direct, strict + XL-Sum-style ROUGE).
    Expected effect (estimate, not measured): +0.01–0.03 R1; only if the dev result beats the current model does
    it go through 04 and 05.

32. **03b result: style model beats lead on the dev set (Colab T4, 2026-10-18, 0.26 T4-hours).** 9,037 filtered
    oddadmix pairs, 0 leaked; lr 3e-5, 3 epochs, best epoch 2 by dev R1 (training-time dev R1, whole article,
    greedy, ~50-word outputs: 0.239 / **0.256** / 0.248). Exported `models/arabart_ft_style` (534.6 MB).
    Dev set through the pipeline (GPU fp16, configured fill to target; strict / XL-Sum-style R1):
    | model | strategy | R1 | R1 xlsum | length ok | filled share | numbers ok | novel rate |
    |---|---|---|---|---|---|---|---|
    | arabart_ft | segmentwise | 0.306 | 0.369 | 1.000 | 0.188 | 0.989 | 0.121 |
    | arabart_ft | direct | 0.307 | 0.363 | 1.000 | 0.243 | 0.956 | 0.120 |
    | arabart_ft_style | segmentwise | 0.343 | 0.405 | 1.000 | 0.152 | 0.967 | 0.015 |
    | **arabart_ft_style** | **direct** | **0.360** | **0.418** | 1.000 | 0.177 | 0.989 | 0.017 |
    | lead / extractive | – | 0.355 / 0.329 | 0.416 / 0.390 | 1.000 | 0 | 1.000 | 0 |
    - **Selected for 04/05: `arabart_ft_style` + `direct`** (dev set; `optimize.source_model`, `evaluate.ft_model`,
      `evaluate.primary_strategy: direct`; segmentwise and hybrid become ablations). Chosen before any benchmark
      run of this model. The 03 CPU model stays on Drive (`arabart_ft_cpu`); 04 writes `arabart_ft_style_cpu`.
    - The style model copies heavily: novel-content rate 0.017 (references 0.12), i.e. mostly compressed source
      sentences. Good for factual accuracy; fluency/coherence and "abstractive" character must be judged in the
      human evaluation and stated in the report. Dev references are Haiku-drafted, benchmark references
      Opus-drafted: the benchmark number may differ.

33. **05, GPU half with the style model (Colab T4, fp32, 2026-10-18; INTERIM: 240 model-drafted references, none
    human-approved).** Code 9d1059b.
    | system | R1 (95 % CI) | R1 xlsum (95 % CI) | R2 | RL | length ok | ratio | numbers ok | novel rate |
    |---|---|---|---|---|---|---|---|---|
    | **ft_style_direct (primary model)** | **0.377** (0.367–0.387) | **0.437** (0.427–0.447) | 0.194 | 0.251 | 1.000 | 0.220 | 0.992 | 0.020 |
    | lead | 0.370 (0.359–0.380) | 0.430 (0.420–0.441) | 0.196 | 0.253 | 1.000 | 0.216 | 1.000 | 0 |
    | ft_style_segmentwise | 0.369 (0.360–0.378) | 0.432 (0.423–0.441) | 0.187 | 0.250 | 1.000 | 0.218 | 0.996 | 0.017 |
    | ft_style_hybrid | 0.361 (0.350–0.371) | 0.423 (0.413–0.432) | 0.177 | 0.238 | 1.000 | 0.216 | 0.992 | 0.018 |
    | extractive | 0.352 (0.343–0.361) | 0.415 (0.405–0.424) | 0.176 | 0.243 | 1.000 | 0.217 | 1.000 | 0 |
    | baseline_direct (Omar10lfc) | 0.317 (0.309–0.326) | 0.376 (0.367–0.384) | 0.138 | 0.191 | 1.000 | 0.220 | 0.888 | 0.148 |
    Paired vs the public baseline: ft_style_direct +0.060 R1 (95 % CI +0.051..+0.070), segmentwise +0.052,
    hybrid +0.044. vs the 03 model on CPU (decision 26): 0.306 → 0.377 strict, 0.367 → 0.437 XL-Sum-style.
    - Strict R1 0.377 < 0.40; XL-Sum-style R1 0.437 > 0.40 with the whole CI above it. Whether the criterion is met
      depends on the project lead accepting the stemmed score (decision 30). Length compliance 1.00 and numbers_ok
      0.99 meet their targets. The dev-set ranking (direct > segmentwise > hybrid) held on the benchmark.
    - Still open: 04 for the style model (INT8, latency, ≤ 0.02 R1 drop), the CPU half of 05 (deployed variant),
      BERTScore, human evaluation (fluency/coherence of a mostly-extractive output), reference review.

34. **04 + 05 CPU half for the style model (Colab CPU, 2 vCPU Xeon 2.20 GHz, 2 threads, 2026-10-18/19, code cd537b3).**
    04 (`direct` strategy; validation R1 is vs one-sentence XL-Sum highlights, only for the FP32→INT8 drop):
    | variant | size MB | RSS MB | median / p90 s | val R1 | drop | gates |
    |---|---|---|---|---|---|---|
    | **torch_int8 (selected)** | 531.3 | 1881 | **12.8** / 20.2 | 0.1801 | 0.002 | all pass |
    | torch_fp32 | 531.3 | 1437 | 22.0 / 38.1 | 0.1818 | – | latency |
    | onnx_fp32 | 1310.2 | 2420 | 13.7 / 23.9 | 0.1818 | 0 | size |
    | onnx_int8 | 329.3 | 1333 | 17.0 / 26.1 | 0.1729 | 0.009 | latency (2 vCPU) |
    Exported `models/arabart_ft_style_cpu` (torch INT8, runtime.json strategy `direct`).
    **05 acceptance, primary `cpu_torch_int8_direct` (240 texts; INTERIM, references model-drafted, pending review):**
    | criterion | value | target | pass |
    |---|---|---|---|
    | ROUGE-1 F1 (primary, unstemmed) | 0.376 (95 % CI 0.367–0.387) | > 0.40 | **no** |
    | ROUGE-1 F1, XL-Sum-style stemming (secondary) | 0.436 (0.426–0.445) | > 0.40 | yes, if approved (decision 30) |
    | ROUGE-2 / ROUGE-L | 0.194 / 0.247 | reported | – |
    | length 10–30 %, 30–300 words | 1.000 | all texts | yes |
    | median latency 300–700 / all (2 vCPU) | 10.4 s / 12.9 s | ≤ 15 / ≤ 25 s | yes (reference machine: bench/) |
    | RAM / model size | 1881 MB / 531 MB | ≤ 8192 / ≤ 1024 MB | yes |
    | factual (auto pre-screen numbers_ok) | 0.996 | human ≥ 0.8 | pending human evaluation |
    | human scores ≥ 4 | – | ≥ 4 each | pending |
    INT8 vs FP32 on the benchmark: 0.3769 → 0.3763 (−0.0006). vs public baseline +0.059 R1 (95 % CI +0.050..+0.069).
    By domain: news 0.362 / 0.423 xlsum, org_report 0.391 / 0.448; by length: 300–499 0.397, 500–699 0.380,
    700–1000 0.353. BERTScore F1 (bert-base-multilingual-cased): primary 0.751, ft_style_direct 0.751, segmentwise
    0.751, hybrid 0.746, extractive 0.745, lead 0.744, baseline 0.729.
    Blind human-evaluation sheets written for the primary (100 items, 50 per domain, 20 calibration):
    `logs/05_evaluate/human_eval/` (rater_A/B, rater_REF_calibration; key.csv hidden). Report: `logs/05_evaluate/report.md`.

35. **Manual review of the primary's outputs (2026-10-19, `cpu_torch_int8_direct`, 240 benchmark texts).**
    Each summary sentence was matched to its closest source sentence (word overlap). Share of summary words:
    verbatim copy 11 %, shortened copy 34 %, lightly edited 22 %, rewritten/fused (< 60 % overlap with any one
    source sentence) 14 %, extractive length fill 19 %. Model sentences come from across the article (median position
    0.20; half from the first 20 %); the first summary sentence is the first source sentence in 78/240 texts.
    The primary beats lead on 123/240 texts (mean +0.007 R1). No near-duplicate sentence pairs inside any summary.
    2.5 numbers per summary vs 4.5 in the references.
    - **Main risk: unfaithful rewritten sentences.** 14 randomly sampled rewritten/fused sentences: 6 correct,
      8 with an error. Examples: NEWS-0002 "they are not Americans" → "the dead are Americans"; ORG-0065 vaccinated
      "140 thousand children" → "500,000–600,000 people" (the number of people sheltering); NEWS-0086 a 1–0 win →
      "1–1 draw in the first half"; NEWS-0101 "less than a third of 15 thousand returned" → "about 15 thousand
      returned"; NEWS-0096 "since the coronavirus spread in 2015"; quotes attributed to the wrong person (ORG-0023,
      NEWS-0116, NEWS-0101). Rewritten sentences occur in 129/240 summaries, so the human criterion "≥ 80 % with no
      critical factual error" is at risk. `numbers_ok` (0.996) misses these: it only checks that each number appears
      in the source, not that it is used correctly. 24 % of model sentences open with a speech verb ("يقول/تقول X إن"),
      the style of the training data, and wrong attribution comes mostly from these sentences.
    - **Length fill:** 47 of 247 fill sentences repeat ≥ 50 % of a model sentence (e.g. ORG-0115, ORG-0160), and
      18 are page titles/headings without end punctuation (org reports: title line + " – اليونيسف"), which is
      why 19 summaries don't end in punctuation. Org pages with salutations / lists of names (ORG-0172, a speech)
      get copied boilerplate.
    - Possible fixes (not run): a per-sentence faithfulness filter (drop model sentences with low source support or
      a number/name used in a different sentence context, fill instead); skip unpunctuated title lines and covered
      sentences at a lower threshold in the fill; strip date/salutation lines in org preprocessing.

36. **Faithfulness filter + fill changes, tuned on the dev set (2026-10-19).** `summarize/faithful.py` checks each
    model sentence against the source sentences: content support, number context, negation, quotes, speaker
    attribution (module docstring). A rejected sentence is dropped and the fill replaces it with source sentences.
    The fill now skips heading lines and sentences already 50 % covered, and ranks sentences with numbers higher
    (×1.3). Dev raw outputs: deployed model (torch INT8, direct) on the 90 dev texts with `keep_parts`
    (`logs/dev_runs/dev_cpu_torch_int8_direct.jsonl`, Colab CPU). Variants re-applied locally with
    `dev_tune.FAITH_VARIANTS`; the benchmark was not used for any choice.
    | post-processing (dev, 90 texts) | R1 | R1 xlsum | R2 | length ok | fill share | numbers ok | rejected / text |
    |---|---|---|---|---|---|---|---|
    | before (decision 34 config) | 0.367 | 0.422 | 0.188 | 1.000 | 0.183 | 0.989 | 0 |
    | filter (support from ≤ 2 source sentences) | 0.367 | 0.423 | 0.189 | 1.000 | 0.305 | 1.000 | 0.88 |
    | + fill changes, number bonus 0.3 | 0.375 | 0.432 | 0.196 | 1.000 | 0.304 | 1.000 | 0.88 |
    | + attributed claim from one sentence | 0.377 | 0.434 | 0.197 | 1.000 | 0.319 | 1.000 | 0.98 |
    | **+ every claim from one source sentence (selected)** | **0.380** | **0.436** | 0.197 | 1.000 | 0.432 | 1.000 | 1.77 |
    | drop every model sentence (all extractive) | 0.352 | 0.411 | 0.182 | 1.000 | 1.000 | 1.000 | 5.2 |
    Number bonus 0.6 / 1.0 and fill cover 0.5 alone: within ±0.003. Both domains improve (news 0.369 → 0.374,
    org 0.366 → 0.375, with the 2-sentence-support filter).
    **Hand review on dev** (model sentences that aren't ≥ 85 % copies of one source sentence: 193 of 468):
    - 2-sentence-support filter: ~9 of 15 sampled rejected sentences were really wrong. 7 of 20 kept were still
      wrong, mostly facts fused from two source sentences under one actor ("109 million fell ill" → "died";
      al-Zahar's pledge → John Kerry; UNICEF's delivery → KSrelief).
    - Selected (one-sentence support): 39 non-copy sentences kept; 3 of 20 sampled wrong (wrong governor/minister
      title from a fused page title, "violence against women" → "sexual violence", KSrelief/UNICEF role).
      Near-copies (≥ 85 %) that drop ≥ 3 source words: 1 of 15 wrong (garbled Obama/Sarkozy quote).
    - Rough estimate for the 90 dev summaries: ~95 wrong model sentences before → ~15 after (−80–85 %); small
      samples, single reviewer (Claude), so the human evaluation remains the measure. What still gets through:
      changed roles of correctly named actors, garbled near-copies, dropped qualifiers.
    - Cost: summaries are more extractive (43 % of words from the fill, model-rewritten sentences ~8 % of model
      sentences). The model's selection and compression still help: all-extractive scores 0.352.
    Config: `faithfulness.enabled: true`, `max_support_sentences: 1`, `min_support: 0.8`,
    `attribution_single_source: true`; `fill_cover_threshold: 0.5`, `fill_skip_titles: true`, `fill_number_bonus: 0.3`.
    Next: re-run the CPU primary of 05 on the 240 texts (old outputs and the unused human-eval sheets moved to
    `archive_decision34/`), new human-eval sheets. The GPU ablations (`ft_style_*`) keep the decision-34 post-processing.
    **05 CPU re-run (commit aeb0afa, Colab 2 vCPU; INTERIM references):** primary `cpu_torch_int8_direct` R1
    **0.382** (95 % CI 0.372–0.393; was 0.376), xlsum **0.443** (0.433–0.453), R2 0.197, RL 0.240, length 1.000,
    numbers_ok 1.000, novel rate 0.009, median latency 10.6 s (300–700) / 13.4 s (all). vs baseline +0.065 R1.
    1.84 model sentences dropped per text (191/240 texts); fill share 0.458; numbers per summary 2.5 → 3.0.
    Per text vs decision 34: R1 better on 103, same on 53, worse on 84.
    Spot check (Claude): the known errors are gone (NEWS-0002 "are Americans", ORG-0065 500–600 thousand vaccinated,
    NEWS-0086 draw, NEWS-0096 2015, ORG-0023 Dr Rahaf quote, NEWS-0101 "15 thousand returned"; NEWS-0079 a
    Kılıçdaroğlu quote given to the Moscow Patriarchate). False rejections of correct sentences: negation window crossing
    a clause (ORG-0139 "لا داعي له، تدعم"), speaker introduced > 2 sentences earlier (NEWS-0083). Fill weaknesses:
    rhetorical questions, datelines ("1 تشرين الأول/ أكتوبر 2026 -"), heading lines glued to a sentence.
    Not run yet: BERTScore, human-eval sheets, report (cells 7/8/10) — pending a second tuning round on these points.

37. **Second filter / fill round, on the dev set (2026-10-19, commit e76cac2).** Fixes for the decision-36 spot check:
    - negation: the governed words stop at the next punctuation ("دون تأخير لا داعي له، تدعم" no longer negates
      "تدعم"; ORG-0139's correct lead sentence now passes);
    - attribution: window 2 → 3; `attribution_name_match` (speaker named by a rare word in the latest speaker
      phrase near the support sentence, description found earlier: "قال ليونز إن" → "الخبير الاقتصادي البريطاني
      جيرارد ليونز"); `attribution_latest_speaker` (rejected if that latest speaker phrase names someone else);
    - fill: `fill_skip_questions`; `fill_strip_leads` (datelines in the first 3 sentences, quoted crossheads such
      as '"مهمة مجتمع" وعوضا'); `fill_fix_quotes` (a quote split across sentences is closed / opened when a
      speaker is named, else the piece is skipped). The number bonus is still computed on the original sentence.
    Dev (90 texts, `dev_tune.ROUND2_VARIANTS`): decision 36 R1 0.380 / xlsum 0.436 / R2 0.197 → selected
    (`clean+name+win3+latest`) 0.379 / 0.435 / 0.198, length 1.000, numbers 1.000; ROUGE-neutral (within ±0.002),
    chosen by reading the changed sentences. Windows 4 / unlimited were rejected: they also kept wrong speakers
    (Kenyatta for "Kenyan officials", the Eurogroup head for a Greenspan quote). Speaker rules: 3 correct sentences
    recovered (Lyons, Criado Perez, Chepkoech), 1 borderline one newly rejected (NEWS-0031 "Israel says" for
    "Netanyahu said the report shows"). Fill cleanup changed 31 dev texts: questions, datelines and crossheads gone,
    ~20 split quotes repaired. Still not caught: unquoted headings glued to a sentence ("اضطراب الساعة البيولوجية
    إنه ..."), datelines without a digit/comma later in a text.
    **05 CPU re-run (commit 9d9bba7, Colab 2 vCPU AMD EPYC 7B12; INTERIM references; decision-36 outputs and report
    in `archive_decision36/`; raw model outputs kept this time, `keep_parts`):** primary `cpu_torch_int8_direct` R1
    **0.376** (95 % CI 0.367–0.386), xlsum **0.438** (0.429–0.448), R2 0.192, RL 0.240, length 1.000, numbers_ok
    1.000, novel 0.008, BERTScore F1 0.751 (ft_style_direct 0.751, extractive 0.745, lead 0.744, baseline 0.729),
    median latency 6.1 s (300–700) / 7.6 s (all), +0.059 R1 vs baseline. Acceptance: ROUGE-1 > 0.4 still fails
    (xlsum passes if approved); length, latency, RAM, size pass; human criteria pending.
    - **The INT8 model's text depends on the CPU.** Decision 36 ran on an Intel Xeon 2.20 GHz, this run on an AMD EPYC:
      only 81 of the 442 model sentences rejected in decision 36 occur again, 25 / 240 summaries are identical when
      the decision-36 post-processing is re-applied, and generation is ~35 % faster. On the same raw outputs the
      decision-36 settings give R1 0.378 and the new ones 0.376 (−0.0015; dev −0.0007), so most of the 0.382 → 0.376
      change is the CPU, not this round. ROUGE of the deployed system can move by ~±0.005 between machines: the
      official numbers must come from the reference machine (`bench/`), and dev tuning should use outputs from it.
    - Fill problems in the 240 summaries (decision 36 → 37): questions 15 → 6, sentences starting with a dateline /
      crosshead 45 → 6, split-quote pieces 59 → 22 (the rest are in model-written sentences, which the fill cleanup
      doesn't touch). ORG-0139 keeps its correct lead sentence (R1 0.253 → 0.376); NEWS-0083 has no questions and
      the wrong Roenneberg attribution is still dropped. On the new raw outputs the filter again dropped the new
      errors (NEWS-0002 "الجنود الأمريكيين ليسوا أمريكيين", "148 American soldiers since 2001"; NEWS-0079 the
      Kılıçdaroğlu quote given to Metropolitan Hilarion; NEWS-0086 "تعادل").
    - Still seen: model sentences with a pronoun speaker ("ويضيف أن ...", "تقول: ..." at the start of a summary) are
      not checked for attribution; NEWS-0101 "المدير السابق للمتحف" title; first-person quote text without quote
      marks in the fill (ORG-0065 "مسقط رأسي هلسنكي").
    Human-eval sheets written for this run (100 items, `logs/05_evaluate/human_eval/`); report.md / report.json updated.
38. **Project lead approved the XL-Sum-style ROUGE-1 as the acceptance metric (2026-10-20).** Decision 30's
    stemmed score (`rouge1_xlsum`, Snowball, as in the XL-Sum authors' scorer) now decides "ROUGE-1 F1 > 0.40":
    `evaluate.acceptance.rouge1_metric: xlsum`; `BE.acceptance` puts pass/fail on it and lists the unstemmed
    score as a "reported" row (every report still shows both; `rouge1_metric: strict` restores the old table).
    On the current run (decision 37, INTERIM references) 0.438 (CI 0.429–0.448) > 0.40: passes, with the whole
    CI above the target; still subject to the reference review and the reference-machine run. The Drive
    `report.md` predates this and shows the old table until 05's acceptance/report cells are re-run.

39. **Third filter / fill round, on the dev set (2026-10-20).** Fixes for decision 37's open points
    (`dev_tune.ROUND3_VARIANTS`, refinish of the 90 dev raw outputs, Intel):
    - **Pronoun speakers** (`attribution_pronoun`): a model sentence opening with a speech verb and no speaker
      ("ويضيف أن ...", "تقول: ...") is rejected if it opens the summary, if its verb gender differs from the
      summary's latest named speaker ("تقول بتول ..." then "يقول: ..."), or if the latest source speaker before its
      support sentence isn't that speaker (or, without one, isn't mentioned in the summary). Speaker parsing fixed
      on the way: "قالت إنها ..." has no speaker (the speaker phrase used to swallow the claim), an addressee or
      filler is no speaker ("قالت لصندوق الأمم المتحدة", "ويضيف في حوار مع بي بي سي", "وأضاف قائلا"), and the
      summary's speaker is matched on the whole sentence (a name after a long title).
    - **Fill**: skips sentences whose speaker is only a pronoun ("وأوضح أن ...") unless the source's speaker is
      named in the summary (`fill_skip_pronoun_speaker`), first-person sentences outside quotes ("التقيت بأمهات",
      `fill_skip_first_person: singular`; "all" also counts "نحن", an agency's voice: same on dev, not chosen),
      headings glued to a sentence ("اضطراب الساعة البيولوجية إنه ...", narrow pattern, `fill_skip_glued_headings`);
      removes the BBC widget text "مواضيع قد تهمك نهاية" (111 occurrences in the 330 texts), list bullets, and
      datelines after a press release's "التاريخ:" / "[بيان صحفي]" lines (`fill_strip_leads_extended`).
    | dev variant (90 texts) | R1 | R1 xlsum | R2 | rejected / text |
    |---|---|---|---|---|
    | decision 37 | 0.3791 | 0.4345 | 0.1981 | 1.74 |
    | + pronoun-speaker check | 0.3792 | 0.4346 | 0.1982 | 1.76 |
    | + fill cleanup (leads, headings, first person) | 0.3791 | 0.4346 | 0.1983 | 1.76 |
    | **+ fill pronoun speakers (selected)** | **0.3778** | **0.4336** | 0.1973 | 1.76 |
    | + latest speaker by description share (`attribution_latest_share`) | 0.3776 | 0.4335 | 0.1974 | 1.80 |
    The fill rule costs 0.0013 R1: the skipped sentences are informative but misattributed out of context (NEWS-0142
    "وتوضح ..." is Criado Perez's, the summary's last speaker is Amber Rudd; NEWS-0139 Rowley never named). The
    description-share variant drops the correct Lyons / Criado Perez sentences again (their full description isn't
    in a speaker phrase), so it stays off. Diagnosis on the decision-37 benchmark raw outputs (not tuned on):
    37 of 240 summaries change, R1 0.3763 -> 0.3764, xlsum 0.4382 -> 0.4393; 24 model sentences newly rejected, 3
    newly kept. Newly caught real misattributions: NEWS-0072 Miliband's statement given to Nick Clegg, NEWS-0147
    Navalny's claim given to Putin ("يقول بوتين إنه ..."), NEWS-0092 the Scottish environment agency -> "هيئة الصحة
    البريطانية", NEWS-0106 the FSA spokesman's statement -> "رئيس الائتلاف"; dangling "وقال إن ..." at a summary's
    start (NEWS-0041/43/47/55/83). Newly kept: NEWS-0130 (2), NEWS-0009 (correct). Pronoun model sentences in the
    240 summaries 29 -> 10 (the rest follow the right speaker). Still missed: NEWS-0034's first sentence gives the
    US Army spokesman's figures to "المتحدث باسم الحرس الثوري" (the generic "المتحدث باسم" matches).
    `summarize-refinish SPEC IN OUT` (run_batch.refinish) re-applies the post-processing to a `keep_parts` run.
    **05 primary refinished on Colab (commit 72886a9, 2026-10-20; INTERIM references):** the decision-37 raw outputs
    (AMD run) with the new post-processing; decision-37 outputs, report and the unused rater sheets (no scores
    filled) moved to `archive_decision37/`. 37 / 240 summaries changed (rejected model sentences 436 -> 457), same as
    the local refinish. R1 0.376 (CI 0.367–0.386), **XL-Sum-style 0.439 (0.430–0.448), passes > 0.40** (decision 38),
    R2 0.192, RL 0.239, length 1.000, numbers_ok 1.000, novel 0.008, BERTScore 0.751; latency unchanged (same
    generation: 6.1 s / 7.6 s median), RAM 1881 MB, size 531 MB. All automatic criteria pass; human criteria pending.
    New blind human-evaluation sheets written (100 items, `logs/05_evaluate/human_eval/`); report.md / .json updated.

40. **Summary sentences in source order (2026-10-20).** The fill appended source sentences by TextRank rank after
    the model's sentences, so summaries jumped back and forth (coherence is a human criterion, >= 4).
    `length_control.order: source` (`postprocess.source_order`) sorts the summary by the source sentence each
    sentence shares most words with (ties: fewest extra words, so a copied sentence maps to itself); a sentence
    whose speaker is only a pronoun stays right after the sentence it followed, an unfinished last piece stays last.
    Dev (90): R1 / xlsum unchanged (0.3778 / 0.4336), RL 0.239 -> 0.269, 70 / 90 summaries reordered, none opens
    with a pronoun speaker. Benchmark raw outputs (diagnostic): RL 0.239 -> 0.263, 195 / 240 reordered.
    Known limit: sentences with an implicit subject but no listed speech verb ("وكمثال على ذلك تستشهد ...") can land
    after another speaker (they could before too).

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
- 03 → 03b → 04 → 05 have all run (decisions 24–34). Docs: `README.md`, `docs/architecture.md`, `docs/README.fa.md`.
- Team: human review of all 240 reference drafts (guideline §1a/§6), then `benchmark-validate --strict`,
  `approved_only: true`, re-run 05 for final (non-interim) numbers.
- Two raters fill `logs/05_evaluate/human_eval/rater_A.csv` / `rater_B.csv`; then 05 cell 9 + acceptance.
- Project lead: XL-Sum-style ROUGE approved (decision 38); still to approve: AI-drafted references, oddadmix (no
  license, research use), the extractive length fill (~47 % of words after the faithfulness filter).
- Official numbers from the reference machine: the INT8 model's output depends on the CPU (decision 37), so run 05's
  primary there (or on one fixed CPU type) and re-tune post-processing on dev outputs from the same machine.
- `bench/` on the 8-core reference machine (official latency; torch vs ONNX INT8).
