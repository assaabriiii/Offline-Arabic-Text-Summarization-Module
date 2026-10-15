# Architecture

## Inference path

```
Arabic text (300–1000 words)
  │  text.normalize   model policy: NFKC, no tatweel/diacritics, Persian ی/ک folded
  │  text.segment     paragraphs → sentences
  │  summarize.length target = ratio × words (10–30 %, 30–300 words) → hard bounds
  ▼
summarize.pipeline.Summarizer  (one strategy per call)
  ├─ segmentwise  sentence-aligned segments ≤ 320 tokens, each gets a word budget   ← default
  ├─ direct       ≤ 900-token chunks, ≤ 80-token overlap (proposal §11.3)
  ├─ hybrid       TextRank keeps ~1.6× the budget, then abstractive rewrite
  ├─ extractive   TextRank only (no model)
  └─ lead         first sentences (no model)
  │  word budgets → decoder min/max new tokens (per-tokenizer tokens-per-word calibration)
  ▼
backend (Seq2SeqBackend protocol: count_tokens, generate)
  ├─ backends.hf.HFBackend    PyTorch FP32, or dynamic INT8 on nn.Linear
  └─ backends.ort.ORTBackend  ONNX Runtime FP32 / dynamic INT8 (optimum)
  ▼
aggregation: sentence split → near-duplicate removal → document order → trim to the
hard maximum (and fill to the minimum at a word boundary)
  ▼
eval.facts check: numbers / content words not supported by the input are reported
```

The CLI, web UI and `bench/` all use one `app.SummarizerService`. It loads a self-contained
model folder (`runtime.json`, `decoding.json`, tokenizer, weights) with the Hugging Face offline
switches set.

## Training and selection path (Colab)

| Step | Module | Key rule |
|---|---|---|
| XL-Sum acquisition | `data.xlsum`, `data.prepare` | pinned dataset commit, archive SHA256 verified |
| Dedup / leakage | `data.dedup`, `data.benchmark` | exact + MinHash (Jaccard ≥ 0.8); benchmark never overlaps train/validation |
| Benchmark | `data.benchmark`, `data.web_sources` | 240 texts, 120 news (XL-Sum test) + 120 org reports, balanced over 3 length bins |
| Train filter | `train.data` | highlight numbers must occur in the article; novel-content rate ≤ 0.35 |
| Fine-tuning | `train.seq2seq` | Seq2SeqTrainer, LR {1e-5, 3e-5, 5e-5}, ≤ 5 epochs, effective batch 16, patience 2, seed 42, resumable on Drive |
| Model / decoding selection | `train.select`, `train.decoding` | best ROUGE-1 among candidates within 0.02 of the best number precision |
| Provenance | `train.provenance`, `train.ledger` | file SHA256s, base revision, data + filter records, metrics, commit, GPU hours |
| CPU optimization | `optimize.*` | each variant measured in a fresh process; gates (≤ 1 GB, ≤ 8 GB RSS, median ≤ 15 s, ROUGE-1 drop ≤ 0.02), then 50/30/10/10 score |
| Evaluation | `eval.*` | Arabic-safe ROUGE with bootstrap CIs, fact check, latency bands, blind human evaluation with weighted kappa |

## Environments

| Environment | Python | Key versions | Used for |
|---|---|---|---|
| Colab training | 3.13 (Colab image) | transformers 5.x, torch (CUDA) | 01–03, 05 model runs |
| ONNX / offline runtime | 3.13 | transformers 4.57.6, optimum-onnx 0.1.0, onnxruntime 1.31, Gradio 5.50 | ONNX export (04), ONNX inference, MVP, `bench/` |

optimum-onnx 0.1.0 requires transformers 4.x, and optimum fails on Python 3.14. Notebooks therefore
drive the ONNX environment through subprocesses (`python -m arabic_summ onnx-export | optimize-measure |
summarize-batch`).

## Why these choices (summary; details in PROGRESS.md)

* **Length**: XL-Sum highlights average about 7% of the article, below the 10% floor. Segment-wise
  summarization makes output length scale with input length, so a model trained on short highlights
  can meet 10–30%.
* **Faithfulness**: the 02 smoke run showed a public XL-Sum model inventing facts. Training pairs are
  filtered, the fact checker takes part in model and decoding selection, and every output reports
  unsupported numbers.
* **No test-set tuning**: decoding is tuned on XL-Sum validation, and the primary strategy is fixed in
  config before any benchmark scoring.
