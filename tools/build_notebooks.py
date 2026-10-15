"""Generate notebooks/*.ipynb so every notebook shares the same setup cells.

Edit the cell sources here, then run:  python tools/build_notebooks.py
(Outputs are not stored; notebooks are thin drivers around src/arabic_summ.)
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/assaabriiii/Offline-Arabic-Text-Summarization-Module.git"


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def notebook(cells: list[dict], gpu: bool = False) -> dict:
    meta = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    }
    if gpu:
        meta["accelerator"] = "GPU"
        meta["colab"]["gpuType"] = "T4"
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}


def setup_cells(tag: str, requirements: str) -> list[dict]:
    return [
        code(f'''
# ==== 0. Settings, pinned install, environment ==========================================
REPO_URL   = "{REPO_URL}"
REPO_REF   = "main"     # branch or commit SHA; use a SHA for a fully reproducible run
CONFIGS    = ["configs/default.yaml"]          # later files override earlier ones
DRIVE_ROOT = "/content/drive/MyDrive/arabic_summ"
REQUIREMENTS = "{requirements}"

import importlib, os, subprocess, sys
REPO_DIR = "/content/arabic_summ_repo"

def sh(cmd, secret=None):
    shown = cmd.replace(secret, "***") if secret else cmd
    print("$", shown)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError((r.stdout + r.stderr).replace(secret, "***") if secret else r.stdout + r.stderr)

token = None
try:  # private repo: add a Colab secret named GITHUB_TOKEN (key icon in the left bar)
    from google.colab import userdata
    token = userdata.get("GITHUB_TOKEN")
except Exception:
    pass
url = REPO_URL.replace("https://", f"https://{{token}}@") if token else REPO_URL
if not os.path.isdir(REPO_DIR):
    sh(f"git clone -q {{url}} {{REPO_DIR}}", token)
sh(f"git -C {{REPO_DIR}} fetch -q {{url}} {{REPO_REF}} && git -C {{REPO_DIR}} checkout -q --detach FETCH_HEAD", token)
sh(f"pip install -q -r {{REPO_DIR}}/{{REQUIREMENTS}}")
sh(f"pip install -q --no-deps -e {{REPO_DIR}}")
sys.path.insert(0, f"{{REPO_DIR}}/src"); importlib.invalidate_caches()
# a reattached runtime may still hold an older copy of the package in memory: drop it
for _m in [m for m in sys.modules if m == "arabic_summ" or m.startswith("arabic_summ.")]:
    del sys.modules[_m]
os.chdir(REPO_DIR)

from arabic_summ.env import environment_report, format_report
print(format_report(environment_report()))
'''),
        code(f'''
# ==== 1. Mount Drive, load config, record environment ===================================
from google.colab import drive
drive.mount("/content/drive")

from arabic_summ.config import load_config, config_hash
from arabic_summ.env import save_environment
cfg = load_config(*CONFIGS, overrides={{"paths": {{"root": DRIVE_ROOT}}}})
save_environment(cfg, "{tag}")
print("artifact root:", cfg["paths"]["root"], "| config hash:", config_hash(cfg))
'''),
        code('''
# ==== 2. Unit tests (CPU, a few seconds) ================================================
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/"],
                   capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-2000:])
assert r.returncode == 0, "unit tests failed - stop here and paste the output"
'''),
    ]


def nb01() -> dict:
    cells = [md('''
# 01 · Data preparation

**What this does** (all outputs go to `DRIVE_ROOT`, every step is resumable and skips work already done):

1. Downloads the Arabic XL-Sum archive for a **pinned dataset commit**, verifies its SHA256.
2. Keeps the official train / validation / test splits; drops near-empty rows from *train*;
   removes from *train* every exact or near duplicate (MinHash, Jaccard ≥ 0.8) of another document.
   Validation/test are never modified, duplicates there are only flagged.
3. Builds the **internal benchmark scaffold**: news candidates drawn from XL-Sum *test*
   (300–1000 words, stratified by length, excluding any article involved in a duplicate pair),
   plus empty templates for organizational reports and human reference summaries.
4. Validates the benchmark and checks for **leakage** against everything that can influence the model
   (XL-Sum train + validation): exact hash, URL and near-duplicate matches.
5. Writes `data/processed/data_manifest.json` (dataset version, license, download date, file hashes,
   split counts, dedup stats, benchmark file hashes, library versions).

Reference summaries are **never** generated by a model; see `docs/reference_guideline.md`.

Runtime: CPU runtime is enough (no GPU needed). About 5–10 minutes on first run.
'''), *setup_cells("01_data", "requirements/colab-data.txt"), code('''
# ==== 3. Download XL-Sum (pinned revision, SHA256-verified) =============================
from arabic_summ.data import prepare
m = prepare.step_download(cfg)
print({k: m[k] for k in ("repo_id", "revision", "archive_sha256", "archive_bytes", "license", "download_date_utc")})
'''), code('''
# ==== 4. Quality filter + exact / near-duplicate removal ================================
import pandas as pd
m = prepare.step_dedup(cfg)
display(pd.DataFrame(m["splits"]).T)
'''), code('''
# ==== 5. Length statistics (why XL-Sum references alone don't fit our 10-30 % spec) =====
rows = []
for split in ("train", "validation", "test"):
    df = prepare.load_processed(cfg, split)
    ratio = df.summary_words / df.text_words
    in_range = df.text_words.between(cfg["benchmark"]["input_min_words"], cfg["benchmark"]["input_max_words"])
    rows.append({"split": split, "docs": len(df),
                 "text_words_median": df.text_words.median(), "text_words_p90": df.text_words.quantile(.9),
                 "summary_words_median": df.summary_words.median(),
                 "summary/text_ratio_median": round(ratio.median(), 3),
                 "share_300_1000_words": round(in_range.mean(), 3),
                 "share_summary_>=30_words": round((df.summary_words >= 30).mean(), 3)})
display(pd.DataFrame(rows).set_index("split"))
'''), code('''
# ==== 6. Internal benchmark scaffold ====================================================
m = prepare.step_benchmark_scaffold(cfg)
print({k: m[k] for k in ("written", "kept_existing", "news_candidates", "news_by_length_bin", "test_articles_in_range")})
'''), md('''
### Benchmark items (rerun cells 7–9 any time; they never overwrite reviewed rows)

* **News** (cell 6): 150 XL-Sum *test* candidates.
* **Organizational reports** (cell 7): the repo ships `benchmark_sources/org_reports.csv`, a list of public
  Arabic pages from WHO EMRO, UNICEF MENA, UNFPA and UN Women Arab States (URLs only; texts are copyrighted and
  are fetched here for non-commercial research evaluation, never committed). You can also drop your own
  UTF-8 `.txt` reports into `DRIVE_ROOT/data/benchmark/incoming_org/` with a `sources.csv` (same columns as
  `external_sources_template.csv`).
* **Selection** (cell 8): automatic, seeded and documented: screening (300–1000 words, ≥ 90 % Arabic script,
  ≤ 15 % numeric tokens), then 120 per domain spread evenly over length bins and round-robin over sources.
  Unsampled items are kept as `rejected` + note `reserve`; promote one to `selected` to replace a bad pick.

**Reference summaries** (cell 8b): at the user's request the 240 references were **drafted by a model**
(`reference_source=llm_draft`, writer `claude-opus-5.5`). Upload the drafts file (never committed) to
`DRIVE_ROOT/data/benchmark/reference_drafts.json`; cell 8b fills only empty rows and leaves
`review_status=pending`. A fluent Arabic reviewer must then check every row in Google Sheets against
`docs/reference_guideline.md` §6, set `reviewer_id` and `review_status=approved` (or `needs_revision`).
Results built on these references are reported as **model-drafted, human-verified**.
'''), code('''
# ==== 7. Organizational reports: fetch the committed URL list + import manual uploads ===
from arabic_summ import config as C
from arabic_summ.data import benchmark as B
rep = prepare.fetch_org_reports(cfg, REPO_DIR)          # resumable: skips files already on Drive
print(rep)
manual = C.path(cfg, "benchmark") / "incoming_org"
manual.mkdir(parents=True, exist_ok=True)
src_csv = manual / "sources.csv"
added = B.import_text_folder(C.path(cfg, "benchmark"), manual, "org_report", cfg["benchmark"],
                             src_csv if src_csv.exists() else None)
print(f"manual uploads imported: {len(added)}")
'''), code('''
# ==== 8. Automatic, seeded item selection (120 per domain) ==============================
import json
print(json.dumps(prepare.run_auto_select(cfg), ensure_ascii=False, indent=1))
'''), code('''
# ==== 8b. Model-drafted references -> references.csv (empty rows only, status pending) ===
drafts = C.path(cfg, "benchmark") / "reference_drafts.json"
if drafts.exists():
    print(json.dumps(prepare.import_reference_drafts(cfg, drafts, writer_id="claude-opus-5.5"),
                     ensure_ascii=False, indent=1))
else:
    print("no reference_drafts.json on Drive yet: skipping")
'''), code('''
# ==== 9. Validate benchmark + leakage check (STRICT=True only when freezing the set) ====
STRICT = False
rep, leak = prepare.validate_benchmark(cfg, strict=STRICT)
print(rep.summary())
print(f"leakage vs XL-Sum train+validation: {leak['n_leaks']} of {leak['items_checked']} items")
'''), code('''
# ==== 10. Manifest summary (paste this output back) ======================================
from arabic_summ.io_utils import read_json
man = read_json(prepare.manifest_path(cfg))
print("dataset :", {k: man["dataset"][k] for k in ("repo_id", "revision", "archive_sha256", "license", "download_date_utc")})
print("files   :", {k: (v["rows"], v["sha256"][:12]) for k, v in man["source_files"].items()})
print("splits  :", {k: (v["raw_rows"], v["kept_rows"], v["matches_expected"]) for k, v in man["splits"].items()})
print("dedup   :", man["dedup"]["stats"])
print("bench   :", man["benchmark"]["validation"]["stats"], "| leakage ok:", man["benchmark"]["leakage"]["ok"])
print("code    :", man["code"])
''')]
    return notebook(cells)


def nb02() -> dict:
    cells = [md('''
# 02 · Preprocessing, chunking and the summarization pipeline

**What this does**
1. Runs the unit tests (normalization, sentence splitting, chunking, length policy, pipeline logic).
2. **Tokenizer calibration**: subword tokens per word for each candidate model's tokenizer, measured on
   XL-Sum *train* (never test). Turns word targets (10–30 % of input, 30–300 words) into decoder
   `min_new_tokens` / `max_new_tokens`. Saved to `data/processed/tokenizer_calibration.json`.
3. Measures how many benchmark texts exceed the model's direct capacity (900 / 1024 tokens).
4. Shows sentence splitting and chunking (≤ 900 tokens, ≤ 80 overlap) on one benchmark text.
5. **Pipeline smoke run** with a public XL-Sum AraBART checkpoint (baseline, not our final model) on 9 benchmark
   texts across strategies `lead`, `extractive`, `direct`, `segmentwise`, `hybrid`: output length vs. the
   10–30 % bounds, model calls and time. No ROUGE yet: human references are still being written.

Runtime: **T4 GPU** recommended (CPU works, slower). Resumable: calibration is cached on Drive.
'''), *setup_cells("02_preprocess", "requirements/colab-data.txt"), code('''
# ==== 3. Tokenizer calibration (XL-Sum train sample; cached) ============================
import pandas as pd
from arabic_summ import models as M
from arabic_summ.data import prepare
from arabic_summ.io_utils import read_json
from arabic_summ.text.normalize import policies_from_config
model_policy, _ = policies_from_config(cfg)
TOKENIZERS = {"arabart": "arabart", "arat5v2": "arat5v2"}   # fine-tuned baselines share these tokenizers
N_SAMPLE = 2000

cal_path = M.calibration_path(cfg)
cached = read_json(cal_path) if cal_path.exists() else {}
train = prepare.load_processed(cfg, "train").sample(N_SAMPLE, random_state=cfg["project"]["seed"])
tokenizers = {}
for key in TOKENIZERS:
    e = M.model_entry(cfg, key)
    tokenizers[key] = M.load_tokenizer(e["repo_id"], revision=e["revision"])
    if key in cached:
        print(f"[skip] {key}: cached"); continue
    rec = {"repo_id": e["repo_id"], "revision": e["revision"], "tokenizer_class": type(tokenizers[key]).__name__,
           "n_sample": N_SAMPLE, "sample": "xlsum train, seed 42",
           "article": M.calibrate_tokens_per_word(tokenizers[key], train.text, model_policy),
           "summary": M.calibrate_tokens_per_word(tokenizers[key], train.summary, model_policy)}
    M.save_calibration(cfg, key, rec)
cal = read_json(cal_path)
display(pd.DataFrame({k: {"tokenizer": v["tokenizer_class"], "tok/word article": v["article"]["tokens_per_word"],
                          "tok/word summary": v["summary"]["tokens_per_word"]} for k, v in cal.items()}).T)
'''), code('''
# ==== 4. How many benchmark texts exceed direct model capacity? =========================
from arabic_summ import config as C
from arabic_summ.data import benchmark as B
items = B.read_csv(C.path(cfg, "benchmark") / B.ITEMS_FILE)
items = items[items.status == "selected"] if (items.status == "selected").any() else items[items.status != "rejected"]
rows = []
for key, tok in tokenizers.items():
    n = items.text.map(lambda t: len(tok(t, add_special_tokens=False)["input_ids"]))
    for dom in sorted(items.domain.unique()):
        m = items.domain == dom
        rows.append({"tokenizer": key, "domain": dom, "texts": int(m.sum()), "tokens_median": int(n[m].median()),
                     "tokens_max": int(n[m].max()), "share_>900": round((n[m] > 900).mean(), 3),
                     "share_>1024": round((n[m] > 1024).mean(), 3)})
display(pd.DataFrame(rows))
'''), code('''
# ==== 5. Sentence splitting + chunking on the longest benchmark text ====================
from arabic_summ.text.chunk import chunk_sentences
from arabic_summ.text.normalize import normalize
from arabic_summ.text.segment import split_sentences
tok = tokenizers["arabart"]
count = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])
longest = items.loc[items.word_count.astype(int).idxmax()]
sents = split_sentences(normalize(longest.text, model_policy))
print(longest.sample_id, longest.word_count, "words |", len(sents), "sentences")
for name, mx in [("direct (900/80)", 900), ("segmentwise", cfg["pipeline"]["segmentwise"]["segment_max_tokens"])]:
    ch = chunk_sentences(sents, count, mx, cfg["pipeline"]["chunking"]["overlap_tokens"],
                         cfg["pipeline"]["chunking"]["paragraph_break_min_fill"])
    print(f"{name:18s}: {len(ch)} chunks, tokens={[c.n_tokens for c in ch]}, overlap sentences={[c.n_overlap_sentences for c in ch]}")
print("\\nfirst 5 sentences:"); [print("  -", s.text[:120]) for s in sents[:5]]
'''), code('''
# ==== 6. Pipeline smoke run with a public XL-Sum baseline ================================
import json, time, torch
from arabic_summ.summarize.pipeline import Summarizer
BASELINE = "arabart_xlsum_omar"            # see configs/default.yaml -> models.baselines
STRATEGIES = ["lead", "extractive", "direct", "segmentwise", "hybrid"]
RATIO = cfg["pipeline"]["default_ratio"]

backend = M.load_hf_backend(cfg, BASELINE, dtype="float16" if torch.cuda.is_available() else "float32")
summ = Summarizer(cfg, backend=backend, tokens_per_word=M.tokens_per_word(cfg, "arabart"))
print(backend.name, "| device:", backend.device, "| tokens/word:", summ.tokens_per_word)

demo = (items.groupby(["domain", "length_bin"], group_keys=False)
             .apply(lambda g: g.sample(min(len(g), 2), random_state=0)).head(9))
out_dir = C.path(cfg, "logs") / "02_pipeline_demo"; out_dir.mkdir(parents=True, exist_ok=True)
records = []
for _, r in demo.iterrows():
    for st in STRATEGIES:
        res = summ.summarize(r.text, ratio=RATIO, strategy=st)
        records.append({"sample_id": r.sample_id, "domain": r.domain, **{k: v for k, v in res.to_dict().items() if k != "parts"}})
pd.DataFrame(records).to_json(out_dir / "results.jsonl", orient="records", lines=True, force_ascii=False)
df = pd.DataFrame(records)
display(df.groupby("strategy").agg(length_ok=("length_ok", "mean"), words=("output_words", "median"),
                                   target=("target_words", "median"), calls=("n_model_calls", "mean"),
                                   seconds=("seconds", lambda s: round(pd.Series([x["total"] for x in s]).median(), 2))))
'''), code('''
# ==== 7. Read one example per strategy (paste this back) ================================
ex = demo.iloc[0]
print(ex.sample_id, ex.domain, ex.word_count, "words\\n")
for st in STRATEGIES:
    r = next(x for x in records if x["sample_id"] == ex.sample_id and x["strategy"] == st)
    print(f"--- {st}: {r['output_words']} words (bounds {r['bounds']}), {r['seconds']['total']}s\\n{r['summary']}\\n")
''')]
    return notebook(cells, gpu=True)


def nb03() -> dict:
    cells = [md('''
# 03 · Fine-tuning AraBART (+ one AraT5v2 comparison run) on faithfulness-filtered XL-Sum

**What this does** (proposal stage 4; PROGRESS decisions 2, 3, 11, 17)
1. Checks the GPU and prints the **GPU-hour ledger** (every training session is logged to `logs/gpu_ledger/`).
2. Runs the **fact checker** (`eval.facts`: unsupported numbers + novel content words) on the 02 smoke
   outputs to show it catches the hallucinations found there.
3. **Filters XL-Sum train**: drops pairs whose highlight has a number missing from the article or too many
   unsupported content words. Validation / test are untouched. Cached on Drive.
4. **LR grid** for AraBART: 1e-5 / 3e-5 / 5e-5, ≤ 5 epochs, effective batch 16, early stopping (patience 2),
   seed 42, on a seeded 25 % of the filtered train set (T4 budget); per-epoch validation ROUGE + fact check
   on 500 fixed validation articles. Selection: best ROUGE-1 among runs within 0.02 of the best number precision.
5. **Final AraBART run** with the selected LR on 100 % of the filtered train set.
6. **AraT5v2** single comparison run (fp32, same 25 % subset) vs. the AraBART run with the same LR.
7. **Decoding tuning** on 300 other validation articles: beams 1–4 × length penalty 1/1.5/2, no-repeat 3.
8. **Export** `models/arabart_ft/` with `decoding.json` (separate file, proposal) and `provenance.json`
   (file SHA256s, base revision, data + filter records, metrics, commit, environment, GPU hours).
9. Sanity check on 6 benchmark texts: length compliance + fact check vs. the public baseline (no ROUGE here).

**Runtime: T4 GPU.** Every run is **resumable**: if Colab disconnects, reconnect, run cells 0–1, then rerun the
cell you were in. Finished runs are skipped; an interrupted run resumes from its last epoch checkpoint on Drive.
Rough T4 estimate (not yet measured): filter ~3 min, each grid run ~0.5–1.5 h, final run ~2–5 h,
AraT5v2 (fp32) ~1–3 h, decoding grid ~20 min. Colab free sessions end after a few hours: just resume.
'''), *setup_cells("03_finetune", "requirements/colab-data.txt"), code('''
# ==== 3. GPU + ledger ==================================================================
import json, torch, pandas as pd
from arabic_summ import config as C, models as M
from arabic_summ.data import prepare, benchmark as B
from arabic_summ.eval import facts as F
from arabic_summ.io_utils import mark_step_done, read_json, sha256_file, step_done, write_json
from arabic_summ.text.normalize import NormalizationPolicy, normalize
from arabic_summ.train import data as D, decoding as DEC, ledger as L, seq2seq as S2S
from arabic_summ.train.select import select_best
assert torch.cuda.is_available(), "No GPU: Runtime > Change runtime type > T4 GPU"
print(torch.cuda.get_device_name(0), "| ledger so far:", L.summary(cfg))
FT, TR = cfg["finetune"], cfg["finetune"]["train"]
SEED = TR["seed"]
POLICY = NormalizationPolicy.from_dict(cfg["normalization"]["model"])
'''), code('''
# ==== 4. Fact checker on the 02 smoke-run outputs ======================================
demo = C.path(cfg, "logs") / "02_pipeline_demo" / "results.jsonl"
items = B.read_csv(C.path(cfg, "benchmark") / B.ITEMS_FILE).set_index("sample_id")
if demo.exists():
    d = pd.read_json(demo, lines=True)
    reps = [F.check(s, items.text[i]) for s, i in zip(d.summary, d.sample_id)]
    d["number_precision"] = [r.number_precision for r in reps]
    d["numbers_ok"] = [r.numbers_ok for r in reps]
    d["novel_rate"] = [r.novel_rate for r in reps]
    display(d.groupby("strategy")[["numbers_ok", "novel_rate"]].mean().round(3))
    ex = d[(d.sample_id == d.sample_id.iloc[0]) & (d.strategy == "segmentwise")].iloc[0]
    r = F.check(ex.summary, items.text[ex.sample_id])
    print(ex.sample_id, "segmentwise | unsupported numbers:", r.unsupported_numbers, "| novel:", r.novel_content[:20])
else:
    print("02 smoke outputs not found; skipping")
'''), code('''
# ==== 5. Faithfulness filter for XL-Sum train (cached) =================================
filt_path = C.path(cfg, "processed", "train_filtered.parquet")
rep_path = C.path(cfg, "processed", "train_filter_report.json")
if not step_done(cfg, "03_train_filter"):
    train = prepare.load_processed(cfg, "train")
    table = D.faithfulness_table(train)
    kept, rep = D.filter_pairs(train, FT["filter"], table)
    kept.to_parquet(filt_path)
    table.to_parquet(C.path(cfg, "processed", "train_faithfulness.parquet"))
    rep["filter_config"] = FT["filter"]
    write_json(rep_path, rep)
    prepare.update_manifest(cfg, "train_filter", {**rep, "output": str(filt_path), "sha256": sha256_file(filt_path)})
    mark_step_done(cfg, "03_train_filter", outputs=[filt_path, rep_path])
FILTER_REPORT = read_json(rep_path)
print(json.dumps(FILTER_REPORT, ensure_ascii=False, indent=1))
train_f = pd.read_parquet(filt_path)
'''), code('''
# ==== 6. Validation subsets + tokenization =============================================
val = prepare.load_processed(cfg, "validation")
ev = val.sample(TR["eval_samples"], random_state=SEED)                                   # per-epoch selection
tune_df = val.drop(ev.index).sample(FT["decoding_grid"]["samples"], random_state=SEED)  # decoding tuning
AB = M.model_entry(cfg, "arabart")
tok = M.load_tokenizer(AB["repo_id"], revision=AB["revision"])
tokenize = lambda df, t=tok: D.tokenize_pairs(df, t, POLICY, FT["max_source_tokens"], FT["max_target_tokens"])
eval_rows = tokenize(ev)
grid_df = D.sample_fraction(train_f, TR["grid_train_fraction"], SEED)
print(f"filtered train {len(train_f)} | grid subset {len(grid_df)} | eval {len(ev)} | decoding-tune {len(tune_df)}")
print("truncated sources in eval:", sum(len(r["input_ids"]) >= FT["max_source_tokens"] for r in eval_rows))
'''), code('''
# ==== 7. AraBART learning-rate grid (resumable; rerun this cell after a disconnect) =====
grid_rows = tokenize(grid_df)
grid = []
for lr in TR["learning_rates"]:
    name = S2S.run_name("arabart", lr, TR["grid_train_fraction"])
    hp = {**TR, "learning_rate": lr, "max_target_tokens": FT["max_target_tokens"]}
    r = S2S.train_run(cfg, name, AB, tok, grid_rows, eval_rows, list(ev.summary), list(ev.text), hp,
                      extra_meta={"stage": "lr_grid"})
    grid.append({"lr": lr, "run": name, "epochs": r["epochs_run"], **r["best"]})
grid_df_res = pd.DataFrame(grid)
display(grid_df_res[["lr", "epochs", "rouge1", "rouge2", "rougeL", "number_precision", "numbers_ok_share",
                     "novel_rate", "gen_words"]].round(4))
BEST_LR = select_best(grid, **FT["select"])["lr"]
write_json(C.path(cfg, "models", "runs", "lr_grid_summary.json"), {"rows": grid, "selected_lr": BEST_LR})
print("selected learning rate:", BEST_LR)
'''), code('''
# ==== 8. Final AraBART run: selected LR, 100 % of filtered train (resumable) ============
del grid_rows
frac = TR["final_train_fraction"]
final_rows = tokenize(D.sample_fraction(train_f, frac, SEED))
FINAL_NAME = S2S.run_name("arabart", BEST_LR, frac)
FINAL = S2S.train_run(cfg, FINAL_NAME, AB, tok, final_rows, eval_rows, list(ev.summary), list(ev.text),
                      {**TR, "learning_rate": BEST_LR, "max_target_tokens": FT["max_target_tokens"]},
                      extra_meta={"stage": "final"})
del final_rows
display(pd.DataFrame(FINAL["per_epoch"])[["epoch", "rouge1", "rouge2", "rougeL", "number_precision", "novel_rate", "loss"]].round(4))
print("best:", {k: FINAL["best"][k] for k in ("epoch", "rouge1", "rouge2", "rougeL", "number_precision")})
'''), code('''
# ==== 9. AraT5v2 single comparison run (fp32; set RUN_ARAT5=False to skip) ==============
RUN_ARAT5 = True
if RUN_ARAT5:
    from transformers import AutoConfig
    T5 = M.model_entry(cfg, "arat5v2")
    tok5 = M.load_tokenizer(T5["repo_id"], revision=T5["revision"])
    c5 = AutoConfig.from_pretrained(T5["repo_id"], revision=T5["revision"])
    if tok5.pad_token is None:
        tok5.pad_token = tok5.convert_ids_to_tokens(c5.pad_token_id)
    if tok5.eos_token is None:
        tok5.eos_token = tok5.convert_ids_to_tokens(c5.eos_token_id)
    print("AraT5v2 tokenizer:", type(tok5).__name__, "| pad/eos:", tok5.pad_token, tok5.eos_token,
          "| sample ids end:", tok5("مرحبا بالعالم")["input_ids"][-3:])
    hp5 = {**TR, **FT["arat5v2"], "max_target_tokens": FT["max_target_tokens"]}
    t5_rows = tokenize(grid_df, tok5)
    t5_eval = tokenize(ev, tok5)
    R5 = S2S.train_run(cfg, S2S.run_name("arat5v2", hp5["learning_rate"], TR["grid_train_fraction"]), T5, tok5,
                       t5_rows, t5_eval, list(ev.summary), list(ev.text), hp5, extra_meta={"stage": "arat5_comparison"})
    same_lr = next((g for g in grid if abs(g["lr"] - hp5["learning_rate"]) < 1e-12), None)
    cmp = [{"model": "arat5v2", "lr": hp5["learning_rate"], **R5["best"]}]
    if same_lr:
        cmp.append({"model": "arabart", **same_lr})
    display(pd.DataFrame(cmp)[["model", "lr", "rouge1", "rouge2", "rougeL", "number_precision", "novel_rate", "gen_words"]].round(4))
    write_json(C.path(cfg, "models", "runs", "arat5_comparison.json"), cmp)
'''), code('''
# ==== 10. Decoding tuning on validation (beams 1-4 x length penalty 1/1.5/2) ===========
from arabic_summ.backends.hf import HFBackend
dec_path = C.path(cfg, "models", "runs", FINAL_NAME, "decoding_grid.json")
if dec_path.exists():
    DEC_ROWS, BEST_DEC = read_json(dec_path)["rows"], read_json(dec_path)["selected"]
else:
    backend = HFBackend(FINAL["best_dir"], dtype="float16", batch_size=16, max_input_tokens=FT["max_source_tokens"])
    with L.GPUSession(cfg, f"{FINAL_NAME}_decoding", "03_finetune", stage="decoding"):
        DEC_ROWS, BEST_DEC = DEC.tune(backend, [normalize(t, POLICY) for t in tune_df.text], list(tune_df.summary),
                                      FT["decoding_grid"], FT["max_target_tokens"], FT["select"])
    write_json(dec_path, {"rows": DEC_ROWS, "selected": BEST_DEC, "n_texts": len(tune_df)})
    del backend; torch.cuda.empty_cache()
display(pd.DataFrame(DEC_ROWS)[["num_beams", "length_penalty", "rouge1", "rouge2", "rougeL", "number_precision",
                                 "novel_rate", "gen_words", "seconds_per_text"]].round(4))
print("selected decoding:", BEST_DEC)
'''), code('''
# ==== 11. Export model + provenance ====================================================
from arabic_summ.train.provenance import export_model
man = read_json(prepare.manifest_path(cfg))
best_dec_row = next(r for r in DEC_ROWS if all(r[k] == v for k, v in BEST_DEC.items()))
PROV = export_model(cfg, FINAL["best_dir"], FT["export_name"], BEST_DEC, {
    "base_model": AB, "training_run": FINAL, "lr_grid": {"rows": grid, "selected_lr": BEST_LR},
    "train_filter": FILTER_REPORT, "decoding_validation": best_dec_row,
    "data": {"dataset": man.get("dataset"), "processed_files": man.get("processed_files"),
             "train_filter": man.get("train_filter")},
    "eval_protocol": {"validation_eval_samples": len(ev), "decoding_tune_samples": len(tune_df), "seed": SEED,
                      "rouge": "arabic_summ.eval.rouge (matching-normalized whitespace tokens, no stemming)"},
    "gpu_ledger": L.summary(cfg),
})
prepare.update_manifest(cfg, "finetune", {k: PROV[k] for k in ("model_name", "exported_utc", "size_mb", "config_hash")}
                        | {"selected_lr": BEST_LR, "decoding": BEST_DEC, "validation_best": FINAL["best"]})
print(PROV["model_name"], PROV["size_mb"], "MB |", {k: v["sha256"][:12] for k, v in PROV["files"].items()})
'''), code('''
# ==== 12. Sanity check on 6 benchmark texts: fine-tuned vs public baseline =============
from arabic_summ.summarize.pipeline import Summarizer
sel = items[items.status == "selected"].reset_index()
demo6 = sel.groupby("domain", group_keys=False).apply(lambda g: g.sample(3, random_state=SEED))
out = []
for key in ("arabart_ft", "arabart_xlsum_omar"):
    be = M.load_hf_backend(cfg, key, dtype="float16")
    sm = Summarizer(cfg, backend=be, decoding=M.decoding_for(cfg, key), tokens_per_word=M.tokens_per_word(cfg, "arabart"))
    for _, r in demo6.iterrows():
        for st in ("hybrid", "segmentwise", "direct"):
            res = sm.summarize(r.text, strategy=st)
            fr = F.check(res.summary, r.text)
            out.append({"model": key, "strategy": st, "sample_id": r.sample_id, "length_ok": res.length_ok,
                        "numbers_ok": fr.numbers_ok, "novel_rate": fr.novel_rate, "seconds": res.seconds["total"],
                        "summary": res.summary})
    del be; torch.cuda.empty_cache()
san = pd.DataFrame(out)
san.to_json(C.path(cfg, "logs") / "03_sanity.jsonl", orient="records", lines=True, force_ascii=False)
display(san.groupby(["model", "strategy"])[["length_ok", "numbers_ok", "novel_rate", "seconds"]].mean().round(3))
x = san[(san.model == "arabart_ft") & (san.strategy == "hybrid")].iloc[0]
print(x.sample_id, "\\n", x.summary)
'''), code('''
# ==== 13. Summary (paste this output back) =============================================
print("filter :", {k: FILTER_REPORT[k] for k in ("rows_in", "rows_kept", "kept_share", "max_novel_rate_used", "relaxed_to_min_keep")})
print("grid   :", [(g["lr"], g["epochs"], round(g["rouge1"], 4), round(g["number_precision"], 4)) for g in grid], "-> lr", BEST_LR)
print("final  :", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in FINAL["best"].items() if k in ("epoch", "rouge1", "rouge2", "rougeL", "number_precision", "novel_rate", "gen_words")})
print("decode :", BEST_DEC, "| rouge1", round(best_dec_row["rouge1"], 4))
print("export :", PROV["model_name"], PROV["size_mb"], "MB, commit", PROV["code"]["git_commit"])
print("ledger :", L.summary(cfg))
''')]
    return notebook(cells, gpu=True)


def nb04() -> dict:
    cells = [md('''
# 04 · CPU optimization: PyTorch vs ONNX Runtime, FP32 vs INT8

**What this does** (proposal stage 5 and the stage-1/2 model-selection rule)
1. Takes the fine-tuned model from 03 (`models/arabart_ft`). If it doesn't exist yet, runs a **dry run** on the
   public baseline so the whole notebook can be tested before 03 finishes.
2. Creates a separate **ONNX environment** (`requirements/colab-onnx.txt`; optimum-onnx needs transformers 4.x)
   and exports ONNX **FP32** and dynamic **INT8** (AVX2 kernels) to Drive.
3. Measures each variant **in a fresh process** (batch size 1, model preloaded): load + warm-up time, peak process
   memory, weight size, ROUGE and fact check on 200 XL-Sum validation articles (direct generation, tuned decoding),
   and full-pipeline latency on 24 benchmark texts (4 per domain × length bin) with length compliance + fact check.
4. Applies the proposal's rule. **Gates:** weights ≤ 1 GB, process memory ≤ 8 GB, median latency ≤ 15 s,
   ROUGE-1 drop ≤ 0.02 vs PyTorch FP32. **Score:** 50 % quality & factual accuracy + 30 % CPU time + 10 % memory + 10 % size.
5. Exports the selected variant to `models/arabart_ft_cpu/` with `runtime.json` for the MVP and `bench/`.

**Runtime: Colab CPU** (no GPU needed). Colab gives 2 vCPUs, so latencies here are only for comparing variants;
the acceptance timing runs later with `bench/` on the 8-core reference machine. Resumable: finished
exports and measurements are kept on Drive.
'''), *setup_cells("04_optimize", "requirements/colab-data.txt"), code('''
# ==== 3. Model to optimize (fine-tuned, or dry run on the public baseline) ==============
import json, os, shutil, subprocess, sys
from pathlib import Path
import pandas as pd
from arabic_summ import config as C, models as M
from arabic_summ.data import prepare, benchmark as B
from arabic_summ.io_utils import read_json, write_json
from arabic_summ.optimize import select as SEL
from arabic_summ.optimize.variants import VARIANTS, model_dir_for
OPT, SEED = cfg["optimize"], cfg["project"]["seed"]
FT_DIR = M.local_path(cfg, "arabart_ft")
DRY_RUN = not (FT_DIR / "model.safetensors").exists()
LOCAL = Path("/content/opt"); LOCAL.mkdir(parents=True, exist_ok=True)
if DRY_RUN:
    from huggingface_hub import snapshot_download
    b = M.model_entry(cfg, "arabart_xlsum_omar")
    SRC_DIR = Path(snapshot_download(b["repo_id"], revision=b["revision"], local_dir=str(LOCAL / "baseline_src")))
    TAG, DECODING = "dryrun_baseline", dict(cfg["decoding"])
    print("DRY RUN: fine-tuned model not found, using", b["repo_id"])
else:
    SRC_DIR, TAG, DECODING = FT_DIR, OPT["export_name"], M.decoding_for(cfg, "arabart_ft")
TORCH_DIR = LOCAL / "torch"
if not (TORCH_DIR / "config.json").exists():
    shutil.copytree(SRC_DIR, TORCH_DIR, dirs_exist_ok=True)
ONNX_ROOT = C.path(cfg, "models", f"{TAG}_onnx", "x").parent
OUT_DIR = C.path(cfg, "logs", f"04_optimize_{TAG}", "x").parent
print("source:", SRC_DIR, "| decoding:", DECODING, "| ONNX ->", ONNX_ROOT)
'''), code('''
# ==== 4. Separate ONNX environment (transformers 4.x + optimum-onnx) ====================
ORT_PY = "/content/ort_env/bin/python"
ENV = {**os.environ, "PYTHONPATH": f"{REPO_DIR}/src"}
if not os.path.exists(ORT_PY):
    sh("python -m venv --system-site-packages /content/ort_env")
    sh(f"{ORT_PY} -m pip install -q -r {REPO_DIR}/requirements/colab-onnx.txt")
chk = subprocess.run([ORT_PY, "-c", "import transformers, onnxruntime, optimum.onnxruntime as o; "
                      "print('transformers', transformers.__version__, '| onnxruntime', onnxruntime.__version__)"],
                     capture_output=True, text=True)
print(chk.stdout, chk.stderr[-1500:])
'''), code('''
# ==== 5. Export ONNX FP32 + dynamic INT8 (skipped if already on Drive) ==================
if not (ONNX_ROOT / "int8" / "encoder_model.onnx").exists():
    r = subprocess.run([ORT_PY, "-m", "arabic_summ", "onnx-export", str(TORCH_DIR), str(ONNX_ROOT),
                        "--arch", OPT["onnx_quant_arch"]], capture_output=True, text=True, env=ENV, cwd=REPO_DIR)
    print(r.stdout[-2000:]); assert r.returncode == 0, r.stderr[-4000:]
for sub in ("fp32", "int8"):
    print(sub, {p.name: round(p.stat().st_size / 2**20, 1) for p in sorted((ONNX_ROOT / sub).glob("*.onnx*"))})
'''), code('''
# ==== 6. Measurement inputs (validation articles + stratified benchmark texts) ==========
val = prepare.load_processed(cfg, "validation").sample(OPT["val_samples"], random_state=SEED + 1)
items = B.read_csv(C.path(cfg, "benchmark") / B.ITEMS_FILE)
items = items[items.status == "selected"]
bench = (items.groupby(["domain", "length_bin"], group_keys=False)
              .apply(lambda g: g.sample(min(len(g), OPT["bench_per_cell"]), random_state=SEED)))
base_spec = {"config_paths": [f"{REPO_DIR}/{c}" for c in CONFIGS], "root": cfg["paths"]["root"],
             "threads": os.cpu_count(), "decoding": DECODING, "max_new_tokens": cfg["finetune"]["max_target_tokens"],
             "val": val[["text", "summary"]].to_dict("records"),
             "bench": bench[["sample_id", "domain", "text"]].to_dict("records"),
             "ratio": cfg["pipeline"]["default_ratio"], "strategy": cfg["pipeline"]["strategy"],
             "tokens_per_word": M.tokens_per_word(cfg, "arabart")}
print(f"validation {len(val)} | benchmark {len(bench)} ({bench.groupby(['domain','length_bin']).size().to_dict()})"
      f" | threads {os.cpu_count()} | strategy {base_spec['strategy']}")
'''), code('''
# ==== 7. Measure every variant in a fresh process (resumable; ~10-40 min each on Colab CPU) =
results = {}
for v in OPT["variants"]:
    out = OUT_DIR / f"{v}.json"
    if not out.exists():
        spec = {**base_spec, "variant": v, "model_dir": str(model_dir_for(v, TORCH_DIR, ONNX_ROOT))}
        sp = OUT_DIR / f"spec_{v}.json"; write_json(sp, spec)
        py = ORT_PY if VARIANTS[v]["runtime"] == "onnx" else sys.executable
        r = subprocess.run([py, "-m", "arabic_summ", "optimize-measure", str(sp), str(out)],
                           capture_output=True, text=True, env=ENV, cwd=REPO_DIR)
        print(v, r.stdout[-600:])
        if r.returncode:
            print(r.stderr[-3000:]); continue
    results[v] = read_json(out)
cols = ["variant", "weights_mb", "peak_rss_mb", "load_seconds", "warmup_seconds", "latency_median_s", "latency_p90_s",
        "rouge1", "rouge2", "rougeL", "val_number_precision", "val_novel_rate", "bench_length_ok", "bench_numbers_ok"]
display(pd.DataFrame(list(results.values()))[cols])
'''), code('''
# ==== 8. Proposal selection rule + export of the selected variant =======================
ranked = SEL.composite(list(results.values()), OPT["gates"], OPT["weights"])
display(pd.DataFrame([{k: r.get(k) for k in ("variant", "score", "score_parts", "gate_failures")} for r in ranked]))
best = ranked[0]
assert best["score"] is not None, "no variant passed the gates: see gate_failures"
write_json(OUT_DIR / "selection.json", {"ranked": [{k: v for k, v in r.items() if k != "per_sample"} for r in ranked],
                                        "selected": best["variant"], "gates": OPT["gates"], "weights": OPT["weights"],
                                        "dry_run": DRY_RUN})
if not DRY_RUN:
    dest = C.path(cfg, "models", OPT["export_name"], "x").parent
    if dest.exists(): shutil.rmtree(dest)
    shutil.copytree(best["model_dir"], dest)
    write_json(dest / "decoding.json", DECODING)
    write_json(dest / "runtime.json", {"variant": best["variant"], **VARIANTS[best["variant"]],
                                       "requirements": "requirements/colab-onnx.txt" if VARIANTS[best["variant"]]["runtime"] == "onnx" else "torch",
                                       "measured": {k: best[k] for k in cols if k in best}, "source_model": str(SRC_DIR)})
    prepare.update_manifest(cfg, "optimize", {"selected": best["variant"], "export": str(dest),
                            "metrics": {k: best[k] for k in cols if k in best}, "score": best["score"]})
    print("exported", best["variant"], "->", dest)
else:
    print("dry run: selection recorded, nothing exported")
'''), code('''
# ==== 9. Summary (paste this output back) ==============================================
ref = results.get("torch_fp32", {})
for r in ranked:
    drop = round(ref.get("rouge1", 0) - r["rouge1"], 4) if ref else None
    print(f"{r['variant']:11s} score={r['score']} size={r['weights_mb']}MB rss={r['peak_rss_mb']}MB "
          f"median={r['latency_median_s']}s p90={r['latency_p90_s']}s R1={r['rouge1']:.4f} (drop {drop}) "
          f"numP={r['val_number_precision']:.3f} gates={r['gate_failures'] or 'ok'}")
print("machine:", next(iter(results.values()))["machine"], "| dry run:", DRY_RUN)
''')]
    return notebook(cells)


def nb05() -> dict:
    cells = [md('''
# 05 · Evaluation on the internal benchmark (240 texts) + blind human evaluation

**What this does** (proposal §7 acceptance criteria and stage 6)
1. Loads the benchmark and references. `evaluate.approved_only: false` uses every drafted reference and labels the
   numbers **interim**; set it to `true` for the final report (only human-approved, model-drafted references).
2. Runs each system over the 240 texts (resumable JSONL per system): `lead`, `extractive`, the public baseline, the
   fine-tuned model with the **primary strategy fixed in config before scoring** (`segmentwise`) plus ablations
   (`direct`, `hybrid`), and the optimized CPU variant from 04 on CPU (latency).
3. Scores ROUGE-1/2/L F1 (bootstrap 95 % CI), length compliance, compression ratio and the automatic fact check, by
   domain and length band; paired CI of fine-tuned vs baseline; latency by band (300–700 / 300–1000 words);
   optional BERTScore.
4. Writes **blind human-evaluation sheets** (100 outputs, 50 per domain; two raters; 20 calibration items for the
   reference rater) and, once filled, computes weighted Cohen's kappa, the third-rater list and final scores.
5. Prints the **acceptance table** and writes `logs/05_evaluate/report.md` + `report.json`.

**Runtime: T4 GPU** recommended (the model systems run on GPU for quality; the optimized CPU system always runs on
CPU for latency). Colab's 2 vCPUs are not the reference machine: the acceptance latency comes from `bench/`.
'''), *setup_cells("05_evaluate", "requirements/colab-data.txt"), code('''
# ==== 3. Benchmark, references and systems ============================================
import json, os, subprocess, sys
from pathlib import Path
import pandas as pd, torch
from arabic_summ import config as C, models as M
from arabic_summ.data import prepare, benchmark as B
from arabic_summ.eval import benchmark_eval as BE, facts as F, human as H
from arabic_summ.eval.stats import paired_diff_ci
from arabic_summ.io_utils import read_json, write_json
from arabic_summ.optimize.variants import VARIANTS
EV, SEED = cfg["evaluate"], cfg["project"]["seed"]
OUT = C.path(cfg, "logs", "05_evaluate", "outputs", "x").parent
REP = OUT.parent
items = B.read_csv(C.path(cfg, "benchmark") / B.ITEMS_FILE)
items = items[items.status == "selected"]
refs = B.read_csv(C.path(cfg, "benchmark") / B.REFS_FILE)
SOURCES = dict(zip(items.sample_id, items.text))
R = BE.reference_table(refs[refs.sample_id.isin(SOURCES)], EV["approved_only"])
INTERIM = not EV["approved_only"]
print(f"{len(items)} texts | references used: {len(R)} | by review_status: {R.review_status.value_counts().to_dict()}"
      f" | by source: {R.reference_source.value_counts().to_dict()} | {'INTERIM' if INTERIM else 'FINAL'}")

ENV = {**os.environ, "PYTHONPATH": f"{REPO_DIR}/src"}
GPU = "cuda" if torch.cuda.is_available() else "cpu"
base = {"config_paths": [f"{REPO_DIR}/{c}" for c in CONFIGS], "root": cfg["paths"]["root"],
        "ratio": cfg["pipeline"]["default_ratio"], "tokens_per_word": M.tokens_per_word(cfg, "arabart"),
        "threads": os.cpu_count(), "items": items[["sample_id", "domain", "text"]].to_dict("records")}
SYSTEMS = {"lead": {"variant": None, "strategy": "lead"}, "extractive": {"variant": None, "strategy": "extractive"}}
from huggingface_hub import snapshot_download
b = M.model_entry(cfg, "arabart_xlsum_omar")
BASE_DIR = snapshot_download(b["repo_id"], revision=b["revision"], local_dir="/content/eval_models/baseline")
SYSTEMS[f"baseline_{EV['primary_strategy']}"] = {"variant": "torch_fp32", "model_dir": BASE_DIR, "device": GPU,
                                                 "strategy": EV["primary_strategy"], "decoding": dict(cfg["decoding"])}
FT_DIR = M.local_path(cfg, "arabart_ft")
if (FT_DIR / "config.json").exists():
    for st in [EV["primary_strategy"], *EV["ablation_strategies"]]:
        SYSTEMS[f"ft_{st}"] = {"variant": "torch_fp32", "model_dir": str(FT_DIR), "device": GPU, "strategy": st,
                               "decoding": M.decoding_for(cfg, "arabart_ft")}
CPU_DIR = C.root(cfg) / "models" / cfg["optimize"]["export_name"]
CPU_RT = read_json(CPU_DIR / "runtime.json") if (CPU_DIR / "runtime.json").exists() else None
if CPU_RT:
    SYSTEMS[f"cpu_{CPU_RT['variant']}_{EV['primary_strategy']}"] = {
        "variant": CPU_RT["variant"], "model_dir": str(CPU_DIR), "device": "cpu", "strategy": EV["primary_strategy"],
        "decoding": read_json(CPU_DIR / "decoding.json")}
PRIMARY = next((k for k in SYSTEMS if k.startswith("cpu_")), next((k for k in SYSTEMS if k.startswith("ft_")), None))
print("systems:", list(SYSTEMS), "| primary:", PRIMARY, "| fine-tuned found:", (FT_DIR / "config.json").exists())
'''), code('''
# ==== 4. ONNX environment (only if the optimized variant is ONNX) ======================
ORT_PY = "/content/ort_env/bin/python"
if CPU_RT and VARIANTS[CPU_RT["variant"]]["runtime"] == "onnx" and not os.path.exists(ORT_PY):
    sh("python -m venv --system-site-packages /content/ort_env")
    sh(f"{ORT_PY} -m pip install -q -r {REPO_DIR}/requirements/colab-onnx.txt")
'''), code('''
# ==== 5. Run every system over the benchmark (resumable; rerun after a disconnect) =====
for name, sysd in SYSTEMS.items():
    out = OUT / f"{name}.jsonl"
    n_done = sum(1 for _ in open(out)) if out.exists() else 0
    if n_done >= len(items):
        print(f"[skip] {name}: {n_done}"); continue
    spec = {**base, **sysd, "system": name}
    sp = OUT / f"spec_{name}.json"; write_json(sp, spec)
    onnx = sysd.get("variant") and VARIANTS[sysd["variant"]]["runtime"] == "onnx"
    r = subprocess.run([ORT_PY if onnx else sys.executable, "-m", "arabic_summ", "summarize-batch", str(sp), str(out)],
                       capture_output=True, text=True, env=ENV, cwd=REPO_DIR)
    print(name, "rc", r.returncode, r.stderr[-1500:] if r.returncode else "")
OUTPUTS = pd.concat([pd.read_json(OUT / f"{n}.jsonl", lines=True) for n in SYSTEMS if (OUT / f"{n}.jsonl").exists()])
print(OUTPUTS.groupby("system").size().to_dict())
'''), code('''
# ==== 6. Automatic metrics: ROUGE (95 % CI), length, fact check, latency ===============
SCORED = BE.score_outputs(OUTPUTS, R, SOURCES)
SUMM = BE.summary_table(SCORED, seed=SEED)
LAT = BE.latency_table(OUTPUTS)
SCORED.to_csv(REP / "scored.csv", index=False, encoding="utf-8-sig")
SUMM.to_csv(REP / "summary.csv", index=False, encoding="utf-8-sig")
display(SUMM[SUMM.subset == "all"].sort_values("rouge1", ascending=False))
display(SUMM[SUMM.system == PRIMARY]) if PRIMARY else None
display(LAT)
bl = f"baseline_{EV['primary_strategy']}"
for s in [k for k in SYSTEMS if k.startswith(("ft_", "cpu_"))]:
    a = SCORED[SCORED.system == s].set_index("sample_id"); c = SCORED[SCORED.system == bl].set_index("sample_id")
    ids = a.index.intersection(c.index)
    d, lo, hi = paired_diff_ci(a.loc[ids, "rouge1"], c.loc[ids, "rouge1"], seed=SEED)
    print(f"{s} - {bl}: ROUGE-1 {d:+.4f} (95% CI {lo:+.4f} .. {hi:+.4f}), n={len(ids)}")
'''), code('''
# ==== 7. Optional BERTScore (multilingual BERT; skipped if unavailable) ================
BERT = None
try:
    from bert_score import score as bscore
except ImportError:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bert-score==0.3.13"], capture_output=True, text=True)
    from bert_score import score as bscore
try:
    rows = []
    for s, g in SCORED.groupby("system"):
        hyp = OUTPUTS.set_index(["system", "sample_id"]).loc[[(s, i) for i in g.sample_id], "summary"].tolist()
        _, _, f1 = bscore(hyp, [R.at[i, "reference_summary"] for i in g.sample_id], model_type=EV["bertscore_model"],
                          batch_size=16, device=GPU)
        rows.append({"system": s, "bertscore_f1": round(float(f1.mean()), 4)})
    BERT = pd.DataFrame(rows); display(BERT)
except Exception as e:
    print("BERTScore skipped:", type(e).__name__, str(e)[:200])
'''), code('''
# ==== 8. Blind human-evaluation sheets (never overwritten) =============================
HE = EV["human_eval"]
HD = C.path(cfg, "logs", "05_evaluate", "human_eval", "x").parent
HUMAN_SYSTEM = PRIMARY
if HUMAN_SYSTEM and not (HD / "key.csv").exists():
    sub = OUTPUTS[OUTPUTS.system == HUMAN_SYSTEM]
    flags = {(r.system, r.sample_id): ("unsupported numbers: " + r.unsupported_numbers) if r.unsupported_numbers else ""
             for r in SCORED[SCORED.system == HUMAN_SYSTEM].itertuples()}
    sheet, key = H.build_sheets(sub, SOURCES, HE["n_per_domain"], HE["n_calibration"], SEED, salt=cfg["project"]["name"],
                                auto_flags=flags)
    for rater in ("rater_A", "rater_B"):
        B.write_csv(sheet, HD / f"{rater}.csv")
    B.write_csv(sheet[sheet.item_id.isin(key.item_id[key.calibration])], HD / "rater_REF_calibration.csv")
    B.write_csv(key, HD / "key.csv")
    print(f"wrote {len(sheet)} items for {HUMAN_SYSTEM} -> {HD}")
print("Raters: fill coverage/accuracy/fluency/coherence (1-5), critical_error (yes/no) and critical_error_type "
      f"({'/'.join(H.ERROR_TYPES)}). Do not open key.csv. Files:", sorted(p.name for p in HD.glob("*.csv")))
'''), code('''
# ==== 9. Human results (runs once the sheets are filled) ===============================
HUMAN = None
fa, fb, fref = HD / "rater_A.csv", HD / "rater_B.csv", HD / "rater_REF_calibration.csv"
if fa.exists() and fb.exists():
    A_, B_ = B.read_csv(fa), B.read_csv(fb)
    errs = H.validate_sheet(A_) + H.validate_sheet(B_)
    if errs:
        print("sheets not complete yet:", errs[:6])
    else:
        key = B.read_csv(HD / "key.csv"); cal = key.item_id[key.calibration.astype(str) == "True"].tolist()
        if fref.exists() and not H.validate_sheet(ref := B.read_csv(fref)):
            for nm, X in (("rater_A", A_), ("rater_B", B_)):
                k = H.agreement(X, ref, HE["kappa_weights"], ids=cal)
                ok = all(k[c] >= HE["min_calibration_kappa"] for c in H.CRITERIA)
                print(f"calibration {nm} vs REF: {k} -> {'qualified' if ok else 'NOT qualified (kappa < 0.7)'}")
        print("agreement A vs B:", H.agreement(A_, B_, HE["kappa_weights"]))
        adj = H.adjudication_items(A_, B_, HE["max_gap"])
        fc = HD / "rater_C_adjudication.csv"
        if len(adj) and not fc.exists():
            B.write_csv(adj.assign(rater_c=""), fc); print(f"{len(adj)} disagreements -> {fc.name} (third rater)")
        third = B.read_csv(fc) if fc.exists() else None
        if third is not None and (third.rater_c.astype(str).str.strip() == "").any():
            print("waiting for the third rater:", int((third.rater_c.astype(str).str.strip() == "").sum()), "rows")
        else:
            HUMAN = H.summarize(H.final_scores(A_, B_, key, third, HE["max_gap"]), HE["min_mean"], HE["max_critical_share"])
            display(HUMAN); HUMAN.to_csv(REP / "human_summary.csv", index=False)
'''), code('''
# ==== 10. Acceptance table + report (paste this output back) ===========================
cpu_meas = None
if CPU_RT:
    cpu_meas = CPU_RT.get("measured")
ACC = pd.DataFrame(BE.acceptance(PRIMARY, SUMM, LAT, EV["acceptance"], HUMAN, cpu_meas)) if PRIMARY else pd.DataFrame()
display(ACC)
label = "INTERIM (references not all human-approved)" if INTERIM else "FINAL (human-approved references only)"
md_lines = [f"# Benchmark evaluation — {label}", "",
            f"References: {len(R)} ({R.reference_source.value_counts().to_dict()}; {R.review_status.value_counts().to_dict()}). "
            "Model-drafted references are reported as *model-drafted, human-verified*.", "",
            f"Primary system: `{PRIMARY}` (strategy fixed before scoring).", "", "## Acceptance", BE.table_md(ACC), "",
            "## ROUGE / length / fact check (all texts)", BE.table_md(SUMM[SUMM.subset == "all"]), "",
            "## By domain and length band (primary)", BE.table_md(SUMM[SUMM.system == PRIMARY]), "",
            "## Latency (Colab; acceptance latency comes from bench/)", BE.table_md(LAT)]
if BERT is not None: md_lines += ["", "## BERTScore", BE.table_md(BERT)]
if HUMAN is not None: md_lines += ["", "## Human evaluation", BE.table_md(HUMAN)]
(REP / "report.md").write_text("\\n".join(md_lines) + "\\n", encoding="utf-8")
write_json(REP / "report.json", {"interim": INTERIM, "primary": PRIMARY, "systems": list(SYSTEMS),
                                 "acceptance": ACC.to_dict("records"), "summary": SUMM.to_dict("records"),
                                 "latency": LAT.to_dict("records"),
                                 "bertscore": None if BERT is None else BERT.to_dict("records"),
                                 "human": None if HUMAN is None else HUMAN.to_dict("records")})
prepare.update_manifest(cfg, "evaluate", {"interim": INTERIM, "primary": PRIMARY,
                        "acceptance": ACC.to_dict("records"), "n_references": int(len(R))})
print(label); print(ACC.to_string(index=False))
''')]
    return notebook(cells, gpu=True)


NOTEBOOKS = {"01_data.ipynb": nb01, "02_preprocess.ipynb": nb02, "03_finetune.ipynb": nb03, "04_optimize.ipynb": nb04, "05_evaluate.ipynb": nb05}


def main() -> None:
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for name, build in NOTEBOOKS.items():
        (out / name).write_text(json.dumps(build(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print("wrote", out / name)


if __name__ == "__main__":
    main()
