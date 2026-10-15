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


NOTEBOOKS = {"01_data.ipynb": nb01, "02_preprocess.ipynb": nb02, "03_finetune.ipynb": nb03}


def main() -> None:
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for name, build in NOTEBOOKS.items():
        (out / name).write_text(json.dumps(build(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print("wrote", out / name)


if __name__ == "__main__":
    main()
