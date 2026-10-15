# Handoff: where the project stands and how to continue (2026-10-16)

Read this first, then `README.md` (how to run) and `PROGRESS.md` (every decision and result so far).
The requirements come from the funding proposal (`docs/proposal.pdf`, Persian). It's kept out of git
because it contains personal contact details, so ask the project owner for a copy.

## 1. Status in one paragraph

All Phase 1 code is written, unit-tested on CPU (106 tests) and pushed to `main`: data
preparation, the 240-text benchmark, preprocessing and chunking, fine-tuning, CPU optimization,
evaluation, the offline MVP (CLI + web UI) and the acceptance benchmark tool. **There is no trained
model yet.** The first Colab training run collapsed because of a tokenizer bug (fixed in commit
`f4ae969`, see PROGRESS decision 21). The fixed run has started but hasn't finished: free Colab GPU
sessions keep disconnecting. Interim benchmark scores, against model-drafted references that aren't
human-reviewed yet: lead baseline ROUGE-1 0.370, extractive 0.352. The proposal target is > 0.40.

## 2. What is left (in order)

| # | Task | Who | Effort | Notes |
|---|---|---|---|---|
| 1 | Run `03_finetune` to the end on a T4 | anyone with the Drive | 4–8 GPU hours over several sessions | Resumable. First check: cell 6 must print `special tokens ok`. In cell 7, ROUGE-1 must **rise** each epoch and "Gen Words" must stay around 20–40 |
| 2 | Run `04_optimize` (Colab CPU runtime) | anyone | 1–3 h | Exports `models/arabart_ft_cpu/`. It can be dry-run before step 1 finishes |
| 3 | Run `05_evaluate` (T4) | anyone | 2–4 h | Writes the benchmark report and blind human-evaluation sheets |
| 4 | Review the 240 reference summaries | **fluent Arabic speaker** | ~15–25 h | `references.csv` on Drive: check each draft against `docs/reference_guideline.md` §6, set `reviewer_id` and `review_status` |
| 5 | Human evaluation | **two Arabic raters + a third for disagreements** | ~10 h each | Sheets from `05` cell 8; 20 calibration items first (weighted kappa ≥ 0.7) |
| 6 | Final report run | anyone | 1 h | Set `evaluate.approved_only: true`, rerun `05` |
| 7 | Acceptance test on the offline 8-core machine | owner | 1–2 h | `bench/run_bench.py` with the network disabled (see README §4) |
| 8 | Project-lead sign-off | project lead | — | Model-drafted references deviate from proposal §11.1 (guideline §1a) |

**If quality falls short of ROUGE-1 0.40:** the plan in PROGRESS decisions 2, 14 and 17 lists the
options. Compare strategies (`hybrid` vs `segmentwise`), use more training data (relax the faithfulness
filter), train for longer or at a different LR, or use a larger GPU (L4/A100 with Colab Pro).

## 3. Getting set up

1. **GitHub:** clone `https://github.com/assaabriiii/Offline-Arabic-Text-Summarization-Module`. Work on a
   branch and open pull requests to `main`.
2. **Google Drive:** all data, models and results live in the owner's Drive folder `MyDrive/arabic_summ/`
   (≈ 5 GB). Either:
   * the owner shares that folder with you and you add a shortcut to it in your own "My Drive" at the
     same path (`MyDrive/arabic_summ`), **or**
   * you rebuild everything in your own Drive by running `01_data` (about 10 minutes, CPU).
     The reference drafts are in git at `benchmark_sources/reference_drafts.json` (added 2026-10-16 at the
     owner's request); `01_data` cell 8b uses that copy when Drive has none.

   Only one person should run a given notebook at a time on a shared Drive. Two concurrent runs
   would write the same checkpoints.
3. **Local machine (optional, for code changes):**
   ```bash
   python3.13 -m venv .venv && source .venv/bin/activate
   pip install -e ".[test]" torch transformers accelerate   # tests run on CPU in ~20 s
   pytest
   ```
   Notebooks are generated: edit `tools/build_notebooks.py`, then run `python tools/build_notebooks.py`.
   Don't edit the `.ipynb` files by hand.

## 4. Running notebooks in Colab: practical tips (learned the hard way)

* Open the notebook from GitHub: `File → Open notebook → GitHub`, or use
  `https://colab.research.google.com/github/assaabriiii/Offline-Arabic-Text-Summarization-Module/blob/main/notebooks/03_finetune.ipynb`.
* Make sure the runtime is a **T4 GPU** (`Runtime → Change runtime type`).
* Run **cell 0**, then **cell 1** by hand. Cell 1 mounts Drive and needs your Google sign-in; it times
  out after about 2 minutes. Then click in cell 2 and choose **Runtime → Run cell and below**.
* Keep the Colab window **in front** and the computer awake (`caffeinate -d` on macOS). GPU sessions
  drop when the window is in the background or the machine sleeps.
* Never open the same notebook in two tabs.
* After a disconnect, repeat the steps above. Finished steps are skipped, and training resumes from the
  last saved epoch.
* "Too many sessions": go to `Runtime → Manage sessions` and terminate old ones.
* The GPU-hour ledger (`logs/gpu_ledger/`) records every training session automatically. The proposal
  requires this.

## 5. Things to know before changing anything

* `configs/default.yaml` holds every setting. When a code fix invalidates earlier training runs, bump
  `finetune.train.run_tag` (currently `r2`) so broken checkpoints are never resumed. The broken `r1`
  checkpoints are in `MyDrive/arabic_summ/models/runs/*_r1*` or under the untagged names, and can be deleted.
* Training uses Colab's **transformers 5**. ONNX export and the offline runtime use **transformers 4.57.6**
  in a separate Python 3.13 environment (`requirements/colab-onnx.txt`, `requirements/cpu-runtime.txt`),
  because optimum-onnx requires 4.x. Tokenizers saved by v5 load under 4.x through fallbacks in
  `src/arabic_summ/models.py`.
* The benchmark's primary strategy is fixed in config before scoring (`evaluate.primary_strategy`). Don't
  tune anything on the benchmark: tune on XL-Sum validation only.
* Licenses: XL-Sum is CC BY-NC-SA 4.0 (non-commercial), which carries over to the fine-tuned model.
  AraT5v2 is research-only.

## 6. Who to ask

The project owner (repository `assaabriiii`) has the proposal, Drive access and the Colab history.
`PROGRESS.md` records every decision with its reason. If something there conflicts with the proposal,
flag it to the owner before changing it.
