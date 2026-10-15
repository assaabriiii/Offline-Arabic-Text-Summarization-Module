"""One resumable fine-tuning run with the Hugging Face Seq2SeqTrainer (transformers v5).

Layout on Drive:  models/runs/<name>/checkpoints/  (epoch checkpoints, deleted when the run
finishes unless keep_checkpoints), models/runs/<name>/best/ (best epoch by validation ROUGE-1),
result.json, log_history.json. A finished run is skipped via its step marker; an interrupted one
resumes from its last epoch checkpoint (optimizer, scheduler, RNG and early-stopping state).
"""

from __future__ import annotations

import inspect
import shutil
from pathlib import Path
from typing import Mapping, Sequence

from .. import config as C
from ..eval import facts, rouge
from ..io_utils import mark_step_done, read_json, step_done, utc_now, write_json
from ..text.words import count_words
from .ledger import GPUSession


def run_dir(cfg: Mapping, name: str) -> Path:
    return C.path(cfg, "models", "runs", name, "x").parent


def run_name(model_key: str, lr: float, fraction: float) -> str:
    return f"{model_key}_lr{lr:g}_f{fraction:g}"


def training_args(**kw):
    """Seq2SeqTrainingArguments, keeping only arguments the installed transformers accepts
    (v5 renamed group_by_length -> train_sampling_strategy; Colab and local versions differ)."""
    from transformers import Seq2SeqTrainingArguments

    params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    if kw.pop("group_by_length", False):
        if "train_sampling_strategy" in params:
            kw["train_sampling_strategy"] = "group_by_length"
        elif "group_by_length" in params:
            kw["group_by_length"] = True
    dropped = sorted(k for k in kw if k not in params)
    args = Seq2SeqTrainingArguments(**{k: v for k, v in kw.items() if k in params})
    return args, dropped


def make_compute_metrics(tokenizer, refs: Sequence[str], sources: Sequence[str]):
    import numpy as np

    idx = [facts.SourceIndex(s) for s in sources]

    def compute(pred):
        ids = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        ids = np.where(ids < 0, tokenizer.pad_token_id, ids)
        hyps = [h.strip() for h in tokenizer.batch_decode(ids, skip_special_tokens=True)]
        out = rouge.corpus(refs[: len(hyps)], hyps)
        out.update({k: v for k, v in facts.aggregate([ix.check(h) for ix, h in zip(idx, hyps)]).items() if k != "n"})
        out["gen_words"] = float(np.mean([count_words(h) for h in hyps])) if hyps else 0.0
        return {k: round(float(v), 5) for k, v in out.items()}

    return compute


def train_run(cfg: Mapping, name: str, model_ref: Mapping, tokenizer, train_rows: list, eval_rows: list,
              eval_refs: Sequence[str], eval_sources: Sequence[str], hp: Mapping, notebook: str = "03_finetune",
              extra_meta: Mapping | None = None) -> dict:
    """Train (or resume, or skip if finished) one configuration and return its result record."""
    step = f"03_run_{name}"
    out = run_dir(cfg, name)
    if step_done(cfg, step):
        return read_json(out / "result.json")

    import torch
    from transformers import (AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, EarlyStoppingCallback,
                              Seq2SeqTrainer, TrainerCallback, set_seed)
    from transformers.trainer_utils import get_last_checkpoint

    set_seed(int(hp["seed"]))
    model = AutoModelForSeq2SeqLM.from_pretrained(model_ref["repo_id"], revision=model_ref.get("revision"))
    fp16 = bool(hp.get("fp16")) and torch.cuda.is_available()
    per_dev = int(hp["per_device_batch_size"])
    accum = max(1, int(hp["effective_batch_size"]) // per_dev)
    ckpt_dir = out / "checkpoints"
    args, dropped = training_args(
        output_dir=str(ckpt_dir), eval_strategy="epoch", save_strategy="epoch", logging_strategy="steps",
        logging_steps=int(hp.get("logging_steps", 50)), learning_rate=float(hp["learning_rate"]),
        per_device_train_batch_size=per_dev, per_device_eval_batch_size=int(hp.get("eval_batch_size", 16)),
        gradient_accumulation_steps=accum, num_train_epochs=float(hp["max_epochs"]),
        weight_decay=float(hp.get("weight_decay", 0.0)), warmup_steps=float(hp.get("warmup", 0.0)),
        label_smoothing_factor=float(hp.get("label_smoothing", 0.0)), fp16=fp16,
        predict_with_generate=True, generation_max_length=int(hp["max_target_tokens"]),
        generation_num_beams=int(hp.get("eval_num_beams", 1)),
        load_best_model_at_end=True, metric_for_best_model="rouge1", greater_is_better=True,
        save_total_limit=2, seed=int(hp["seed"]), data_seed=int(hp["seed"]), report_to="none",
        group_by_length=True, dataloader_num_workers=int(hp.get("dataloader_workers", 2)),
    )
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8 if fp16 else None)
    session = GPUSession(cfg, name, notebook, run=name, learning_rate=float(hp["learning_rate"]),
                         model=model_ref["repo_id"], train_rows=len(train_rows), **dict(extra_meta or {}))

    class Heartbeat(TrainerCallback):
        def on_log(self, a, state, control, **kw):
            session.beat(epoch=round(state.epoch or 0, 3), step=state.global_step)

        def on_evaluate(self, a, state, control, metrics=None, **kw):
            session.beat(force=True, epoch=round(state.epoch or 0, 3), last_eval=metrics)

    trainer = Seq2SeqTrainer(
        model=model, args=args, data_collator=collator, train_dataset=train_rows, eval_dataset=eval_rows,
        processing_class=tokenizer, compute_metrics=make_compute_metrics(tokenizer, eval_refs, eval_sources),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(hp["patience"])), Heartbeat()],
    )
    resume = get_last_checkpoint(str(ckpt_dir)) if ckpt_dir.exists() else None
    print(f"[{name}] train={len(train_rows)} eval={len(eval_rows)} accum={accum} fp16={fp16} "
          f"resume={resume} ignored_args={dropped}")
    try:
        trainer.train(resume_from_checkpoint=resume)
    except BaseException as e:
        session.close(f"interrupted: {type(e).__name__}")
        raise

    best = out / "best"
    trainer.save_model(str(best))
    tokenizer.save_pretrained(str(best))
    hist = trainer.state.log_history
    evals = [h for h in hist if "eval_rouge1" in h]
    best_eval = max(evals, key=lambda h: h["eval_rouge1"]) if evals else {}
    result = {
        "name": name, "model": dict(model_ref), "hyperparameters": dict(hp), "accumulation_steps": accum,
        "fp16": fp16, "train_rows": len(train_rows), "eval_rows": len(eval_rows),
        "epochs_run": round(trainer.state.epoch or 0, 3), "global_step": trainer.state.global_step,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best": {k.removeprefix("eval_"): v for k, v in best_eval.items()},
        "per_epoch": [{k.removeprefix("eval_"): v for k, v in h.items()} for h in evals],
        "best_dir": str(best), "finished_utc": utc_now(), "ignored_args": dropped,
    }
    write_json(out / "log_history.json", hist)
    write_json(out / "result.json", result)
    session.close("finished", best_rouge1=result["best"].get("rouge1"))
    if not hp.get("keep_checkpoints", False):
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    mark_step_done(cfg, step, outputs=[best / "config.json", out / "result.json"], best=result["best"])
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
