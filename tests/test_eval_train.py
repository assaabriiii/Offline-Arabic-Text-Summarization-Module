import json

import pandas as pd
import pytest

from arabic_summ.backends.base import EchoBackend
from arabic_summ.config import load_config
from arabic_summ.eval import facts as F
from arabic_summ.eval import rouge as R
from arabic_summ.train import data as D
from arabic_summ.train import decoding as DEC
from arabic_summ.train import ledger as L
from arabic_summ.train.select import select_best

CFG = load_config("configs/default.yaml")
SRC = "قالت منظمة الصحة العالمية إن 959 شخصا لقوا حتفهم في 10 محافظات باليمن، وإن الاحتياجات ارتفعت إلى 31 مليون دولار."
GOOD = "أفادت منظمة الصحة العالمية بوفاة 959 شخصا في 10 محافظات، وارتفاع الاحتياجات إلى 31 مليون دولار."
BAD = "قال الحوثيون في صنعاء إن 1200 شخص قتلوا في هجوم على مقر المخابرات."


def test_rouge_handles_arabic_and_normalization():
    assert R.score(SRC, SRC) == {"rouge1": 1.0, "rouge2": 1.0, "rougeL": 1.0}
    # alef / teh-marbuta / digit-script variants and punctuation match after matching normalization
    assert R.score("إلى المدرسة ٣", "الى المدرسه 3")["rouge1"] == 1.0
    assert R.score(SRC, "")["rouge1"] == 0.0
    s = R.score("a b c d", "a b x d")
    assert s["rouge1"] == pytest.approx(0.75) and s["rouge2"] == pytest.approx(1 / 3) and s["rougeL"] == pytest.approx(0.75)
    assert R.corpus([SRC, SRC], [SRC, ""])["rouge1"] == pytest.approx(0.5)


def test_fact_checker_flags_unsupported_numbers_and_words():
    g, b = F.check(GOOD, SRC), F.check(BAD, SRC)
    assert g.numbers_ok and g.number_precision == 1.0 and g.novel_rate < 0.4
    assert b.unsupported_numbers == ["1200"] and b.number_precision == 0.0 and b.novel_rate > 0.8
    assert "صنعاء" in b.novel_content
    # Arabic-Indic digits and thousands separators match their ASCII form
    assert F.check("بلغ العدد ٨٦٬٢٩٠ أسرة", "يقدر بنحو 86,290 أسرة").numbers_ok
    agg = F.aggregate([g, b, F.check("لا أرقام هنا", SRC)])
    assert agg["number_precision"] == pytest.approx(0.5) and agg["numbers_ok_share"] == pytest.approx(2 / 3)


def test_stem_strips_clitics_but_keeps_short_words():
    assert F.stem("والمحافظات") == "محافظ"
    assert F.stem("بلد") == "بلد"


def test_filter_pairs_and_relaxation():
    df = pd.DataFrame({"text": [SRC] * 4, "summary": [GOOD, BAD, GOOD, "منظمة الصحة العالمية تقول إن الناس سعداء جدا اليوم"]})
    kept, rep = D.filter_pairs(df, {"max_novel_rate": 0.4, "min_keep_fraction": 0.0})
    assert list(kept.index) == [0, 2] and rep["dropped_numbers"] == 1 and rep["dropped_novel_only"] == 1
    kept, rep = D.filter_pairs(df, {"max_novel_rate": 0.0, "min_keep_fraction": 0.75})
    assert rep["relaxed_to_min_keep"] and len(kept) == 3 and 1 not in kept.index


def test_select_best_respects_faithfulness_tolerance():
    rows = [{"id": "a", "rouge1": 0.40, "number_precision": 0.80},
            {"id": "b", "rouge1": 0.35, "number_precision": 0.95},
            {"id": "c", "rouge1": 0.37, "number_precision": 0.94}]
    assert select_best(rows)["id"] == "c"
    assert select_best(rows, faith_tolerance=0.2)["id"] == "a"


def test_decoding_grid_and_evaluate():
    g = DEC.grid(CFG["finetune"]["decoding_grid"])
    assert len(g) == 10 and sum(d["num_beams"] == 1 for d in g) == 1
    assert all(d["no_repeat_ngram_size"] == 3 for d in g)
    rows, best = DEC.tune(EchoBackend(), [SRC, SRC], [GOOD, GOOD], {"num_beams": [1, 2], "length_penalty": [1.0, 2.0],
                          "no_repeat_ngram_size": 3}, 20, CFG["finetune"]["select"], log=lambda *_: None)
    assert len(rows) == 3 and set(best) == {"num_beams", "length_penalty", "no_repeat_ngram_size", "early_stopping"}


def test_gpu_ledger_records_sessions(tmp_path):
    cfg = load_config("configs/default.yaml", overrides={"paths": {"root": str(tmp_path)}})
    with L.GPUSession(cfg, "exp1", "test") as s:
        s.beat(force=True, step=3)
    rec = json.loads(next((tmp_path / "logs" / "gpu_ledger").glob("*.json")).read_text())
    assert rec["status"] == "finished" and rec["step"] == 3 and rec["experiment"] == "exp1"
    assert L.summary(cfg)["sessions"] == 1


def _tiny_seq2seq(tmp_path):
    tok_lib = pytest.importorskip("tokenizers")
    tr = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    from tokenizers import Tokenizer, models, pre_tokenizers

    words = sorted({w for t in (SRC, GOOD, BAD) for w in t.split()})
    vocab = {w: i for i, w in enumerate(["<pad>", "<s>", "</s>", "<unk>"] + words)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    from tokenizers.processors import TemplateProcessing

    tk.post_processor = TemplateProcessing(single="<s> $A </s>", special_tokens=[("<s>", 1), ("</s>", 2)])
    tok = tr.PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", bos_token="<s>", eos_token="</s>",
                                     unk_token="<unk>")
    conf = tr.BartConfig(vocab_size=len(vocab), d_model=16, encoder_layers=1, decoder_layers=1,
                         encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32,
                         decoder_ffn_dim=32, max_position_embeddings=64, pad_token_id=0, bos_token_id=1,
                         eos_token_id=2, decoder_start_token_id=2, forced_eos_token_id=2)
    tr.set_seed(0)
    d = tmp_path / "tiny"
    tr.BartForConditionalGeneration(conf).save_pretrained(d)
    tok.save_pretrained(d)
    return tok, d


def test_train_run_end_to_end_resumable(tmp_path):
    tok, base = _tiny_seq2seq(tmp_path)
    from arabic_summ.text.normalize import NormalizationPolicy
    from arabic_summ.train import seq2seq as S2S
    from arabic_summ.train.provenance import export_model

    cfg = load_config("configs/default.yaml", overrides={"paths": {"root": str(tmp_path / "root")}})
    pol = NormalizationPolicy.from_dict(cfg["normalization"]["model"])
    df = pd.DataFrame({"text": [SRC] * 8, "summary": [GOOD] * 8})
    rows = D.tokenize_pairs(df, tok, pol, 64, 32)
    hp = {**cfg["finetune"]["train"], "learning_rate": 1e-3, "max_epochs": 2, "per_device_batch_size": 4,
          "effective_batch_size": 4, "eval_batch_size": 4, "max_target_tokens": 16, "fp16": False,
          "dataloader_workers": 0, "logging_steps": 1}
    name = S2S.run_name("tiny", 1e-3, 1.0)
    res = S2S.train_run(cfg, name, {"repo_id": str(base)}, tok, rows, rows[:2], [GOOD] * 2, [SRC] * 2, hp)
    assert res["epochs_run"] == 2 and len(res["per_epoch"]) == 2 and "number_precision" in res["best"]
    assert (S2S.run_dir(cfg, name) / "best" / "config.json").exists()
    assert not (S2S.run_dir(cfg, name) / "checkpoints").exists()
    assert S2S.train_run(cfg, name, {"repo_id": "unused"}, tok, rows, rows, [], [], hp) == res   # skipped
    import shutil
    shutil.rmtree(S2S.run_dir(cfg, name) / "best")          # marker present but outputs lost -> retrains
    again = S2S.train_run(cfg, name, {"repo_id": str(base)}, tok, rows, rows[:2], [GOOD] * 2, [SRC] * 2, hp)
    assert (S2S.run_dir(cfg, name) / "best" / "config.json").exists() and again["epochs_run"] == 2
    prov = export_model(cfg, res["best_dir"], "tiny_ft", {"num_beams": 2}, {"base": "tiny"})
    out = tmp_path / "root" / "models" / "tiny_ft"
    assert json.loads((out / "decoding.json").read_text()) == {"num_beams": 2}
    assert "model.safetensors" in prov["files"] and len(prov["files"]["model.safetensors"]["sha256"]) == 64


def test_checkpoints_go_to_local_dir_and_are_removed(tmp_path):
    tok, base = _tiny_seq2seq(tmp_path)
    from arabic_summ.text.normalize import NormalizationPolicy
    from arabic_summ.train import seq2seq as S2S

    cfg = load_config("configs/default.yaml", overrides={"paths": {"root": str(tmp_path / "root")}})
    pol = NormalizationPolicy.from_dict(cfg["normalization"]["model"])
    rows = D.tokenize_pairs(pd.DataFrame({"text": [SRC] * 4, "summary": [GOOD] * 4}), tok, pol, 64, 32)
    local = tmp_path / "local_ckpt"
    hp = {**cfg["finetune"]["train"], "learning_rate": 1e-3, "max_epochs": 1, "per_device_batch_size": 4,
          "effective_batch_size": 4, "eval_batch_size": 4, "max_target_tokens": 16, "fp16": False,
          "dataloader_workers": 0, "checkpoint_dir": str(local), "keep_checkpoints": True}
    S2S.train_run(cfg, "loc", {"repo_id": str(base)}, tok, rows, rows[:2], [GOOD] * 2, [SRC] * 2, hp)
    assert list((local / "loc").glob("checkpoint-*")), "epoch checkpoints belong on the local disk"
    out = S2S.run_dir(cfg, "loc")
    assert (out / "best" / "config.json").exists() and not (out / "checkpoints").exists()
    # parent missing (e.g. /content off Colab) -> next to the run
    assert S2S.checkpoint_dir({"checkpoint_dir": str(tmp_path / "nope" / "x")}, out, "loc") == out / "checkpoints"
    assert S2S.checkpoint_dir({}, out, "loc") == out / "checkpoints"


def test_train_run_eval_keeps_dataset_order(tmp_path):
    """group_by_length must not reorder eval predictions (transformers 5 applies it to eval too)."""
    tok, base = _tiny_seq2seq(tmp_path)
    from arabic_summ.text.normalize import NormalizationPolicy
    from arabic_summ.train import seq2seq as S2S

    cfg = load_config("configs/default.yaml", overrides={"paths": {"root": str(tmp_path / "root")}})
    pol = NormalizationPolicy.from_dict(cfg["normalization"]["model"])
    w = SRC.split()
    texts = [" ".join(w[:k]) for k in (3, 15, 5, 12, 2, 9, 4, 14)]        # lengths in no particular order
    df = pd.DataFrame({"text": texts, "summary": texts})
    rows = D.tokenize_pairs(df, tok, pol, 64, 32)
    hp = {**cfg["finetune"]["train"], "learning_rate": 1e-3, "max_epochs": 1, "per_device_batch_size": 2,
          "effective_batch_size": 2, "eval_batch_size": 2, "max_target_tokens": 16, "fp16": False,
          "dataloader_workers": 0, "logging_steps": 1}
    res = S2S.train_run(cfg, "order", {"repo_id": str(base)}, tok, rows, rows, texts, texts, hp)
    assert len(res["per_epoch"]) == 1      # compute_metrics raises if predictions come back reordered


def test_compute_metrics_detects_misordered_predictions():
    np = pytest.importorskip("numpy")
    from types import SimpleNamespace

    from arabic_summ.train.seq2seq import make_compute_metrics

    class Tok:
        pad_token_id = 0

        def batch_decode(self, ids, skip_special_tokens=True):
            return ["x"] * len(ids)

    labels = [[5, 6, 2], [7, 2]]
    cm = make_compute_metrics(Tok(), ["a", "b"], ["a", "b"], labels)
    padded = np.array([[5, 6, 2], [7, 2, -100]])
    cm(SimpleNamespace(predictions=np.zeros((2, 3), int), label_ids=padded))           # in order: fine
    with pytest.raises(RuntimeError, match="not in eval-set order"):
        cm(SimpleNamespace(predictions=np.zeros((2, 3), int), label_ids=padded[::-1]))


def test_ensure_special_tokens_adds_missing_eos_template():
    tr = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    from arabic_summ.models import ensure_special_tokens

    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "قال": 4, "الوزير": 5}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()     # no post-processor: like v5 BarthezTokenizer
    tok = tr.PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", pad_token="<pad>",
                                     unk_token="<unk>")
    assert tok("قال الوزير")["input_ids"] == [4, 5]
    tok = ensure_special_tokens(tok)
    assert tok("قال الوزير")["input_ids"] == [0, 4, 5, 2]
    assert tok(text_target="قال الوزير")["input_ids"] == [0, 4, 5, 2]
    assert tok("قال الوزير الوزير", max_length=3, truncation=True)["input_ids"][-1] == 2
    # already-correct tokenizers are left alone
    assert ensure_special_tokens(tok)("قال")["input_ids"] == [0, 4, 2]


def test_run_name_tag():
    from arabic_summ.train.seq2seq import run_name

    assert run_name("arabart", 3e-5, 0.25) == "arabart_lr3e-05_f0.25"
    assert run_name("arabart", 3e-5, 0.25, "r2") == "arabart_lr3e-05_f0.25_r2"


def test_latest_checkpoint_skips_incomplete_and_searches_all_dirs(tmp_path):
    from arabic_summ.train.seq2seq import CHECKPOINT_FILES, latest_checkpoint

    def ckpt(d, step, skip=()):
        c = tmp_path / d / f"checkpoint-{step}"
        c.mkdir(parents=True)
        for f in (*CHECKPOINT_FILES, "model.safetensors"):
            if f not in skip:
                (c / f).write_text("x")
        return str(c)

    local, drive = tmp_path / "local", tmp_path / "drive"
    assert latest_checkpoint(local, drive) is None
    a = ckpt("drive", 1183)
    assert latest_checkpoint(local, drive) == a                      # copy on Drive is found
    ckpt("drive", 3549, skip=("optimizer.pt",))                      # half-uploaded: ignored
    b = ckpt("local", 2366)
    assert latest_checkpoint(local, drive) == b


def test_load_tokenizer_reads_v5_saved_files_with_list_extra_special_tokens(tmp_path):
    """transformers 5 writes `extra_special_tokens` as a list, which 4.x rejects (04 ONNX env, 2026-10-18)."""
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    from arabic_summ.models import _fast_tokenizer_from_files, load_tokenizer

    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "قال": 4, "الوزير": 5}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tk.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "backend": "tokenizers", "bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>",
        "unk_token": "<unk>", "extra_special_tokens": ["<s>", "<pad>", "</s>"],
        "tokenizer_class": "TokenizersBackend"}), encoding="utf-8")
    for tok in (_fast_tokenizer_from_files(tmp_path), load_tokenizer(str(tmp_path))):
        assert (tok.eos_token, tok.pad_token_id) == ("</s>", 1)
    assert load_tokenizer(str(tmp_path))("قال الوزير")["input_ids"] == [0, 4, 5, 2]
