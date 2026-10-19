"""Command-line entry point: ``python -m arabic_summ <command> ...``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .io_utils import read_json


def _cfg(args):
    over = {"paths": {"root": args.root}} if args.root else None
    return load_config(*args.config, overrides=over)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="arabic_summ")
    ap.add_argument("--config", nargs="+", default=["configs/default.yaml"], help="YAML file(s), merged in order")
    ap.add_argument("--root", help="override paths.root (artifact root)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("data-prepare", help="download XL-Sum, deduplicate, build benchmark scaffold")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("benchmark-validate", help="validate benchmark CSVs + leakage check")
    p.add_argument("--strict", action="store_true", help="freeze mode: only 'selected' rows, all approved")

    p = sub.add_parser("benchmark-import", help="add a folder of .txt files as benchmark candidates")
    p.add_argument("folder", type=Path)
    p.add_argument("--domain", required=True, choices=["news", "org_report"])
    p.add_argument("--sources-csv", type=Path)

    p = sub.add_parser("benchmark-import-drafts",
                       help="fill empty references.csv rows from a drafts JSON (status stays pending)")
    p.add_argument("drafts", type=Path, help='JSON {sample_id: {"summary": ..., "key_points": [...]}}')
    p.add_argument("--writer-id", required=True)
    p.add_argument("--reference-source", default="llm_draft", choices=["human", "source", "llm_draft"])

    p = sub.add_parser("onnx-export", help="export a fine-tuned model to ONNX FP32 and dynamic INT8 (ONNX env)")
    p.add_argument("model_dir", type=Path)
    p.add_argument("out_root", type=Path, help="writes <out_root>/fp32 and <out_root>/int8")
    p.add_argument("--arch", default="avx2", help="optimum AutoQuantizationConfig preset (avx2, avx512_vnni, arm64)")

    p = sub.add_parser("optimize-measure", help="measure one CPU variant from a spec JSON (fresh process)")
    p.add_argument("spec", type=Path)
    p.add_argument("out", type=Path)

    p = sub.add_parser("summarize-batch", help="summarize benchmark texts with one system (resumable JSONL)")
    p.add_argument("spec", type=Path)
    p.add_argument("out", type=Path)

    p = sub.add_parser("summarize-refinish", help="re-apply the post-processing to a keep_parts run (new JSONL)")
    p.add_argument("spec", type=Path)
    p.add_argument("inp", type=Path)
    p.add_argument("out", type=Path)

    p = sub.add_parser("summarize", help="summarize Arabic text offline (file, or stdin with '-')")
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--input", default="-", help="UTF-8 text file, or - for stdin")
    p.add_argument("--ratio", type=float)
    p.add_argument("--strategy", choices=["segmentwise", "hybrid", "direct", "extractive", "lead"])
    p.add_argument("--threads", type=int)
    p.add_argument("--json", action="store_true", help="print the full result as JSON")

    p = sub.add_parser("serve", help="local web UI (Gradio) on 127.0.0.1")
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--threads", type=int)

    args = ap.parse_args(argv)
    if args.cmd in ("summarize", "serve"):
        from .app.service import SummarizerService, force_offline

        force_offline()
        cfg = _cfg(args)
        svc = SummarizerService(cfg, args.model_dir, threads=args.threads)
        if args.cmd == "serve":
            from .app.web import build, launch

            launch(build(svc, cfg), args.host, args.port)
            return 0
        import json as _json

        text = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8-sig")
        res = svc.summarize(text, ratio=args.ratio, strategy=args.strategy)
        print(_json.dumps(res, ensure_ascii=False, indent=1) if args.json else res["summary"])
        if not args.json:
            print(f"\n[{res['input_words']} -> {res['output_words']} words, bounds {res['bounds']}, "
                  f"{res['seconds']['total']} s, {res['variant']}]", file=sys.stderr)
        return 0
    if args.cmd == "summarize-batch":
        from .eval import run_batch

        return run_batch.run(read_json(args.spec), args.out)
    if args.cmd == "summarize-refinish":
        from .eval import run_batch

        return run_batch.refinish(read_json(args.spec), args.inp, args.out)
    if args.cmd == "onnx-export":
        from .optimize import onnx_export as X

        print("fp32:", X.export_fp32(args.model_dir, args.out_root / "fp32"))
        print("int8:", X.quantize_int8(args.out_root / "fp32", args.out_root / "int8", args.arch))
        return 0
    if args.cmd == "optimize-measure":
        from .optimize import measure

        return measure.main(str(args.spec), str(args.out))
    cfg = _cfg(args)

    if args.cmd == "data-prepare":
        from .data import prepare

        prepare.step_download(cfg, force=args.force)
        prepare.step_dedup(cfg, force=args.force)
        prepare.step_benchmark_scaffold(cfg, force=args.force)
        rep, _ = prepare.validate_benchmark(cfg)
        print(rep.summary())
        return 0
    if args.cmd == "benchmark-validate":
        from .data import prepare

        rep, leak = prepare.validate_benchmark(cfg, strict=args.strict)
        print(rep.summary())
        print(f"leakage: {leak['n_leaks']} of {leak['items_checked']} items")
        return 0 if rep.ok else 1
    if args.cmd == "benchmark-import":
        from . import config as C
        from .data import benchmark as B

        added = B.import_text_folder(C.path(cfg, "benchmark"), args.folder, args.domain, cfg["benchmark"], args.sources_csv)
        print(f"added {len(added)} new candidate texts")
        return 0
    if args.cmd == "benchmark-import-drafts":
        from .data import prepare

        res = prepare.import_reference_drafts(cfg, args.drafts, args.writer_id, args.reference_source)
        print(res)
        return 0 if res["validation_ok"] else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
