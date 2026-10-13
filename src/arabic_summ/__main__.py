"""Command-line entry point: ``python -m arabic_summ <command> ...``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config


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

    args = ap.parse_args(argv)
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
    return 2


if __name__ == "__main__":
    sys.exit(main())
