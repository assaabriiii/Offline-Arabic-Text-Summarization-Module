"""Build benchmark_sources/org_reports.csv: candidate organizational-report pages (300-1000 words).

    python tools/harvest_org_reports.py --per-source 45

Crawls the listing pages of each source in arabic_summ.data.web_sources.SOURCES, extracts the
main text and keeps pages that are predominantly Arabic and inside the benchmark length range.
Writes URLs + metadata + content hash only (no text), so the CSV is safe to commit.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arabic_summ.config import load_config  # noqa: E402
from arabic_summ.data.benchmark import content_hash, length_bin  # noqa: E402
from arabic_summ.data.web_sources import SOURCES, article_links, extract, fetch, iri_to_uri  # noqa: E402
from arabic_summ.io_utils import utc_now  # noqa: E402
from arabic_summ.text.normalize import arabic_letter_ratio  # noqa: E402
from arabic_summ.text.words import count_words  # noqa: E402

FIELDS = ["source_key", "org", "url", "title", "word_count", "length_bin", "content_hash",
          "retrieved_date", "terms_url", "license_note"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=45)
    ap.add_argument("--out", type=Path, default=ROOT / "benchmark_sources" / "org_reports.csv")
    ap.add_argument("--max-pages-fetched", type=int, default=150, help="per source")
    ap.add_argument("--sources", nargs="*", help="only re-harvest these source keys; other rows are kept")
    args = ap.parse_args()
    bcfg = load_config(ROOT / "configs" / "default.yaml")["benchmark"]

    rows, hashes = [], set()
    if args.sources and args.out.exists():
        with open(args.out, encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["source_key"] not in args.sources]
        hashes = {r["content_hash"] for r in rows}
    for src in SOURCES.values():
        if args.sources and src.key not in args.sources:
            continue
        kept = fetched = 0
        for url in article_links(src):
            if kept >= args.per_source or fetched >= args.max_pages_fetched:
                break
            fetched += 1
            try:
                title, text = extract(fetch(iri_to_uri(url)))
            except RuntimeError as e:
                print("  skip (fetch)", e, file=sys.stderr)
                continue
            n = count_words(text)
            h = content_hash(text)
            ok = (bcfg["input_min_words"] <= n <= bcfg["input_max_words"]
                  and arabic_letter_ratio(text) >= 0.9 and h not in hashes)
            print(f"  {src.key:12s} {'KEEP' if ok else 'drop'} {n:5d}w  {title[:60]}")
            if ok:
                hashes.add(h)
                kept += 1
                rows.append({"source_key": src.key, "org": src.org, "url": url, "title": title,
                             "word_count": n, "length_bin": length_bin(n, bcfg["length_bins"]),
                             "content_hash": h, "retrieved_date": utc_now()[:10],
                             "terms_url": src.terms_url, "license_note": src.license_note})
            time.sleep(1.0)
        print(f"{src.key}: kept {kept} of {fetched} fetched")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
