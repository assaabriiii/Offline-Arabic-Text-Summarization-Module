"""Organizational-report texts for the internal benchmark, fetched from public Arabic pages
of international organizations.

Only URLs + metadata are committed (benchmark_sources/org_reports.csv); the texts are
re-fetched with the same extraction code inside Colab and their content hash compared to
the recorded one. The pages are copyrighted by their publishers and are used for
non-commercial research evaluation only, never redistributed (see `terms_url`).
"""

from __future__ import annotations

import re
import time
import urllib.parse as up
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Iterator

from ..text.normalize import normalize_for_model

USER_AGENT = "Mozilla/5.0 (compatible; arabic-summ-benchmark/0.1; research)"


@dataclass(frozen=True)
class Source:
    key: str
    org: str
    base: str
    listings: tuple[str, ...]   # listing URLs; those containing {page} are paginated over `pages`
    pages: range
    link_re: str          # applied to the percent-decoded listing HTML
    terms_url: str
    license_note: str


SOURCES: dict[str, Source] = {s.key: s for s in [
    Source(
        "who_emro", "World Health Organization – Eastern Mediterranean (WHO EMRO)",
        "https://www.emro.who.int", ("https://www.emro.who.int/ar/media/news/Page-{page}-10.html",), range(2, 40),  # Page-1 is empty
        r'href="(?:https://www\.emro\.who\.int)?(/ar/media/news/(?!Page-)[^"#?]+\.html)"',
        "https://www.emro.who.int/ar/copyright/copyright.html",
        "© WHO. Website terms permit copying parts for research with attribution (URL); non-commercial; not redistributed.",
    ),
    Source(
        "unicef_mena", "UNICEF Middle East and North Africa",
        # the regional listing paginates client-side only, so Arabic country-office media pages are added
        "https://www.unicef.org", (
            "https://www.unicef.org/mena/ar/press-releases",
            *(f"https://www.unicef.org/{c}/ar/المركز-الإعلامي" for c in ("egypt", "sudan", "sop", "yemen", "iraq", "gulf")),
            "https://www.unicef.org/morocco/ar/البيانات-الصحفية",
        ), range(0, 1),
        r'href="(/[a-z]+/ar/(?:[^"#?]+/البيانات-الصحفية|press-releases/[^"#?]+))"',
        "https://www.unicef.org/legal",
        "© UNICEF. Used under website terms for non-commercial research evaluation with attribution; not redistributed.",
    ),
    Source(
        "unfpa", "United Nations Population Fund (UNFPA)",
        "https://www.unfpa.org", ("https://www.unfpa.org/ar/news?page={page}",), range(0, 25),
        r'href="(/ar/news/[^"#?]+)"',
        "https://www.unfpa.org/ar/شروط-الاستخدام",
        "© UNFPA. Used under website terms for non-commercial research evaluation with attribution; not redistributed.",
    ),
    Source(
        "unwomen_as", "UN Women – Arab States",
        "https://arabstates.unwomen.org", ("https://arabstates.unwomen.org/ar/news-and-events/stories?page={page}",), range(0, 30),
        r'href="(/ar/stories/(?:byan|byan-shfy|mqal|press-release|akhbar)/\d{4}/\d{2}/[^"#?]+)"',
        "https://www.unwomen.org/en/about-the-website/terms-of-use",
        "© UN Women. Used under website terms for non-commercial research evaluation with attribution; not redistributed.",
    ),
]}

# Trailing boilerplate in press releases (about-us blurbs, media contacts); text is cut at the
# first line that starts with one of these.
_TAIL_MARKERS = re.compile(
    r"^\s*(#+\s*)?(نبذة عن|حول (منظمة|اليونيسف|صندوق|هيئة)|عن (اليونيسف|منظمة|صندوق|هيئة)|"
    r"لمزيد من المعلومات|للمزيد من المعلومات|للاستفسارات|جهات الاتصال|للتواصل|ملاحظات للمحررين|"
    r"ملاحظة للمحررين|تابعوا|###)"
)


def fetch(url: str, retries: int = 5, timeout: int = 60, backoff: float = 2.0) -> str:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as e:  # network is flaky; retry with backoff
            last = e
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def iri_to_uri(url: str) -> str:
    parts = up.urlsplit(url)
    return up.urlunsplit(parts._replace(path=up.quote(parts.path, safe="/%-._~")))


def clean_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if _TAIL_MARKERS.match(line):
            break
        lines.append(line)
    return normalize_for_model("\n".join(lines))


def extract(html: str) -> tuple[str, str]:
    """Return (title, cleaned main text) using trafilatura in precision mode."""
    import trafilatura

    body = trafilatura.extract(html, favor_precision=True, include_comments=False, include_tables=False) or ""
    meta = trafilatura.extract_metadata(html)
    title = (meta.title if meta and meta.title else "").strip()
    return title, clean_text(body)


def article_links(source: Source, pages: Iterable[int] | None = None, delay: float = 1.0) -> Iterator[str]:
    seen: set[str] = set()
    for listing in source.listings:
        paged = "{page}" in listing
        for page in (pages if pages is not None else source.pages) if paged else (None,):
            try:
                html = up.unquote(fetch(iri_to_uri(listing.format(page=page) if paged else listing)))
            except RuntimeError:
                continue
            new = 0
            for path in re.findall(source.link_re, html):
                url = source.base + path
                if url not in seen:
                    seen.add(url)
                    new += 1
                    yield url
            time.sleep(delay)
            if paged and new == 0:  # past the last page
                break
