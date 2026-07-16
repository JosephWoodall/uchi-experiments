"""Curated bulk text corpus from Project Gutenberg -- approved relaxation
of "hand-picked, not scraped" to "curated bulk, still license-clean": every
text is genuinely public domain in the US (Gutenberg's own basis for
hosting them). Selection is now driven by Gutenberg's own public catalog
(pg_catalog.csv, fetched from gutenberg.org) rather than hand-listing IDs
from memory -- reliable at the scale needed to meaningfully close the
Chinchilla data gap (a few hundred books, not a few dozen), and avoids
guessing whether any specific ID is even valid.

BOOK_IDS below is the original hand-picked seed list (kept for continuity
with the first pass); EXTRA_COUNT more are drawn from the catalog itself,
filtered to English-language "Text" entries, deterministically sampled
(fixed seed) so re-running this script is reproducible rather than a
moving target.

Strips Gutenberg's standard boilerplate header/footer (license text, not
the actual work) via its own stable, well-documented start/end markers.
Skips (logs, does not silently ignore) any ID that fails to fetch or
doesn't contain the expected markers.
"""
import csv
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
EXTRA_COUNT = 900  # sampled from the catalog, on top of the seed list below
SEED = 7

# Original hand-picked seed list, kept for continuity with the first pass.
BOOK_IDS = [
    11, 84, 1342, 76, 74, 2701, 1661, 244, 98, 1400, 46, 730, 174, 345, 768,
    1260, 43, 35, 36, 164, 103, 2554, 2600, 28054, 5200, 996, 135, 1184,
    1497, 1998, 4363, 2680, 132, 1232, 3300, 61, 147, 205, 1228, 2009, 1946,
    28233, 148, 16328, 2383, 26, 1322, 8800, 1524, 1533, 23042, 27761, 120,
    514, 16, 55, 2591, 21, 6130, 1727,
]

HEADER_RE = re.compile(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)
FOOTER_RE = re.compile(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*", re.IGNORECASE | re.DOTALL)

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "text"


def fetch_catalog_ids(extra_count: int, seed: int, exclude: set) -> list:
    """Real catalog IDs, not guessed -- filtered to English-language "Text"
    entries (excludes audio recordings, non-English works), deterministically
    sampled so this script is reproducible.
    """
    catalog_path = OUT_DIR / "_pg_catalog_cache.csv"
    if not catalog_path.exists():
        with urllib.request.urlopen(CATALOG_URL, timeout=60) as resp:
            catalog_path.write_bytes(resp.read())

    candidates = []
    with catalog_path.open(encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Language") != "en" or row.get("Type") != "Text":
                continue
            try:
                book_id = int(row["Text#"])
            except (KeyError, ValueError):
                continue
            if book_id in exclude:
                continue
            candidates.append(book_id)

    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:extra_count]


def fetch(book_id: int, retries: int = 2) -> str | None:
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                ConnectionError, OSError) as e:
            # A shared public server serving ~1000 sequential requests will
            # have occasional transient drops (RemoteDisconnected, resets,
            # etc.) -- one bad connection must not kill a 900-book run.
            if attempt < retries:
                time.sleep(1.0)
                continue
            print(f"  id={book_id}: fetch failed after {retries + 1} attempts ({e})")
            return None


def clean(raw_text: str) -> str | None:
    header_match = HEADER_RE.search(raw_text)
    footer_match = FOOTER_RE.search(raw_text)
    if not header_match or not footer_match:
        return None
    body = raw_text[header_match.end(): footer_match.start()]
    return body.strip()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_ids = list(dict.fromkeys(BOOK_IDS))  # dedupe, preserve order
    extra_ids = fetch_catalog_ids(EXTRA_COUNT, SEED, exclude=set(all_ids))
    all_ids.extend(extra_ids)
    print(f"seed list: {len(BOOK_IDS)} ids, catalog-sampled: {len(extra_ids)} ids, total attempted: {len(all_ids)}")

    chunks = []
    n_failed = 0
    for i, book_id in enumerate(all_ids):
        raw = fetch(book_id)
        if raw is None:
            n_failed += 1
            continue
        body = clean(raw)
        if body is None or len(body) < 1000:
            n_failed += 1
            continue
        chunks.append(f"# --- gutenberg id={book_id} ---\n{body}")
        if (i + 1) % 50 == 0:
            print(f"  progress: {i + 1}/{len(all_ids)} attempted, {len(chunks)} succeeded so far")
        time.sleep(0.3)  # polite delay to a shared public server

    combined = "\n\n".join(chunks)
    (OUT_DIR / "gutenberg_corpus.txt").write_text(combined)
    print(f"\ngutenberg_corpus.txt: {len(combined):,} chars from {len(chunks)} texts "
          f"({n_failed} failed/skipped out of {len(all_ids)} attempted)")


if __name__ == "__main__":
    main()
