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
EXTRA_COUNT = 0  # already fetched 900 general books in the prior pass; bump this again for a further general expansion
SEED = 7

# Drama/plays specifically -- unlike general prose, plays are literally
# formatted as back-and-forth dialogue between named characters, the one
# real conversational-STRUCTURE signal available in Gutenberg's catalog
# (filtered by its own Subjects/Bookshelves metadata, not guessed titles).
DRAMA_COUNT = 300
DRAMA_SEED = 13

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


def fetch_drama_catalog_ids(count: int, seed: int, exclude: set) -> list:
    """Real catalog IDs tagged as drama/plays in Gutenberg's own Subjects/
    Bookshelves metadata -- plays are actually formatted as dialogue
    (character: line, character: line), the one reliable conversational-
    structure signal available without guessing which specific titles are
    dialogue-heavy.
    """
    catalog_path = OUT_DIR / "_pg_catalog_cache.csv"
    candidates = []
    with catalog_path.open(encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Language") != "en" or row.get("Type") != "Text":
                continue
            subjects = (row.get("Subjects") or "").lower()
            bookshelves = (row.get("Bookshelves") or "").lower()
            if "drama" not in subjects and "plays" not in subjects and "drama" not in bookshelves:
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
    return candidates[:count]


def already_fetched_ids(corpus_path: Path) -> set:
    """IDs already present in an existing gutenberg_corpus.txt, parsed from
    its own '# --- gutenberg id=N ... ---' markers -- lets a follow-up run
    add new texts without re-fetching what's already there.
    """
    if not corpus_path.exists():
        return set()
    return {int(m) for m in re.findall(r"# --- gutenberg id=(\d+)", corpus_path.read_text())}


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


def _fetch_batch(ids: list, tag: str) -> tuple:
    chunks = []
    n_failed = 0
    for i, book_id in enumerate(ids):
        raw = fetch(book_id)
        if raw is None:
            n_failed += 1
            continue
        body = clean(raw)
        if body is None or len(body) < 1000:
            n_failed += 1
            continue
        chunks.append(f"# --- gutenberg id={book_id} type={tag} ---\n{body}")
        if (i + 1) % 50 == 0:
            print(f"  [{tag}] progress: {i + 1}/{len(ids)} attempted, {len(chunks)} succeeded so far")
        time.sleep(0.3)  # polite delay to a shared public server
    return chunks, n_failed


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus_path = OUT_DIR / "gutenberg_corpus.txt"
    existing_text = corpus_path.read_text() if corpus_path.exists() else ""
    existing_ids = already_fetched_ids(corpus_path)
    print(f"already present: {len(existing_ids)} texts (kept, not re-fetched)")

    all_ids = list(dict.fromkeys(BOOK_IDS))
    extra_ids = fetch_catalog_ids(EXTRA_COUNT, SEED, exclude=set(all_ids) | existing_ids)
    general_ids = [i for i in (all_ids + extra_ids) if i not in existing_ids]

    drama_ids = fetch_drama_catalog_ids(DRAMA_COUNT, DRAMA_SEED, exclude=existing_ids | set(general_ids))
    print(f"new general: {len(general_ids)} ids, new drama/dialogue: {len(drama_ids)} ids")

    general_chunks, n_failed_general = _fetch_batch(general_ids, tag="general") if general_ids else ([], 0)
    drama_chunks, n_failed_drama = _fetch_batch(drama_ids, tag="drama")

    new_combined = "\n\n".join(general_chunks + drama_chunks)
    combined = (existing_text + "\n\n" + new_combined) if existing_text else new_combined
    corpus_path.write_text(combined)
    n_failed = n_failed_general + n_failed_drama
    n_attempted = len(general_ids) + len(drama_ids)
    print(f"\ngutenberg_corpus.txt: {len(combined):,} chars total "
          f"({len(existing_ids)} kept + {len(general_chunks) + len(drama_chunks)} new = "
          f"{len(existing_ids) + len(general_chunks) + len(drama_chunks)} texts, "
          f"{n_failed} failed/skipped out of {n_attempted} newly attempted)")


if __name__ == "__main__":
    main()
