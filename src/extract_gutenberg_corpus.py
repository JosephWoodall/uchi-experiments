"""Curated bulk text corpus from Project Gutenberg -- approved relaxation
of "hand-picked, not scraped" to "curated bulk, still license-clean": a
fixed, reviewed list of well-known public-domain titles (not "download
everything"), spanning fiction, philosophy, history, and science for
breadth beyond Shakespeare alone. Every text is genuinely public domain in
the US (Gutenberg's own basis for hosting them).

Strips Gutenberg's standard boilerplate header/footer (license text, not
the actual work) via its own stable, well-documented start/end markers.
Skips (logs, does not silently ignore) any ID that fails to fetch or
doesn't contain the expected markers, rather than assuming every ID in
the curated list is still valid.
"""
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

# Curated: well-known, high-confidence public-domain titles, spanning
# multiple subjects for breadth (not just novels) -- reviewed list, not a
# scrape of Gutenberg's full catalog.
BOOK_IDS = [
    11,     # Alice's Adventures in Wonderland
    84,     # Frankenstein
    1342,   # Pride and Prejudice
    76,     # Adventures of Huckleberry Finn
    74,     # The Adventures of Tom Sawyer
    2701,   # Moby Dick
    1661,   # The Adventures of Sherlock Holmes
    244,    # A Study in Scarlet
    98,     # A Tale of Two Cities
    1400,   # Great Expectations
    46,     # A Christmas Carol
    730,    # Oliver Twist
    174,    # The Picture of Dorian Gray
    345,    # Dracula
    768,    # Wuthering Heights
    1260,   # Jane Eyre
    43,     # The Strange Case of Dr Jekyll and Mr Hyde
    35,     # The Time Machine
    36,     # The War of the Worlds
    164,    # Twenty Thousand Leagues Under the Sea
    103,    # Around the World in Eighty Days
    2554,   # Crime and Punishment
    2600,   # War and Peace
    28054,  # The Brothers Karamazov
    5200,   # Metamorphosis
    996,    # Don Quixote
    135,    # Les Miserables
    1184,   # The Count of Monte Cristo
    1497,   # The Republic (Plato)
    1998,   # Thus Spoke Zarathustra
    4363,   # Beyond Good and Evil
    2680,   # Meditations (Marcus Aurelius)
    132,    # The Art of War
    1232,   # The Prince (Machiavelli)
    3300,   # The Wealth of Nations
    61,     # The Communist Manifesto
    147,    # Common Sense
    205,    # Walden
    1228,   # On the Origin of Species
    2009,   # Relativity: The Special and General Theory (Einstein)
    1946,   # The Interpretation of Dreams (Freud, translated)
    28233,  # Autobiography of Benjamin Franklin (alt ID kept as fallback)
    148,    # A Tale of Two Cities (dup guard tolerated -- dedup by id below anyway)
    16328,  # Beowulf
    2383,   # The Canterbury Tales
    26,     # Paradise Lost
    1322,   # Leaves of Grass
    8800,   # The Divine Comedy
    1524,   # Hamlet
    1533,   # Macbeth
    23042,  # Othello
    27761,  # The Adventures of Sherlock Holmes (alt, dedup guard)
    120,    # Treasure Island
    514,    # Little Women
    16,     # Peter Pan
    55,     # The Wonderful Wizard of Oz
    2591,   # Grimms' Fairy Tales
    21,     # Aesop's Fables
    6130,   # The Iliad
    1727,   # The Odyssey
]

HEADER_RE = re.compile(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)
FOOTER_RE = re.compile(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*", re.IGNORECASE | re.DOTALL)

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "text"


def fetch(book_id: int) -> str | None:
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        print(f"  id={book_id}: fetch failed ({e})")
        return None


def clean(raw_text: str) -> str | None:
    header_match = HEADER_RE.search(raw_text)
    footer_match = FOOTER_RE.search(raw_text)
    if not header_match or not footer_match:
        return None  # doesn't match expected Gutenberg boilerplate shape -- skip, don't guess
    body = raw_text[header_match.end(): footer_match.start()]
    return body.strip()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seen_ids = []
    chunks = []
    n_failed = 0

    for book_id in BOOK_IDS:
        if book_id in seen_ids:
            continue
        seen_ids.append(book_id)
        raw = fetch(book_id)
        if raw is None:
            n_failed += 1
            continue
        body = clean(raw)
        if body is None or len(body) < 1000:
            print(f"  id={book_id}: boilerplate markers not found or body too short -- skipped")
            n_failed += 1
            continue
        chunks.append(f"# --- gutenberg id={book_id} ---\n{body}")
        print(f"  id={book_id}: {len(body):,} chars")
        time.sleep(0.5)  # polite delay between requests to a shared public server

    combined = "\n\n".join(chunks)
    (OUT_DIR / "gutenberg_corpus.txt").write_text(combined)
    print(f"\ngutenberg_corpus.txt: {len(combined):,} chars from {len(chunks)} texts "
          f"({n_failed} failed/skipped out of {len(seen_ids)} attempted)")


if __name__ == "__main__":
    main()
