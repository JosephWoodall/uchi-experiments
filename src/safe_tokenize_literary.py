"""Safely tokenizes rj + gutenberg_corpus.txt ONLY (deliberately excluding
chat_corpus.txt) for the real scale-up's "literary" text pool
(tasks/ducky.md Phase AI). Reuses safe_tokenize_text.py's chunked,
memory-safe approach (peak RSS ~5.6GB on the full rj+gutenberg+chat
corpus, verified safe) -- same mechanism, different source list.

chat_corpus.txt is only ~0.015% of the combined rj+gutenberg+chat byte
count, so reusing the existing "text" cache would have been numerically
almost identical -- but the user's exclusion was an explicit quality
decision (anonymized IRC chatroom noise), not a volume one, so this
tokenizes cleanly without it rather than accepting a technically-tiny
but deliberately-rejected contamination.
"""
from pathlib import Path

import torch

from data import CACHE_DIR
from safe_tokenize_text import chunked_tokenize_multi
from tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
RJ_PATH = ROOT / "data" / "text" / "romeo_and_juliet.txt"
GUTENBERG_PATH = ROOT / "data" / "text" / "gutenberg_corpus.txt"


def main(model_path: str):
    tok = Tokenizer(model_path=model_path)
    cache_path = CACHE_DIR / f"literary_{tok.cache_key}.pt"
    if cache_path.exists():
        print(f"already cached: {cache_path}")
        return
    paths = [RJ_PATH, GUTENBERG_PATH]
    total_bytes = sum(p.stat().st_size for p in paths)
    print(f"tokenizing {[p.name for p in paths]} ({total_bytes:,} bytes total) under {tok.cache_key}...")
    ids = chunked_tokenize_multi(tok, paths)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(ids, cache_path)
    print(f"saved {len(ids):,} tokens to {cache_path}")
    return len(ids)


if __name__ == "__main__":
    import sys
    import resource
    n_tokens = main(sys.argv[1])
    print(f"peak_rss_mb: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f}")
