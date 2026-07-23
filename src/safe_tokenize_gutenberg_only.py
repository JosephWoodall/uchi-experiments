"""Safely tokenizes gutenberg_corpus.txt ALONE (Phase AJ rj/gutenberg pool
split -- rj and gutenberg used to share one combined "literary" cache, but
that let gutenberg's general prose dilute rj's own dramatic voice when the
two pools couldn't be weighted independently). Reuses safe_tokenize_text.py's
chunked, memory-safe approach (peak RSS ~5.6GB on the larger combined
rj+gutenberg+chat corpus, verified safe) -- same mechanism, single source.

Caches under the "gutenberg_only" key, matching
data.py's load_scale_up_corpus._tokenize_corpus_lazy(tok, "gutenberg_only", ...) call.
"""
from pathlib import Path

import torch

from data import CACHE_DIR
from safe_tokenize_text import chunked_tokenize_multi
from tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
GUTENBERG_PATH = ROOT / "data" / "text" / "gutenberg_corpus.txt"


def main(model_path: str):
    tok = Tokenizer(model_path=model_path)
    cache_path = CACHE_DIR / f"gutenberg_only_{tok.cache_key}.pt"
    if cache_path.exists():
        print(f"already cached: {cache_path}")
        return
    total_bytes = GUTENBERG_PATH.stat().st_size
    print(f"tokenizing {GUTENBERG_PATH.name} ({total_bytes:,} bytes) under {tok.cache_key}...")
    ids = chunked_tokenize_multi(tok, [GUTENBERG_PATH])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(ids, cache_path)
    print(f"saved {len(ids):,} tokens to {cache_path}")
    return len(ids)


if __name__ == "__main__":
    import sys
    import resource
    n_tokens = main(sys.argv[1])
    print(f"peak_rss_mb: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f}")
