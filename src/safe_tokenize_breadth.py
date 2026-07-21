"""Safely tokenizes data/code/corpus_breadth.txt (167MB) under a given
tokenizer for the first time, avoiding the transient RAM spike already
observed once this session (tokenizing a large corpus under a new
tokenizer identity for the first time hit ~19.5GB peak RSS via a single
SentencePieceProcessor.encode() call on the whole string). Streams the
file in ~10MB, line-bounded chunks, encodes each chunk separately, and
concatenates the resulting token-id lists -- peak memory bounded by chunk
size plus the growing id list (~1.5GB for ~40M tokens), not by however
much internal overhead a single huge encode() call carries.

Saves to the exact cache path data.py's _tokenize_corpus/
load_weighted_code_corpus already expect (data/cache/code_breadth_
{tok.cache_key}.pt), so every existing downstream function works
unmodified afterward -- this is a RAM-safety variant of the same
tokenize-and-cache step, not a new data format.
"""
from pathlib import Path

import torch

from data import CACHE_DIR
from tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
BREADTH_PATH = ROOT / "data" / "code" / "corpus_breadth.txt"
CHUNK_BYTES = 10_000_000


def chunked_tokenize(tok: Tokenizer, path: Path, chunk_bytes: int = CHUNK_BYTES) -> torch.Tensor:
    ids: list[int] = []
    buf = ""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        while True:
            data = f.read(chunk_bytes)
            if not data:
                break
            buf += data
            # Only tokenize up to the last newline in buf, so no chunk boundary
            # ever splits a line mid-way (irrelevant to token correctness, but
            # keeps chunk sizes predictable and avoids any edge-case surprises).
            last_nl = buf.rfind("\n")
            if last_nl == -1:
                continue  # pathological: no newline in 10MB, just keep buffering
            to_encode, buf = buf[: last_nl + 1], buf[last_nl + 1 :]
            ids.extend(tok.encode(to_encode))
            print(f"  ...{len(ids):,} tokens so far", flush=True)
        if buf:
            ids.extend(tok.encode(buf))
    return torch.tensor(ids, dtype=torch.long)


def main(vocab_size: int = 32768, variant: str = "balanced"):
    tok = Tokenizer(vocab_size=vocab_size, variant=variant)
    cache_path = CACHE_DIR / f"code_breadth_{tok.cache_key}.pt"
    if cache_path.exists():
        print(f"already cached: {cache_path}")
        return
    print(f"tokenizing {BREADTH_PATH} ({BREADTH_PATH.stat().st_size:,} bytes) under {tok.cache_key}...")
    ids = chunked_tokenize(tok, BREADTH_PATH)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(ids, cache_path)
    print(f"saved {len(ids):,} tokens to {cache_path}")
    return len(ids)


if __name__ == "__main__":
    import resource
    n_tokens = main()
    print(f"peak_rss_mb: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f}")
