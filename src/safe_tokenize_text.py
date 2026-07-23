"""Safely tokenizes the 'text' domain (rj + gutenberg_corpus.txt + chat_corpus.txt,
same "\n".join concatenation data.py's _load_text_domain uses) under a given
tokenizer, for the first time since the Gutenberg expansion (EXTRA_COUNT
900->2000, DRAMA_COUNT 300->500) grew gutenberg_corpus.txt to ~1.3GB.

Same safety fix as safe_tokenize_breadth.py, applied to a corpus ~8x bigger:
a single SentencePieceProcessor.encode() call on the whole concatenated
string previously spiked to ~19.5GB peak RSS on a 167MB corpus (see
safe_tokenize_breadth.py's own docstring) -- naively encoding a 1.3GB
corpus in one call would extrapolate to well beyond this machine's 39GB
RAM and OOM-kill the process before any output is even flushed. This is
almost certainly why the prior "text xxl retrain" attempt
(runs/text_base_xxl_rwkv_rank96/train.log) is a 0-byte file with no
config.json: `train.py`'s `_load_text_domain` calls `_tokenize_corpus`
directly, no chunking, on this exact (already-expanded) corpus.

Streams chunk-by-chunk across all three files as one continuous logical
stream (never splitting mid-line), encoding each ~10MB chunk separately --
peak memory bounded by chunk size plus the growing id list, not by
whatever internal overhead a single huge encode() call carries. Saves to
the exact cache path data.py's _tokenize_corpus already expects
(data/cache/text_{tok.cache_key}.pt), so `load_lm_corpus("text", tok)`
works unmodified afterward -- this is a RAM-safety variant of the same
tokenize-and-cache step, not a new data format.
"""
from pathlib import Path

import torch

from data import CACHE_DIR
from tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
RJ_PATH = ROOT / "data" / "text" / "romeo_and_juliet.txt"
GUTENBERG_PATH = ROOT / "data" / "text" / "gutenberg_corpus.txt"
CHAT_PATH = ROOT / "data" / "text" / "chat_corpus.txt"
CHUNK_BYTES = 10_000_000


def _stream_sources():
    """Yields (path, is_last) for every existing source, in the same order
    and "\n"-joined semantics as data.py's _load_text_domain."""
    paths = [RJ_PATH]
    if GUTENBERG_PATH.exists():
        paths.append(GUTENBERG_PATH)
    if CHAT_PATH.exists():
        paths.append(CHAT_PATH)
    return paths


def chunked_tokenize_multi(tok: Tokenizer, paths: list[Path], chunk_bytes: int = CHUNK_BYTES) -> torch.Tensor:
    """Per-chunk tensors, concatenated once at the end via torch.cat --
    not a single growing Python list[int]. At this corpus's real scale
    (~8x safe_tokenize_breadth.py's 167MB target), a flat Python list of
    ints carries ~30-40 bytes/token of object overhead versus a tensor's
    8 -- for an estimated ~300M+ tokens that's a ~10GB+ difference, real
    risk of OOM on this machine's 39GB total RAM. Chunk tensors are
    compact from the moment they're created."""
    chunks: list[torch.Tensor] = []
    total = 0
    buf = ""
    for i, path in enumerate(paths):
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            while True:
                data = f.read(chunk_bytes)
                if not data:
                    break
                buf += data
                last_nl = buf.rfind("\n")
                if last_nl == -1:
                    continue  # pathological: no newline in 10MB, just keep buffering
                to_encode, buf = buf[: last_nl + 1], buf[last_nl + 1 :]
                chunk_ids = tok.encode(to_encode)
                chunks.append(torch.tensor(chunk_ids, dtype=torch.long))
                total += len(chunk_ids)
                print(f"  ...{total:,} tokens so far (in {path.name})", flush=True)
        if i < len(paths) - 1:
            buf += "\n"  # matches "\n".join(parts) between sources
    if buf:
        chunk_ids = tok.encode(buf)
        chunks.append(torch.tensor(chunk_ids, dtype=torch.long))
        total += len(chunk_ids)
    return torch.cat(chunks)


def main(vocab_size: int = 32768, variant: str = "balanced"):
    tok = Tokenizer(vocab_size=vocab_size, variant=variant)
    cache_path = CACHE_DIR / f"text_{tok.cache_key}.pt"
    if cache_path.exists():
        print(f"already cached: {cache_path}")
        return
    paths = _stream_sources()
    total_bytes = sum(p.stat().st_size for p in paths)
    print(f"tokenizing {[p.name for p in paths]} ({total_bytes:,} bytes total) under {tok.cache_key}...")
    ids = chunked_tokenize_multi(tok, paths)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(ids, cache_path)
    print(f"saved {len(ids):,} tokens to {cache_path}")
    return len(ids)


if __name__ == "__main__":
    import resource
    n_tokens = main()
    print(f"peak_rss_mb: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f}")
