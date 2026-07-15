"""Dataset loading for the ablation harness. Two corpora: 'rj' (Romeo &
Juliet, plain LM) and 'code' (stdlib extract, plain LM + paired doc/code
views for the jepa-aux arm). Everything tokenized once and cached as .pt
tensors so repeat sweep runs skip re-tokenizing.
"""
import json
from pathlib import Path

import torch

from tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"


def _tokenize_corpus(tok: Tokenizer, name: str, path: Path) -> torch.Tensor:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{name}.pt"
    if cache_path.exists():
        return torch.load(cache_path)
    ids = tok.encode(path.read_text())
    t = torch.tensor(ids, dtype=torch.long)
    torch.save(t, cache_path)
    return t


def load_lm_corpus(name: str, tok: Tokenizer, val_frac: float = 0.1):
    """name in {'rj', 'code'} -> (train_ids, val_ids), 1D long tensors."""
    paths = {
        "rj": ROOT / "data" / "text" / "romeo_and_juliet.txt",
        "code": ROOT / "data" / "code" / "corpus.txt",
    }
    ids = _tokenize_corpus(tok, name, paths[name])
    n_val = int(len(ids) * val_frac)
    return ids[:-n_val], ids[-n_val:]


def load_code_pairs(tok: Tokenizer, block_size: int, val_frac: float = 0.1):
    """Doc<->code pairs for the jepa-aux arm, tokenized and truncated/padded
    to block_size. Returns (train_pairs, val_pairs), each a list of
    (doc_ids: LongTensor[block_size], code_ids: LongTensor[block_size]).
    """
    cache_path = CACHE_DIR / f"pairs_bs{block_size}.pt"
    if cache_path.exists():
        pairs = torch.load(cache_path)
    else:
        raw = [json.loads(line) for line in (ROOT / "data" / "code" / "pairs.jsonl").open()]
        pairs = []
        for p in raw:
            doc_ids = _pad_or_truncate(tok.encode(p["doc"]), block_size, tok)
            code_ids = _pad_or_truncate(tok.encode(p["code"]), block_size, tok)
            pairs.append((doc_ids, code_ids))
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(pairs, cache_path)

    n_val = max(1, int(len(pairs) * val_frac))
    return pairs[:-n_val], pairs[-n_val:]


def _pad_or_truncate(ids: list[int], block_size: int, tok: Tokenizer) -> torch.Tensor:
    pad_id = tok.sp.pad_id()
    ids = ids[:block_size]
    ids = ids + [pad_id] * (block_size - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def get_lm_batch(ids: torch.Tensor, batch_size: int, block_size: int, n_future: int = 1):
    """Random contiguous chunks for teacher-forced LM training.
    Returns x: (B, block_size), targets: (B, block_size, n_future) where
    targets[:, t, k] = token at position t + k + 1 (k=0 is the standard
    next-token target; k>0 are the extra mtp targets), -1 where out of range.
    """
    max_start = len(ids) - block_size - n_future
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([ids[s : s + block_size] for s in starts])
    targets = torch.full((batch_size, block_size, n_future), -1, dtype=torch.long)
    for k in range(n_future):
        for i, s in enumerate(starts.tolist()):
            targets[i, :, k] = ids[s + k + 1 : s + k + 1 + block_size]
    return x, targets
