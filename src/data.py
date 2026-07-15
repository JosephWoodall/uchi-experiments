"""Dataset loading for the ablation harness. Two corpora: 'rj' (Romeo &
Juliet, plain LM) and 'code' (stdlib extract, plain LM + paired doc/code
views for the jepa-aux arm). Everything tokenized once and cached as .pt
tensors so repeat sweep runs skip re-tokenizing.
"""
import json
from pathlib import Path

import torch

from codec import N_CODES, train_codec
from tokenizer import VOCAB_SIZE as TEXT_VOCAB_SIZE
from tokenizer import Tokenizer

# Unified vocab layout: text/code share the BPE vocab (offset 0, they were
# tokenized with the same jointly-trained tokenizer already); pixel and
# audio each get an offset range; four marker tokens (one per modality)
# are prepended to every training crop so the model has an explicit signal
# for which regime it's in, not just the implicit token-id range.
TEXT_OFFSET = 0
PIXEL_OFFSET = TEXT_VOCAB_SIZE
AUDIO_OFFSET = PIXEL_OFFSET + N_CODES
MARK_RJ = AUDIO_OFFSET + N_CODES
MARK_CODE = MARK_RJ + 1
MARK_PIXEL = MARK_RJ + 2
MARK_AUDIO = MARK_RJ + 3
UNIFIED_VOCAB_SIZE = MARK_RJ + 4
MARKERS = {"rj": MARK_RJ, "code": MARK_CODE, "pixel": MARK_PIXEL, "audio": MARK_AUDIO}

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


def load_joint_modalities(tok: Tokenizer, val_frac: float = 0.1):
    """All four modalities, offset into one shared vocab. Returns
    {name: (train_ids, val_ids)}. Deliberately NOT concatenated into one
    static sequence -- text/code (~50K tokens each) would drown out pixel/
    audio (4-8K tokens) by sheer length. get_joint_batch instead samples
    which modality each training example comes from per-example, so every
    modality gets proportionate exposure regardless of its native size.
    """
    rj_train, rj_val = load_lm_corpus("rj", tok, val_frac)
    code_train, code_val = load_lm_corpus("code", tok, val_frac)
    pixel_train, pixel_val = load_modality_corpus("pixel", val_frac)
    audio_train, audio_val = load_modality_corpus("audio", val_frac)
    return {
        "rj": (rj_train, rj_val),
        "code": (code_train, code_val),
        "pixel": (pixel_train + PIXEL_OFFSET, pixel_val + PIXEL_OFFSET),
        "audio": (audio_train + AUDIO_OFFSET, audio_val + AUDIO_OFFSET),
    }


def get_joint_batch(ids_by_modality: dict, weights: dict, batch_size: int, block_size: int):
    """Each of the batch_size examples independently picks a modality
    (weighted random), then a random crop of that modality's own tensor,
    with the modality's marker token prepended as position 0. n_future=1
    only -- joint mode isn't wired for mtp/jepa-aux in this pass.
    Returns x: (B, block_size), targets: (B, block_size, 1).
    """
    names = list(ids_by_modality.keys())
    w = torch.tensor([weights[n] for n in names], dtype=torch.float)
    choice_idx = torch.multinomial(w, batch_size, replacement=True)
    xs, ys = [], []
    for c in choice_idx.tolist():
        name = names[c]
        ids = ids_by_modality[name]
        start = torch.randint(0, len(ids) - block_size, (1,)).item()
        crop = ids[start : start + block_size]
        x = torch.cat([torch.tensor([MARKERS[name]]), crop[:-1]])
        xs.append(x)
        ys.append(crop)
    x = torch.stack(xs)
    targets = torch.stack(ys).unsqueeze(-1)
    return x, targets


def load_modality_corpus(name: str, val_frac: float = 0.1):
    """name in {'pixel', 'audio'} -> (train_ids, val_ids), code-index
    sequences from the trained VQ-VAE codec (see codec.py). Vocab size for
    both is codec.N_CODES, not the text tokenizer's vocab.
    """
    codes = train_codec(name)
    n_val = max(1, int(len(codes) * val_frac))
    return codes[:-n_val], codes[-n_val:]


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
