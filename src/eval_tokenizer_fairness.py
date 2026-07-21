"""Fertility (tokens/char) + cross-domain fairness (Gini coefficient) for a
given tokenizer, measured on small held-out slices from text, code, and
terminal-commands -- the concrete, checkable test for whether one domain
is dominating the shared BPE vocab (see tasks/ducky.md's tokenizer-fairness
section for why this question came up). Works against any SentencePiece
.model file, so the same script measures both the current production
tokenizer and any new candidate.

Reads only a small tail slice of each domain's source file (seek + read,
never the whole file) -- gutenberg_corpus.txt alone is 1.3GB, and this is a
diagnostic, not something that needs the full corpus.
"""
import json
import sys
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent

# Held-out tail slices, roughly matching data.py's val_frac=0.1 convention
# scaled to each domain's actual size -- not the full corpus.
SLICE_CHARS = {"text": 200_000, "code": 200_000, "terminal": 57_000}


def _tail_slice(path: Path, n_chars: int) -> str:
    size = path.stat().st_size
    with path.open("rb") as f:
        f.seek(max(0, size - n_chars * 4))  # *4 headroom for multi-byte utf-8
        raw = f.read()
    return raw.decode("utf-8", errors="ignore")[-n_chars:]


def held_out_samples() -> dict:
    return {
        "text": _tail_slice(ROOT / "data" / "text" / "gutenberg_corpus.txt", SLICE_CHARS["text"]),
        "code": _tail_slice(ROOT / "data" / "code" / "corpus_core.txt", SLICE_CHARS["code"]),
        "terminal": _tail_slice(ROOT / "data" / "terminal" / "nl2bash_corpus.txt", SLICE_CHARS["terminal"]),
    }


def fertility(sp: spm.SentencePieceProcessor, text: str) -> float:
    return len(sp.encode(text, out_type=int)) / len(text)


def gini(values: list) -> float:
    """Standard Gini coefficient over a small set of per-domain fertility
    values -- 0 = every domain compresses equally well, higher = one domain
    is getting a noticeably worse (or better) deal than the others. Same
    fairness-on-a-per-group-metric idea as "Equity with Efficiency"
    (arXiv:2606.15044), small enough to implement directly, no new
    dependency.
    """
    n = len(values)
    if n == 0:
        return 0.0
    sorted_v = sorted(values)
    total = sum(sorted_v)
    if total == 0:
        return 0.0
    cum = sum((i + 1) * v for i, v in enumerate(sorted_v))
    return (2 * cum) / (n * total) - (n + 1) / n


def evaluate(model_path: str) -> dict:
    sp = spm.SentencePieceProcessor(model_file=model_path)
    samples = held_out_samples()
    per_domain = {name: round(fertility(sp, text), 4) for name, text in samples.items()}
    return {
        "tokenizer": Path(model_path).name,
        "vocab_size": sp.vocab_size(),
        "fertility_tokens_per_char": per_domain,
        "gini_across_domains": round(gini(list(per_domain.values())), 4),
    }


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "tokenizer" / "spm_32768.model")
    print(json.dumps(evaluate(model_path), indent=2))
