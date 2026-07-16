"""Single shared tokenizer trained jointly on both corpora (text + code), so
the same vocab can later be reused as the "text" leg of a multimodal token
space (see tasks/core_principle.md).

Grown from 1024 -> 8192: the small vocab turned out to be the most
foundational, least-tested limiting factor for output quality -- rare
identifiers/words fragment to near-character level at 1024 tokens, capping
AST-fact precision, identifier grounding, and basic fluency all at once
(see tasks/ducky.md's architecture writeup). Versioned by vocab size
(spm_{vocab_size}.model/.vocab) rather than one fixed filename -- the old
1024-vocab tokenizer stays on disk, untouched, so any earlier checkpoint
that needs it can still find it. This is a new generation of Ducky, not a
patch on the old one: every checkpoint trained under the 1024 vocab is
incompatible with the new default (different vocab means different token
ID meanings, different embedding table shape) and won't reload correctly
against Tokenizer() going forward without explicitly requesting
vocab_size=1024.
"""
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
TOK_DIR = ROOT / "data" / "tokenizer"
VOCAB_SIZE = 8192


def _model_prefix(vocab_size: int) -> Path:
    return TOK_DIR / f"spm_{vocab_size}"


def train_if_missing(vocab_size: int = VOCAB_SIZE) -> Path:
    model_prefix = _model_prefix(vocab_size)
    model_path = model_prefix.with_suffix(".model")
    if model_path.exists():
        return model_path

    TOK_DIR.mkdir(parents=True, exist_ok=True)
    rj = (ROOT / "data" / "text" / "romeo_and_juliet.txt").read_text()
    code = (ROOT / "data" / "code" / "corpus.txt").read_text()
    combined_path = TOK_DIR / f"_combined_{vocab_size}.txt"
    combined_path.write_text(rj + "\n" + code)

    spm.SentencePieceTrainer.train(
        input=str(combined_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        unk_id=3,
    )
    combined_path.unlink()
    return model_path


class Tokenizer:
    def __init__(self, vocab_size: int = VOCAB_SIZE):
        model_path = train_if_missing(vocab_size)
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))

    @property
    def vocab_size(self) -> int:
        return self.sp.vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)


if __name__ == "__main__":
    tok = Tokenizer()
    print(f"vocab_size={tok.vocab_size}")
    sample = "ROMEO: But soft, what light through yonder window breaks?"
    ids = tok.encode(sample)
    print(f"'{sample}' -> {len(ids)} tokens -> '{tok.decode(ids)}'")
