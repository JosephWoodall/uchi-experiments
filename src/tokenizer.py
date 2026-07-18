"""Single shared tokenizer trained jointly on both corpora (text + code), so
the same vocab can later be reused as the "text" leg of a multimodal token
space (see tasks/core_principle.md).

Grown twice now: 1024 -> 8192 -> 32768. The first growth (1024->8192) fixed
rare identifiers/words fragmenting to near-character level. This growth is
driven by the corpus itself growing ~30-100x (stdlib + curated site-packages
libraries for code; 1,255 Gutenberg texts + chat for text) and getting far
more diverse -- ML-library identifiers (torch/jax/sklearn) and much broader
English vocabulary than the 8192 vocab was ever trained to compress well.
Versioned by vocab size (spm_{vocab_size}.model/.vocab) rather than one
fixed filename -- every earlier vocab generation stays on disk, untouched,
so any earlier checkpoint that needs it can still find it. Checkpoints
trained under a smaller vocab are incompatible with a newer default
(different vocab means different token ID meanings, different embedding
table shape) and won't reload correctly against Tokenizer() going forward
without explicitly requesting the matching vocab_size.
"""
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
TOK_DIR = ROOT / "data" / "tokenizer"
VOCAB_SIZE = 32768


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
    gutenberg_path = ROOT / "data" / "text" / "gutenberg_corpus.txt"
    gutenberg = gutenberg_path.read_text() if gutenberg_path.exists() else ""
    chat_path = ROOT / "data" / "text" / "chat_corpus.txt"
    chat = chat_path.read_text() if chat_path.exists() else ""
    combined_path = TOK_DIR / f"_combined_{vocab_size}.txt"
    combined_path.write_text(rj + "\n" + gutenberg + "\n" + chat + "\n" + code)

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
        # The combined training text is now ~600MB+ (vs a few MB originally)
        # -- cap and shuffle the sentences SentencePiece actually trains on
        # so this stays tractable, rather than processing every line of
        # every book. 5M sentences is generous for stable BPE merge
        # statistics well beyond this vocab size.
        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
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
