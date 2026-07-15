"""Single shared tokenizer trained jointly on both corpora (text + code), so
the same vocab can later be reused as the "text" leg of a multimodal token
space (see tasks/core_principle.md). Small vocab keeps sequences short,
which is what actually keeps training fast at this scale, not just param
count.
"""
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
TOK_DIR = ROOT / "data" / "tokenizer"
MODEL_PREFIX = TOK_DIR / "spm"
VOCAB_SIZE = 1024


def train_if_missing(vocab_size: int = VOCAB_SIZE) -> Path:
    model_path = MODEL_PREFIX.with_suffix(".model")
    if model_path.exists():
        return model_path

    TOK_DIR.mkdir(parents=True, exist_ok=True)
    rj = (ROOT / "data" / "text" / "romeo_and_juliet.txt").read_text()
    code = (ROOT / "data" / "code" / "corpus.txt").read_text()
    combined_path = TOK_DIR / "_combined.txt"
    combined_path.write_text(rj + "\n" + code)

    spm.SentencePieceTrainer.train(
        input=str(combined_path),
        model_prefix=str(MODEL_PREFIX),
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
