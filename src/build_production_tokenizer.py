"""Real, full-corpus-scale production tokenizer that includes the
terminal-command domain and doesn't let text dominate BPE merge selection
-- see tasks/ducky.md's tokenizer-fairness section for the small-scale
comparison this generalizes from.

alpha=0.5 (temperature/alpha-weighted resampling, weight_domain =
size_domain**alpha) was chosen over the small-scale sweep's marginally
better alpha=0.3 deliberately: at real domain sizes (text ~1.3GB, code
~178MB, terminal ~574KB -- ~2300:300:1, not the small comparison's ~4:4:1),
alpha=0.3 implies repeating the entire terminal corpus ~156x to hit its
target share -- pathological, mostly-duplicate-driven merges. alpha=0.5
implies ~39x, still meaningful upsampling of a genuinely diverse small
corpus (206 unique flags, 102 utilities per NL2Bash's own stats) without
that risk. Real, computed numbers are printed at build time, not assumed.

Builds the combined training corpus via STREAMING line-by-line I/O (never
holds a full domain's ~1.3GB of text in one Python string, unlike
tokenizer.py's existing train_if_missing) -- deliberately more careful
given concurrent CPU/RAM load from another process. Saved as
data/tokenizer/spm_32768_balanced.model via tokenizer.py's new `variant`
param -- alongside, not replacing, spm_32768.model.
"""
import random
from pathlib import Path

import sentencepiece as spm

from tokenizer import _model_prefix

ROOT = Path(__file__).resolve().parent.parent
TOK_DIR = ROOT / "data" / "tokenizer"
VOCAB_SIZE = 32768
VARIANT = "balanced"
ALPHA = 0.5
MAX_REPEAT = 50  # safety cap regardless of what the alpha formula implies

DOMAIN_SOURCES = {
    "text": [
        ROOT / "data" / "text" / "romeo_and_juliet.txt",
        ROOT / "data" / "text" / "gutenberg_corpus.txt",
        ROOT / "data" / "text" / "chat_corpus.txt",
    ],
    "code": [
        ROOT / "data" / "code" / "corpus_core.txt",
        ROOT / "data" / "code" / "corpus_breadth.txt",
    ],
    "terminal": [ROOT / "data" / "terminal" / "nl2bash_corpus.txt"],
}


def domain_byte_sizes() -> dict:
    return {
        name: sum(p.stat().st_size for p in paths if p.exists())
        for name, paths in DOMAIN_SOURCES.items()
    }


def alpha_targets(sizes: dict, alpha: float, total_bytes: int) -> dict:
    weights = {name: size**alpha for name, size in sizes.items()}
    total_weight = sum(weights.values())
    return {name: total_bytes * w / total_weight for name, w in weights.items()}


def _stream_domain(paths: list, keep_prob: float, repeat: int, out_f, rng: random.Random) -> None:
    """Streams each source file line-by-line -- never reads a whole file
    into memory. keep_prob<1 randomly drops lines (domain bigger than its
    target share); repeat>1 rewrites every kept line `repeat` times
    (domain smaller than its target, e.g. terminal) -- capped at
    MAX_REPEAT so a tiny domain's exact lines can't dominate merge
    statistics through pure duplication.
    """
    repeat = min(repeat, MAX_REPEAT)
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if keep_prob < 1.0 and rng.random() > keep_prob:
                    continue
                for _ in range(repeat):
                    out_f.write(line)


def build_combined_corpus(alpha: float = ALPHA, seed: int = 7) -> Path:
    sizes = domain_byte_sizes()
    total_bytes = sum(sizes.values())
    targets = alpha_targets(sizes, alpha, total_bytes)
    rounded_targets = {k: round(v) for k, v in targets.items()}
    print(f"domain sizes (bytes): {sizes}")
    print(f"alpha={alpha} targets (bytes): {rounded_targets}")

    rng = random.Random(seed)
    TOK_DIR.mkdir(parents=True, exist_ok=True)
    combined_path = TOK_DIR / f"_combined_{VOCAB_SIZE}_{VARIANT}.txt"
    with combined_path.open("w", encoding="utf-8") as out_f:
        for name, paths in DOMAIN_SOURCES.items():
            size = sizes[name]
            target = targets[name]
            if size == 0:
                continue
            if target >= size:
                keep_prob, repeat = 1.0, max(1, round(target / size))
            else:
                keep_prob, repeat = target / size, 1
            print(f"  {name}: size={size:,} target={round(target):,} keep_prob={keep_prob:.3f} repeat={repeat}")
            _stream_domain(paths, keep_prob, repeat, out_f, rng)
    return combined_path


def train_production_tokenizer(alpha: float = ALPHA) -> Path:
    combined_path = build_combined_corpus(alpha)
    model_prefix = _model_prefix(VOCAB_SIZE, VARIANT)
    spm.SentencePieceTrainer.train(
        input=str(combined_path),
        model_prefix=str(model_prefix),
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0,
        bos_id=1, eos_id=2, pad_id=0, unk_id=3,
        # Same cap tokenizer.py's own production training already uses, for
        # the identical reason (bound total training work regardless of how
        # big the combined file is).
        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
        # Explicit lower thread count (SentencePiece's default used 16 in
        # the small-scale run) -- deliberately less aggressive given the
        # concurrent CPU load from another process right now.
        num_threads=4,
    )
    combined_path.unlink()
    return model_prefix.with_suffix(".model")


if __name__ == "__main__":
    import resource
    path = train_production_tokenizer()
    print(f"saved: {path}")
    print(f"peak_rss_mb: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f}")
