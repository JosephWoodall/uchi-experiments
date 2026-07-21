"""Three small candidate tokenizers, trained to compare against the current
production tokenizer's text-dominated fertility (see tasks/ducky.md's
tokenizer-fairness section): does resampling the training mix toward
equal per-domain representation, and/or switching BPE->Unigram, actually
close the gap `eval_tokenizer_fairness.py` measured?

Deliberately small comparison scale (vocab_size=8192, a few MB of input
total per candidate) -- this is a fairness/model-type comparison, not a
final production vocab-size decision, and the machine is under real CPU/
GPU load from another process right now. Trains into data/tokenizer/
candidates/, never touching the production spm_32768.model.

- naive_bpe: today's approach at small scale -- concatenate text/code/
  terminal training slices proportional to their natural size (~8:1
  text:code, terminal barely represented at all).
- resampled_bpe: same three domains, resampled toward roughly equal
  representation before concatenation -- the standard fix for BPE merge
  domination (Parity-Aware BPE, arXiv:2508.04796; "Equity with
  Efficiency," arXiv:2606.15044).
- resampled_unigram: identical resampled corpus, model_type="unigram"
  instead of "bpe" -- isolates the model-type variable from the
  resampling variable (evidence on Unigram vs. BPE is genuinely mixed
  across domains, arxiv.org/pdf/2607.05691, so this has to be measured
  here, not assumed from the literature).
"""
import math
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
CAND_DIR = ROOT / "data" / "tokenizer" / "candidates"
VOCAB_SIZE = 8192

# Training slices -- HEAD of each file, distinct from eval_tokenizer_fairness.py's
# TAIL slices, so candidates are never trained on the exact text they're scored on.
TRAIN_CHARS = {"text": 2_000_000, "code": 2_000_000, "terminal": 500_000}
RESAMPLE_TARGET_CHARS = 1_500_000  # roughly equal per-domain contribution


def _head_slice(path: Path, n_chars: int) -> str:
    with path.open("rb") as f:
        raw = f.read(n_chars * 4)  # *4 headroom for multi-byte utf-8
    return raw.decode("utf-8", errors="ignore")[:n_chars]


def _resample_to(text: str, target_chars: int) -> str:
    """Truncate if longer than target, repeat (with a separator) if
    shorter -- the simplest form of the temperature/stratified resampling
    used in the literature above, sized for a small comparison corpus
    rather than tuning a continuous temperature parameter.
    """
    if len(text) >= target_chars:
        return text[:target_chars]
    reps = math.ceil(target_chars / max(len(text), 1))
    return (text * reps)[:target_chars]


def raw_domain_slices() -> dict:
    return {
        "text": _head_slice(ROOT / "data" / "text" / "gutenberg_corpus.txt", TRAIN_CHARS["text"]),
        "code": _head_slice(ROOT / "data" / "code" / "corpus_core.txt", TRAIN_CHARS["code"]),
        "terminal": _head_slice(ROOT / "data" / "terminal" / "nl2bash_corpus.txt", TRAIN_CHARS["terminal"]),
    }


def train_candidate(name: str, combined_text: str, model_type: str) -> Path:
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    corpus_path = CAND_DIR / f"_{name}_train.txt"
    corpus_path.write_text(combined_text)
    model_prefix = CAND_DIR / name
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=VOCAB_SIZE,
        model_type=model_type,
        character_coverage=1.0,
        bos_id=1, eos_id=2, pad_id=0, unk_id=3,
    )
    corpus_path.unlink()
    return model_prefix.with_suffix(".model")


def _alpha_resample(domains: dict, alpha: float, total_chars: int) -> str:
    """Temperature/alpha-weighted resampling: weight_domain = size_domain**alpha.
    alpha=1.0 -> proportional to natural size (today's naive approach);
    alpha=0.0 -> every domain equal regardless of size. The standard fix
    for BPE merge domination at size ratios where forcing full equality
    would mean absurd repetition of a tiny domain (Parity-Aware BPE,
    arXiv:2508.04796) -- picked empirically here rather than assumed,
    reusing `_resample_to`'s truncate-or-repeat mechanics per domain.
    Targets a fixed total_chars so different alphas are comparable at
    matched overall corpus size.
    """
    sizes = {name: len(text) for name, text in domains.items()}
    weights = {name: size**alpha for name, size in sizes.items()}
    total_weight = sum(weights.values())
    targets = {name: round(total_chars * w / total_weight) for name, w in weights.items()}
    print(f"  alpha={alpha}: targets={targets}")
    return "".join(_resample_to(domains[name], targets[name]) for name in domains)


def sweep_alpha(alphas=(1.0, 0.5, 0.3)) -> dict:
    """alpha=1.0 reproduces the naive_bpe candidate's proportions (already
    scored); 0.5 and 0.3 are genuinely new points, trained and left for
    eval_tokenizer_fairness.py to score -- picking alpha by measurement,
    not by picking a textbook default.
    """
    domains = raw_domain_slices()
    total_chars = sum(len(t) for t in domains.values())
    results = {}
    for alpha in alphas:
        text = _alpha_resample(domains, alpha, total_chars)
        # NOT f"alpha_{alpha}" -- Path.with_suffix() in train_candidate() treats
        # the LAST "." as an existing suffix and replaces it, so "alpha_0.5" and
        # "alpha_0.3" would both collapse to "alpha_0.model" (silent overwrite,
        # caught by inspecting the actual output paths after a first run).
        name = f"alpha_{str(alpha).replace('.', '_')}"
        results[name] = train_candidate(name, text, "bpe")
    return results


def main():
    domains = raw_domain_slices()
    print({name: len(text) for name, text in domains.items()})

    naive_text = domains["text"] + domains["code"] + domains["terminal"]
    resampled_text = "".join(_resample_to(t, RESAMPLE_TARGET_CHARS) for t in domains.values())

    results = {
        "naive_bpe": train_candidate("naive_bpe", naive_text, "bpe"),
        "resampled_bpe": train_candidate("resampled_bpe", resampled_text, "bpe"),
        "resampled_unigram": train_candidate("resampled_unigram", resampled_text, "unigram"),
    }
    for name, path in results.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "sweep-alpha":
        results = sweep_alpha()
        for name, path in results.items():
            print(f"{name}: {path}")
    else:
        main()
