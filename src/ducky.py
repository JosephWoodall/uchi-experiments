"""Ducky's public SDK. Was styled after uchi's non-negotiable API shape
(uchi/README.md: "learn() always accepts a string; ask() always returns
one... knowledge compounds across instances with zero glue code") --
that's no longer honest to keep claiming (see below), so this now just
documents what Ducky actually does: a toy-scale next-token predictor,
pure neural, nothing else mixed in. See tasks/ducky.md for the full
history of what was tried and removed.

    from ducky import Ducky
    d = Ducky()
    answer = d.ask("ROMEO:")   # always returns a string, never raises

Deliberately single-domain and single-backbone right now (2026-07-22,
user's explicit call): "nail text generation from one corpus first, add
domains later" -- rj (Romeo & Juliet) is the one corpus in scope, RWKV-
hybrid (the established best backbone, tasks/ducky.md Phase Z/Y) is the
only backbone. No domain= or backbone= argument exists to select
otherwise; when code comes back into scope this file's own history
(DEFAULT_RUNS, code_source/symbol_table/call_graph wiring, use_retrieval)
is the reference for how to reintroduce it properly.

No learn() anymore, and this is a real capability loss worth stating
plainly, not glossing over: learn() only ever worked by adding edges to
the TokenGraph (self.graph.add_edge(...)) -- "no retraining, graph update
instead" WAS the mechanism, not a description of one option among several.
Removing graph-blending (2026-07-22, user's explicit call: "that's
causing Ducky to give bad results") removed the only thing learn() had
to act on. Ducky currently has no way to incorporate new information
short of retraining -- if that capability matters again, it needs a new
mechanism designed on its own terms, not a graph half-kept-around just to
give learn() something to do.

No separate response()/generate() method: ask() already always returns the
answer string, so a third method would just duplicate it.
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from data import load_lm_corpus
from grounding import build_ngram_index
from inference import generate_with_grounding, generate_with_resampling
from mcts_lite import mcts_generate
from repair_loop import generate_with_repair
from session_history import SessionHistory
from model import GPTConfig, TinyGPT
from tokenizer import Tokenizer
from train import SIZES

ROOT = Path(__file__).resolve().parent.parent

# Single default checkpoint: the real scale-up (tasks/ducky.md Phase AI).
# 17.49M params (RWKV-hybrid, vocab=32768, embedding_rank=80), Chinchilla-
# matched to the real combined token budget across four weighted pools:
# "literary" (rj + gutenberg_corpus.txt, ~300M tokens -- deliberately
# excluding chat_corpus.txt, an explicit quality decision from Phase AG),
# "conversation" (the small hand-curated set, ~9K tokens), "code_core"/
# "code_breadth" (this project's existing stdlib/site-packages corpora,
# ~2.6M/~41.6M tokens) -- 344.5M tokens total, weighted 40% code / 20%
# conversation / 40% literary. Tokenizer (spm_ducky_scale_32768.model)
# built specifically on this combined mix with stratified resampling
# (same discipline as Phase O's tokenizer-fairness work): rj+conversation
# repeated 50x in the tokenizer's own training input, since they're only
# ~185KB against gutenberg+code's ~1.48GB (~8000:1) and would otherwise be
# too rare for character names to earn dedicated BPE merges even while
# physically present in the training data. best_val=4.1542, early-stopped
# at step 21,500 of a 90,000-step ceiling (~2.4 hours on GPU). Supersedes
# every toy-scale rj/conversation(/code) checkpoint before it -- this is
# the first checkpoint built at genuinely real (not toy) scale.
DEFAULT_RUN = "rj_base_chinchilla_scaleup_rwkv_rank80_tokspm_ducky_scale_32768_scaleup_lrplateau_nanogpt_seed57"


class EnsembleModel:
    """Presents the identical model(idx) -> (logits, extra_logits, aux_loss,
    new_states) call contract as TinyGPT, but internally averages softmax
    PROBABILITIES across N member models first (eval_ensemble.py's already-
    validated approach -- probability-averaging, not logit-averaging, is
    the correct way to combine independently-calibrated models). Every
    existing caller (predict_next, self_critique_score, and therefore
    generate_with_grounding/mcts_generate/generate_with_repair, which all
    just pass `model` through) works completely unchanged -- they only
    ever call model(idx) and read model.cfg.block_size, never a
    TinyGPT-specific attribute beyond that.

    extra_logits/new_states are not supported (returned as None) -- Ducky's
    ask() path never uses mtp's extra heads or carries chunked RWKV state
    across calls, so this doesn't lose anything the SDK actually exercises.
    """

    def __init__(self, models: list):
        assert len(models) >= 2, "EnsembleModel needs at least 2 member models"
        self.models = models
        self.cfg = models[0].cfg

    def eval(self):
        for m in self.models:
            m.eval()

    def __call__(self, idx, rwkv_states=None):
        probs = []
        for m in self.models:
            logits, _, _, _ = m(idx, rwkv_states)
            probs.append(F.softmax(logits, dim=-1))
        avg_probs = torch.stack(probs).mean(dim=0)
        pseudo_logits = torch.log(avg_probs + 1e-12)  # so downstream softmax reproduces avg_probs exactly
        return pseudo_logits, None, torch.tensor(0.0), None


class Ducky:
    @staticmethod
    def _load_single_model(run_name: str):
        """Returns (model, cfg_dict, tok, checkpoint_path) for one checkpoint
        -- factored out so __init__ can call this once (single-model, the
        original path, unchanged) or N times (ensemble_run_names, wrapped
        in EnsembleModel below).
        """
        run_dir = ROOT / "runs" / run_name
        checkpoint_path = run_dir / "model_best.pt"
        cfg_dict = json.loads((run_dir / "config.json").read_text())
        # vocab_size is only recorded in config.json from the 8192-vocab
        # generation onward. Checkpoints predating that field (rj_base_m,
        # rj_base_m_seed1 among DEFAULT_RUNS) were trained under vocab=1024
        # against the ORIGINAL unversioned data/tokenizer/spm.model -- not
        # today's spm_1024.model, which was itself retrained later (2026-
        # 07-15 19:50) against a different corpus snapshot and has
        # different token-ID meanings despite the same vocab size. Verified
        # directly: loading rj_base_m against spm_1024.model gives val loss
        # ~8.6 nats (worse than random); against the real original spm.model
        # it reproduces the recorded 4.3746 (see tasks/ducky.md). Route
        # these checkpoints to the preserved original file explicitly,
        # rather than guessing vocab_size=1024 and hoping today's
        # spm_1024.model still matches.
        if cfg_dict.get("tokenizer_model_path"):
            # A dedicated single-corpus tokenizer (e.g. spm_rj_only_2000.model,
            # trained only on romeo_and_juliet.txt) bypasses the shared
            # vocab_size/variant family entirely -- must be loaded by the
            # exact same file path the checkpoint was trained under, same
            # reasoning as the legacy spm.model case below.
            tok = Tokenizer(model_path=cfg_dict["tokenizer_model_path"])
        elif "vocab_size" in cfg_dict:
            # tokenizer_variant distinguishes same-vocab-size tokenizers trained on
            # different recipes (e.g. "balanced" -- spm_32768_balanced.model -- vs.
            # the default naive-concat spm_32768.model); missing this would silently
            # load the wrong one for any checkpoint trained with --tokenizer-variant,
            # the same class of bug already fixed once this session for the legacy
            # pre-versioning case below.
            tok = Tokenizer(vocab_size=cfg_dict["vocab_size"], variant=cfg_dict.get("tokenizer_variant", ""))
        else:
            tok = Tokenizer(model_path=ROOT / "data" / "tokenizer" / "spm.model")
        size_cfg = SIZES[cfg_dict["size"]]
        cfg = GPTConfig(
            vocab_size=tok.vocab_size,
            block_size=cfg_dict["block_size"],
            use_rwkv_hybrid=cfg_dict.get("rwkv_hybrid", False),
            attention_layers=tuple(cfg_dict.get("attention_layers", [])),
            use_bitlinear=cfg_dict.get("use_bitlinear", False),
            embedding_rank=cfg_dict.get("embedding_rank", 0),
            **size_cfg,
        )
        model = TinyGPT(cfg)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()
        return model, cfg_dict, tok, checkpoint_path

    def __init__(self, run_name: str = None, ensemble_run_names: list = None,
                 max_new_tokens: int = 300, track_history: bool = False):
        # Hardcoded, not a parameter: single-corpus focus (rj) for now.
        # generate_with_grounding/_resampling/mcts_generate/generate_with_repair
        # still take a domain string internally (it decides whether code-only
        # checks like syntax validity run) -- "rj" naturally skips those,
        # which is exactly correct here, and those functions stay generic
        # for when code comes back.
        self.domain = "rj"
        self.max_new_tokens = max_new_tokens
        # RAM-only, cross-call (not cross-process) memory -- see
        # session_history.py. Opt-in: most callers use Ducky() for
        # independent one-off asks, where folding prior Q&A into every new
        # prompt would just be noise.
        self.history = SessionHistory() if track_history else None

        if ensemble_run_names:
            # eval_ensemble.py's validated approach (probability-averaging beats
            # the best single seed on both domains tested) -- wrapped so every
            # downstream mechanism (predict_next, mcts, repair) works
            # completely unchanged against EnsembleModel's identical call
            # contract. Assumes matched tokenizer/config across members
            # (asserted below, not just hoped).
            assert len(ensemble_run_names) >= 2, "ensemble_run_names needs at least 2 checkpoints"
            loaded = [self._load_single_model(rn) for rn in ensemble_run_names]
            members, cfg_dicts, toks, _ = zip(*loaded)
            for cd in cfg_dicts[1:]:
                assert cd["vocab_size"] == cfg_dicts[0]["vocab_size"] and cd.get("tokenizer_variant", "") == cfg_dicts[0].get("tokenizer_variant", ""), \
                    "ensemble_run_names must all share the same tokenizer (vocab_size + variant)"
            self.tok = toks[0]
            self.model = EnsembleModel(list(members))
        else:
            run_name = run_name or DEFAULT_RUN
            self.model, _, self.tok, _ = self._load_single_model(run_name)

        # load_lm_corpus creates the vocab-versioned cache file if it doesn't
        # exist yet -- a raw torch.load here would assume it already does,
        # which broke the first time a checkpoint used a vocab size nothing
        # had tokenized the full corpus under yet.
        train_ids, val_ids = load_lm_corpus("rj", self.tok)
        rj_ids = torch.cat([train_ids, val_ids])
        # build_ngram_index materializes one Python tuple per token position
        # in a set -- genuinely expensive regardless of corpus size, not
        # just at some previously-untested scale: a controlled measurement
        # showed 20M tokens alone pushing peak RSS to ~3.5GB. rj is small
        # (~50K tokens) so this is a non-issue here and needs no caching;
        # flagged as a real, not-yet-fixed cost if a much bigger single
        # corpus is ever wired back in as the default.
        self.ngram_index = build_ngram_index(rj_ids, n=4)

    def ask(self, prompt: str, max_new_tokens: int = None, n_candidates: int = 1,
            temperature: float = 0.5, top_p: float = 0.5, repetition_penalty: float = 1.3,
            use_mcts: bool = False, mcts_kwargs: dict = None,
            use_repair: bool = False, max_attempts: int = 4) -> str:
        """Always returns a real generated string, never an exception --
        abstention was removed (2026-07-22): Ducky no longer refuses to
        answer regardless of confidence. Also returns the grounding
        metadata (syntax validity, self-critique, identifier grounding) as
        a side channel via self.last_result, for callers who want more
        than just the text.

        temperature=0.5/top_p=0.5 are measured defaults, not guessed ones.
        Greedy (temperature=0) over a long 300-token generation degrades
        into repetitive phrasing (no_repeat_ngram_size alone doesn't fully
        prevent it). The first fix tried, temperature=0.8/top_p=0.9, was
        real sampling but produced fluent-*sounding* output full of garbled
        non-words ("shadn", "wcup", "penk") -- this checkpoint's per-token
        confidence isn't peaked enough for its 2nd/3rd-choice subword
        pieces to still compose into real words. A direct side-by-side
        swept down to temperature=0.3-0.5/top_p=0.5-0.7: legibility came
        back sharply (real words, recognizable character names) while
        keeping real diversity over greedy. 0.5/0.5 is the tested point
        kept as default; set temperature=0 for the old fully-deterministic
        behavior.

        repetition_penalty=1.3 (measured default, CTRL-style -- Keskar et
        al. 2019, arXiv:1909.05858): fixes a real, distinct problem
        temperature/top_p alone didn't touch -- generation degrading into
        repetitive phrasing ("I'll not, I's my lady? I'st thou not...")
        over a long span, which is different from no_repeat_ngram_size's
        exact-n-gram blocking (this is the same handful of tokens recurring
        across many non-identical short phrases, not literal repeats).
        Verified side-by-side: 1.3-1.5 both broke the loop (real character
        variety, stage directions instead of cycling); 1.3 kept slightly
        less collateral word garbling than 1.5 (the penalty discouraging
        an already-used subword piece can occasionally corrupt an
        unrelated multi-piece word). Set to 1.0 to disable.

        n_candidates=1 (default) is the original single deterministic-path
        behavior, unchanged. n_candidates>1 switches to reject-and-resample:
        generate that many independent candidates (temperature>0, so they
        can actually differ) and keep the one with the best self-critique
        (+ syntax-validity bonus for code) score, rather than only reporting
        those signals after the fact on a single generation.

        use_mcts=True switches to value-guided search over chunk-level
        generation branches (mcts_lite.mcts_generate) instead -- a
        different, stronger mechanism than best-of-N resampling: it can
        abandon a branch partway through instead of only ever comparing
        finished candidates. Takes precedence over n_candidates when set.

        use_repair=True switches to a sequential, feedback-informed retry
        loop (repair_loop.generate_with_repair): unlike resampling/MCTS,
        each retry sees the real failure (syntax error) from the previous
        attempt spliced into its prompt. Takes precedence over both
        use_mcts and n_candidates when set.

        If track_history was set on __init__, prior asks in this session
        are folded into the prompt as a plain-text comment (verbatim
        excerpts, never paraphrased -- see session_history.py), and this
        call is recorded into that history afterward regardless of which
        generation mode was used.
        """
        effective_prompt = prompt
        if self.history is not None:
            context = self.history.context_string()
            if context:
                effective_prompt = f"# {context}\n{prompt}"

        if use_repair:
            result = generate_with_repair(
                self.model, self.tok, effective_prompt,
                max_new_tokens or self.max_new_tokens, domain=self.domain,
                temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty,
                ngram_index=self.ngram_index,
                max_attempts=max_attempts,
            )
        elif use_mcts:
            result = mcts_generate(
                self.model, self.tok, effective_prompt,
                max_new_tokens or self.max_new_tokens, domain=self.domain,
                ngram_index=self.ngram_index,
                **{"temperature": temperature, "top_p": top_p, "repetition_penalty": repetition_penalty,
                   **(mcts_kwargs or {})},
            )
        elif n_candidates > 1:
            result = generate_with_resampling(
                self.model, self.tok, effective_prompt,
                max_new_tokens or self.max_new_tokens, domain=self.domain,
                ngram_index=self.ngram_index,
                n_candidates=n_candidates, temperature=temperature, top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        else:
            result = generate_with_grounding(
                self.model, self.tok, effective_prompt,
                max_new_tokens or self.max_new_tokens, domain=self.domain,
                temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty,
                ngram_index=self.ngram_index,
            )
        self.last_result = result
        response = result["generated_text"] if result["n_tokens_generated"] > 0 else ""
        if self.history is not None:
            self.history.record(prompt, response)
            self.history.maybe_compact()
        return response
