"""Ducky's public SDK -- mirrors uchi's own non-negotiable API shape
(uchi/README.md: "learn() always accepts a string; ask() always returns
one... knowledge compounds across instances with zero glue code") as
closely as is honest for what Ducky actually is: a toy-scale next-token
predictor with a confidence-gated knowledge graph and single-model
abstention, not a full retrieval+verification system with an LLM-scale
knowledge base. See tasks/ducky.md for what's real and what's scoped down.

    from ducky import Ducky
    d = Ducky(domain="code")
    d.learn("some new code or docstring text")   # graph update, no retraining
    answer = d.ask("def ")                          # always returns a string, never raises

No separate response()/generate() method: ask() already always returns the
answer string (uchi's own contract), so a third method would just
duplicate it.
"""
import json
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F

from data import load_lm_corpus
from graph import TokenGraph, add_model_prediction_edges, build_ast_fact_edges, build_graph
from grounding import build_ngram_index, build_symbol_table
from inference import calibrate_thresholds, generate_with_grounding, generate_with_resampling
from mcts_lite import mcts_generate
from repair_loop import generate_with_repair
from session_history import SessionHistory
from synthetic_relations import build_call_graph, inject_context
from model import GPTConfig, TinyGPT
from tokenizer import Tokenizer
from train import SIZES

ROOT = Path(__file__).resolve().parent.parent
SETUP_CACHE_DIR = ROOT / "data" / "cache" / "ducky_setup"

# Ducky's current best-validated checkpoints. "hybrid" (RWKV + periodic
# attention) is Ducky's actual identity and the default -- it beat dense
# 6/6 seeds across both corpora (tasks/ducky.md). "dense" is the plain
# attention-only baseline it's compared against, exposed here so you can
# run the same SDK against either backbone and see the difference directly,
# not just read about it in the results table.
#
# code/hybrid and rj/hybrid now point at the best-validated EXPERIMENTAL
# SETUP (architecture + training recipe), not the biggest/most-trained
# checkpoint -- a deliberate choice (tasks/todo.md Phase X): production-
# scale retrains are explicitly out of scope until the architecture is
# more settled, so "best" here means "best combination of levers proven
# at toy scale," not "best absolute quality." Both combine RWKV-hybrid
# (the established best backbone) with --nanogpt-recipe --lr-schedule
# plateau (betas/weight-decay-scope/scaled-init + adaptive LR, tasks/
# todo.md Phase X -- the one clean, reproducible free win found this
# session, bigger than most single architecture ablations). rj/hybrid
# (rj_base_m_rwkv_lrplateau_nanogpt_seed57): best_val=3.5944, cleanly
# early-stopped at step 600/1200 -- beats every prior rj reference point,
# including the earlier hybrid-only result (4.351, no recipe fix) and
# today's dense+recipe-only result (3.66), confirming the two levers
# stack. code/hybrid (code_base_m_rwkv_rank32_tokbalanced_lrplateau_
# nanogpt_seed57): best_val=4.5421 at step 4250/5000 -- extended from an
# initial 2000-step run that stopped mid-descent at 5.4292; ran the full
# 5000-step budget without plateau ever triggering a decay, so may still
# have room left, but this is a real, substantially-converged number now.
#
# The much bigger, fully-converged xl-scale checkpoints from the pre-
# recipe-fix generation (code_base_xl_rwkv_rank64: val 3.4822 on the real
# ~43M-token corpus; text_base_xl_rwkv_rank64: val 5.1040, vocab=32768)
# are NOT currently wired as defaults for that reason -- they're bigger
# and better-trained in absolute terms, but were never retrained with the
# validated recipe, and re-running them at that scale is exactly the
# production commitment this phase is deliberately deferring. Still on
# disk, still loadable via run_name=... below, not deleted.
DEFAULT_RUNS = {
    ("code", "hybrid"): "code_base_m_rwkv_rank32_tokbalanced_lrplateau_nanogpt_seed57",
    ("code", "dense"): "code_base_xl_rank64",
    ("text", "hybrid"): "text_base_xl_rwkv_rank64",
    # ("text", "dense") not yet trained -- only the hybrid backbone has
    # been run on the text domain so far.
    ("rj", "hybrid"): "rj_base_m_rwkv_lrplateau_nanogpt_seed57",
    ("rj", "dense"): "rj_base_m_seed1",
}


class EnsembleModel:
    """Presents the identical model(idx) -> (logits, extra_logits, aux_loss,
    new_states) call contract as TinyGPT, but internally averages softmax
    PROBABILITIES across N member models first (eval_ensemble.py's already-
    validated approach -- probability-averaging, not logit-averaging, is
    the correct way to combine independently-calibrated models). Every
    existing caller (predict_next, self_critique_score, calibrate_thresholds,
    add_model_prediction_edges, and therefore generate_with_grounding/
    mcts_generate/generate_with_repair, which all just pass `model` through)
    works completely unchanged -- they only ever call model(idx) and read
    model.cfg.block_size, never a TinyGPT-specific attribute beyond that.

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
        if "vocab_size" in cfg_dict:
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

    def __init__(self, domain: str = "code", backbone: str = "hybrid", run_name: str = None,
                 ensemble_run_names: list = None,
                 max_new_tokens: int = 60, use_cache: bool = True, track_history: bool = False):
        if domain not in ("code", "rj", "text"):
            raise ValueError(f"domain must be 'code', 'rj', or 'text', got {domain!r}")
        if backbone not in ("hybrid", "dense"):
            raise ValueError(f"backbone must be 'hybrid' or 'dense', got {backbone!r}")
        self.domain = domain
        self.backbone = backbone
        self.max_new_tokens = max_new_tokens
        # RAM-only, cross-call (not cross-process) memory -- see
        # session_history.py. Opt-in: most callers use Ducky() for
        # independent one-off asks, where folding prior Q&A into every new
        # prompt would just be noise.
        self.history = SessionHistory() if track_history else None

        if ensemble_run_names:
            # eval_ensemble.py's validated approach (probability-averaging beats
            # the best single seed on both domains tested) -- wrapped so every
            # downstream mechanism (graph, thresholds, predict_next, mcts,
            # repair) works completely unchanged against EnsembleModel's
            # identical call contract. Assumes matched tokenizer/config across
            # members (asserted below, not just hoped).
            assert len(ensemble_run_names) >= 2, "ensemble_run_names needs at least 2 checkpoints"
            loaded = [self._load_single_model(rn) for rn in ensemble_run_names]
            members, cfg_dicts, toks, checkpoint_paths = zip(*loaded)
            for cd in cfg_dicts[1:]:
                assert cd["vocab_size"] == cfg_dicts[0]["vocab_size"] and cd.get("tokenizer_variant", "") == cfg_dicts[0].get("tokenizer_variant", ""), \
                    "ensemble_run_names must all share the same tokenizer (vocab_size + variant)"
            self.tok = toks[0]
            cfg_dict = cfg_dicts[0]
            self.model = EnsembleModel(list(members))
            run_name = "+".join(ensemble_run_names)  # for cache naming only
            checkpoint_path = checkpoint_paths[0]
            checkpoint_mtime = max(p.stat().st_mtime for p in checkpoint_paths)
        else:
            run_name = run_name or DEFAULT_RUNS[(domain, backbone)]
            self.model, cfg_dict, self.tok, checkpoint_path = self._load_single_model(run_name)
            checkpoint_mtime = checkpoint_path.stat().st_mtime

        # Graph-building (300 forward passes for model-prediction edges) and
        # threshold calibration (500+ more) are real, repeated work -- every
        # Ducky() instantiation redid them from scratch. Cached to disk,
        # keyed by run_name + the checkpoint file's own mtime, so a silently
        # retrained/overwritten checkpoint (this session hit that bug twice
        # already, see tasks/ducky.md) invalidates the cache instead of
        # serving stale graph/thresholds for new weights.
        SETUP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = SETUP_CACHE_DIR / f"{run_name}.pkl"
        cached = None
        if use_cache and cache_path.exists():
            with cache_path.open("rb") as f:
                cached = pickle.load(f)
            if cached.get("checkpoint_mtime") != checkpoint_mtime:
                cached = None  # checkpoint changed since this was cached -- rebuild

        # corpus_core.txt (stdlib only, ~11MB) for everything that does
        # ast.parse -- symbol table, call graph, AST-fact edges. The full
        # corpus.txt (~162MB incl. curated site-packages libraries) made
        # each of these take long enough to hit real timeout/resource
        # limits (measured: build_symbol_table alone was killed after
        # several minutes). Same reasoning already applied to
        # synthetic_relations.py's call graph: stdlib is cleaner, more
        # idiomatic, and better-matched to what identifier grounding and
        # AST facts are actually for, not just faster to parse. Token-level
        # signals (ngram_index, co-occurrence edges) still use the FULL
        # corpus below -- those were measured fast (12s for 39M tokens)
        # and benefit from the full breadth.
        code_source = (ROOT / "data" / "code" / "corpus_core.txt").read_text()
        # load_lm_corpus creates the vocab-versioned cache file if it doesn't
        # exist yet -- a raw torch.load here would assume it already does,
        # which broke the first time a checkpoint used a vocab size nothing
        # had tokenized the full corpus under yet.
        # Text-side ids for the graph: "text" (rj + Gutenberg) if that's the
        # active domain, else plain "rj" -- matches existing behavior for
        # code/rj domains exactly, just extends it to the new combined one.
        text_domain_name = "text" if domain == "text" else "rj"
        text_train_ids, text_val_ids = load_lm_corpus(text_domain_name, self.tok)
        code_train_ids, code_val_ids = load_lm_corpus("code", self.tok)
        rj_ids = torch.cat([text_train_ids, text_val_ids])
        code_ids = torch.cat([code_train_ids, code_val_ids])
        domain_ids = code_ids if domain == "code" else rj_ids
        self.symbol_table = build_symbol_table(code_source) if domain == "code" else None
        self.ngram_index = build_ngram_index(domain_ids, n=4)
        # Whole-identifier call graph for retrieval-injection (synthetic_relations.py)
        # -- distinct from self.graph (TokenGraph, BPE-token-level, consulted via
        # logit blending): this is used to pull real facts into the prompt as
        # text, at whole-function-name granularity.
        self.call_graph = build_call_graph(code_source) if domain == "code" else None

        if cached is not None:
            self.graph = TokenGraph()
            self.graph.edges = cached["graph_edges"]
            self.fast_t, self.abstain_t, self.slow_abstain_t = cached["thresholds"]
        else:
            self.graph = build_graph(self.tok, code_source, rj_ids, code_ids)
            add_model_prediction_edges(self.graph, self.model, domain_ids, confidence_threshold=0.95, n_samples=300)

            _, val_ids = load_lm_corpus(domain, self.tok)
            self.fast_t, self.abstain_t, self.slow_abstain_t = calibrate_thresholds(
                self.model, self.graph, val_ids, cfg_dict["block_size"]
            )
            if use_cache:
                with cache_path.open("wb") as f:
                    pickle.dump({
                        "checkpoint_mtime": checkpoint_mtime,
                        "graph_edges": self.graph.edges,
                        "thresholds": (self.fast_t, self.abstain_t, self.slow_abstain_t),
                    }, f)

    def ask(self, prompt: str, max_new_tokens: int = None, n_candidates: int = 1,
            temperature: float = 0.8, use_mcts: bool = False, mcts_kwargs: dict = None,
            use_repair: bool = False, max_attempts: int = 4,
            use_retrieval: bool = False, max_facts: int = 3) -> str:
        """Always returns a string -- an empty one on immediate abstention,
        never an exception. Also returns the grounding metadata (syntax
        validity, self-critique, identifier grounding) as a side channel
        via self.last_result, for callers who want more than just the text.

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

        use_retrieval=True (code domain only) surfaces real call-graph
        facts relevant to whatever's already mentioned in the prompt as
        literal text prepended to it (synthetic_relations.inject_context)
        -- not blended into logits like self.graph is. Composable with
        every other option above; this only changes what the prompt looks
        like before generation starts.
        """
        effective_prompt = prompt
        if self.history is not None:
            context = self.history.context_string()
            if context:
                effective_prompt = f"# {context}\n{prompt}"
        if use_retrieval and self.call_graph is not None:
            effective_prompt = inject_context(effective_prompt, self.call_graph, max_facts=max_facts)

        if use_repair:
            result = generate_with_repair(
                self.model, self.tok, self.graph, effective_prompt,
                max_new_tokens or self.max_new_tokens, domain=self.domain,
                fast_threshold=self.fast_t, abstain_threshold=self.abstain_t,
                slow_abstain_threshold=self.slow_abstain_t,
                symbol_table=self.symbol_table, ngram_index=self.ngram_index,
                max_attempts=max_attempts,
            )
        elif use_mcts:
            result = mcts_generate(
                self.model, self.tok, self.graph, effective_prompt,
                max_new_tokens or self.max_new_tokens, domain=self.domain,
                fast_threshold=self.fast_t, abstain_threshold=self.abstain_t,
                slow_abstain_threshold=self.slow_abstain_t,
                symbol_table=self.symbol_table, ngram_index=self.ngram_index,
                **(mcts_kwargs or {}),
            )
        elif n_candidates > 1:
            result = generate_with_resampling(
                self.model, self.tok, self.graph, effective_prompt,
                max_new_tokens or self.max_new_tokens, domain=self.domain,
                fast_threshold=self.fast_t, abstain_threshold=self.abstain_t,
                slow_abstain_threshold=self.slow_abstain_t,
                symbol_table=self.symbol_table, ngram_index=self.ngram_index,
                n_candidates=n_candidates, temperature=temperature,
            )
        else:
            result = generate_with_grounding(
                self.model, self.tok, self.graph, effective_prompt,
                max_new_tokens or self.max_new_tokens, domain=self.domain,
                fast_threshold=self.fast_t, abstain_threshold=self.abstain_t,
                slow_abstain_threshold=self.slow_abstain_t,
                symbol_table=self.symbol_table, ngram_index=self.ngram_index,
            )
        self.last_result = result
        response = result["generated_text"] if result["n_tokens_generated"] > 0 else ""
        if self.history is not None:
            self.history.record(prompt, response)
            self.history.maybe_compact()
        return response

    def learn(self, text: str) -> "Ducky":
        """Accepts a string, updates the graph immediately -- no
        retraining, no restart needed. Chainable, matching uchi's
        ingest().ingest() pattern. Confidence is fixed (0.75, "user_taught")
        rather than frequency-derived like corpus statistics: a single
        short snippet has no repetition to earn confidence from the usual
        formula, but it's still a direct, explicit teaching signal --
        trusted less than a verified AST fact (0.99), more than an
        unproven single occurrence would otherwise get.
        """
        ids = self.tok.encode(text)
        for src, tgt in zip(ids[:-1], ids[1:]):
            self.graph.add_edge(src, tgt, relation_type="co_occurrence", weight=0.5,
                                 confidence=0.75, provenance="user_taught")
        if self.domain == "code":
            try:
                for src, tgt, meta in build_ast_fact_edges(self.tok, text):
                    self.graph.add_edge(src, tgt, **meta)
            except SyntaxError:
                pass  # not valid code -- the co-occurrence edges above still applied
        return self
