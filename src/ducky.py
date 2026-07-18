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
# code/* and text/* point at the latest-generation checkpoints (vocab=32768,
# xl depth, rank-64 embedding, ~30-100x expanded corpus per domain --
# stdlib+curated site-packages for code, rj+1,255 Gutenberg texts+chat for
# text). code: best val 3.4822 (was 5.1643 under the old 8192-vocab/smaller
# corpus generation -- lower despite the harder, 4x bigger vocab, meaning
# the data expansion more than offset it). text: best val 5.1040, and the
# first checkpoint this whole project to produce genuinely coherent
# English samples, not word-salad. rj/* still point at the old (vocab=1024)
# generation, kept only for backward-compatible comparison -- "text" is
# rj's real successor domain going forward. self.tok's vocab_size is read
# per-checkpoint from config.json (falling back to 1024 for checkpoints
# that predate that field), so all three generations load correctly side
# by side.
DEFAULT_RUNS = {
    ("code", "hybrid"): "code_base_xl_rwkv_rank64",
    ("code", "dense"): "code_base_xl_rank64",
    ("text", "hybrid"): "text_base_xl_rwkv_rank64",
    # ("text", "dense") not yet trained -- only the hybrid backbone has
    # been run on the text domain so far.
    ("rj", "hybrid"): "rj_base_m",  # note: this checkpoint is hybrid despite the
    # plain-looking name -- a relic of an earlier naming-collision bug (fixed for
    # all runs since), see tasks/ducky.md
    ("rj", "dense"): "rj_base_m_seed1",
}


class Ducky:
    def __init__(self, domain: str = "code", backbone: str = "hybrid", run_name: str = None,
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
        run_name = run_name or DEFAULT_RUNS[(domain, backbone)]
        run_dir = ROOT / "runs" / run_name
        checkpoint_path = run_dir / "model_best.pt"

        cfg_dict = json.loads((run_dir / "config.json").read_text())
        # vocab_size is only recorded in config.json from the 8192-vocab
        # generation onward -- checkpoints trained before that fix predate
        # vocab versioning entirely and were all trained under vocab=1024,
        # so that's the correct fallback, not the SDK's current default.
        self.tok = Tokenizer(vocab_size=cfg_dict.get("vocab_size", 1024))
        size_cfg = SIZES[cfg_dict["size"]]
        cfg = GPTConfig(
            vocab_size=self.tok.vocab_size,
            block_size=cfg_dict["block_size"],
            use_rwkv_hybrid=cfg_dict.get("rwkv_hybrid", False),
            attention_layers=tuple(cfg_dict.get("attention_layers", [])),
            use_bitlinear=cfg_dict.get("use_bitlinear", False),
            embedding_rank=cfg_dict.get("embedding_rank", 0),
            **size_cfg,
        )
        self.model = TinyGPT(cfg)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.model.eval()

        # Graph-building (300 forward passes for model-prediction edges) and
        # threshold calibration (500+ more) are real, repeated work -- every
        # Ducky() instantiation redid them from scratch. Cached to disk,
        # keyed by run_name + the checkpoint file's own mtime, so a silently
        # retrained/overwritten checkpoint (this session hit that bug twice
        # already, see tasks/ducky.md) invalidates the cache instead of
        # serving stale graph/thresholds for new weights.
        SETUP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = SETUP_CACHE_DIR / f"{run_name}.pkl"
        checkpoint_mtime = checkpoint_path.stat().st_mtime
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
