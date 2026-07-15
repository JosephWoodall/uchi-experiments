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
from graph import add_model_prediction_edges, build_ast_fact_edges, build_graph
from grounding import build_ngram_index, build_symbol_table
from inference import calibrate_thresholds, generate_with_grounding
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
DEFAULT_RUNS = {
    ("code", "hybrid"): "code_base_m_rwkv",
    ("code", "dense"): "code_base_m",
    ("rj", "hybrid"): "rj_base_m",  # note: this checkpoint is hybrid despite the
    # plain-looking name -- a relic of an earlier naming-collision bug (fixed for
    # all runs since), see tasks/ducky.md
    ("rj", "dense"): "rj_base_m_seed1",
}


class Ducky:
    def __init__(self, domain: str = "code", backbone: str = "hybrid", run_name: str = None,
                 max_new_tokens: int = 60):
        if domain not in ("code", "rj"):
            raise ValueError(f"domain must be 'code' or 'rj', got {domain!r}")
        if backbone not in ("hybrid", "dense"):
            raise ValueError(f"backbone must be 'hybrid' or 'dense', got {backbone!r}")
        self.domain = domain
        self.backbone = backbone
        self.max_new_tokens = max_new_tokens
        run_name = run_name or DEFAULT_RUNS[(domain, backbone)]
        run_dir = ROOT / "runs" / run_name

        self.tok = Tokenizer()
        cfg_dict = json.loads((run_dir / "config.json").read_text())
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
        self.model.load_state_dict(torch.load(run_dir / "model_best.pt", map_location="cpu"))
        self.model.eval()

        code_source = (ROOT / "data" / "code" / "corpus.txt").read_text()
        rj_ids = torch.load(ROOT / "data" / "cache" / "rj.pt")
        code_ids = torch.load(ROOT / "data" / "cache" / "code.pt")
        self.graph = build_graph(self.tok, code_source, rj_ids, code_ids)
        domain_ids = code_ids if domain == "code" else rj_ids
        add_model_prediction_edges(self.graph, self.model, domain_ids, confidence_threshold=0.95, n_samples=300)

        self.symbol_table = build_symbol_table(code_source) if domain == "code" else None
        self.ngram_index = build_ngram_index(domain_ids, n=4)

        _, val_ids = load_lm_corpus(domain, self.tok)
        self.fast_t, self.abstain_t, self.slow_abstain_t = calibrate_thresholds(
            self.model, self.graph, val_ids, cfg_dict["block_size"]
        )

    def ask(self, prompt: str, max_new_tokens: int = None) -> str:
        """Always returns a string -- an empty one on immediate abstention,
        never an exception. Also returns the grounding metadata (syntax
        validity, self-critique, identifier grounding) as a side channel
        via self.last_result, for callers who want more than just the text.
        """
        result = generate_with_grounding(
            self.model, self.tok, self.graph, prompt,
            max_new_tokens or self.max_new_tokens, domain=self.domain,
            fast_threshold=self.fast_t, abstain_threshold=self.abstain_t,
            slow_abstain_threshold=self.slow_abstain_t,
            symbol_table=self.symbol_table, ngram_index=self.ngram_index,
        )
        self.last_result = result
        return result["generated_text"] if result["n_tokens_generated"] > 0 else ""

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
