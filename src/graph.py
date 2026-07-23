"""Plain-Python directed graph over BPE token ids -- no NetworkX/FAISS/Neo4j,
no new dependency, matching this session's small-scale ethos (see
tasks/swarm.md and the conversation record for the full cut list from the
original spec). Two edge sources, not three: AST-grounded facts (code only,
deterministic, reuses the ast-based extraction already proven in
extract_code_corpus.py) and consecutive-token co-occurrence (domain-agnostic,
works on both rj and code). Text (rj) fact extraction via an IE model is
skipped entirely -- dialogue isn't factual prose, so subject-relation-object
extraction has no sensible target there, and no IE model is installed.

Semantic facts always win over co-occurrence on a conflicting edge (per
swarm.md's own stated principle) -- add_edge keeps whichever has higher
confidence, so a 0.99-confidence AST fact naturally overrides a
lower-confidence co-occurrence edge for the same (src, tgt) pair.
"""
import ast
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TokenGraph:
    def __init__(self):
        self.edges = defaultdict(dict)  # source_token -> {target_token: metadata}

    def add_edge(self, src, tgt, **metadata):
        existing = self.edges[src].get(tgt)
        if existing is None or metadata.get("confidence", 0) > existing.get("confidence", 0):
            self.edges[src][tgt] = metadata

    def successors(self, src):
        return self.edges.get(src, {})

    def num_edges(self):
        return sum(len(v) for v in self.edges.values())

    def edges_by_provenance(self, provenance):
        return [(s, t) for s in self.edges for t, m in self.edges[s].items() if m["provenance"] == provenance]


def _token_at_char(tok, line: str, char_pos: int):
    """Token whose decoded prefix first reaches char_pos in `line`, tokenized
    in the line's real context -- not an isolated substring. Fixes the bug
    above: encoding a bare word like "self" alone tokenizes differently
    (different leading-space/merge behavior) than "self" as it actually
    appears mid-line, which is what produced the garbage edges.
    """
    ids = tok.encode(line)
    for i, tid in enumerate(ids):
        if len(tok.decode(ids[: i + 1])) >= char_pos:
            return tid
    return ids[-1] if ids else None


def build_ast_fact_edges(tok, code_source: str) -> list:
    """def/attribute/import patterns -> (src_token, tgt_token, metadata),
    located via AST line/column offsets so tokens are read in their real
    surrounding context.

    Quality filter: with a 1024-token vocab on a 5-module corpus, only
    common identifiers get merged into a whole BPE piece -- rare ones
    (most function/variable names) decompose to near-character-level, and
    "last token before the boundary" is then a meaningless 1-char fragment
    (observed first pass: 'self' -> 'break', 'import' -> 'ce'). Skip any
    token whose decoded text is under 2 characters -- keeps the real
    signal ('import' -> 'Fraction') and drops the fragment noise.
    """
    tree = ast.parse(code_source)
    lines = code_source.splitlines()
    facts = []

    def line_of(lineno):
        return lines[lineno - 1] if 0 < lineno <= len(lines) else ""

    def is_real_unit(token_id):
        return token_id is not None and len(tok.decode([token_id]).strip()) >= 2

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            line = line_of(node.lineno)
            name_start = line.find(node.name, line.find("def "))
            if name_start < 0:
                continue
            t_def = _token_at_char(tok, line, name_start)
            t_name = _token_at_char(tok, line, name_start + len(node.name))
            if is_real_unit(t_def) and is_real_unit(t_name):
                facts.append((t_def, t_name, dict(relation_type="semantic_fact", weight=1.0,
                                                    confidence=0.99, provenance="ast_extraction")))
        elif isinstance(node, ast.Attribute):
            if not hasattr(node.value, "end_col_offset") or node.value.end_col_offset is None:
                continue
            line = line_of(node.value.end_lineno)
            t_val = _token_at_char(tok, line, node.value.end_col_offset)
            t_attr = _token_at_char(tok, line_of(node.end_lineno), node.end_col_offset)
            if is_real_unit(t_val) and is_real_unit(t_attr):
                facts.append((t_val, t_attr, dict(relation_type="semantic_fact", weight=1.0,
                                                    confidence=0.99, provenance="ast_extraction")))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            line = line_of(node.lineno)
            t_import = _token_at_char(tok, line, line.find("import") + len("import"))
            for alias in node.names:
                name_start = line.find(alias.name)
                if name_start < 0:
                    continue
                t_name = _token_at_char(tok, line, name_start + len(alias.name))
                if is_real_unit(t_import) and is_real_unit(t_name):
                    facts.append((t_import, t_name, dict(relation_type="semantic_fact", weight=1.0,
                                                          confidence=0.99, provenance="ast_extraction")))
    return facts


def build_cooccurrence_edges(ids: list, min_freq: int = 3) -> list:
    """Consecutive-token counts. Confidence = P(tgt | src), the empirical
    conditional probability (freq of this pair / total outgoing frequency
    of src) -- NOT an absolute-frequency clamp against a fixed constant.

    Found and fixed a real, scale-breaking bug: the original formula was
    confidence=min(0.99, freq/max_freq) with max_freq=80, a constant
    calibrated for a ~50K-token corpus ("scaled down from swarm.md's (5,
    100), those assumed 100K-1M tokens, ours are ~50K"). Once the corpus
    grew to 47M+ tokens (~940x), this stopped meaning anything: a pair
    occurring ~77 times -- statistical noise at this corpus size -- still
    computed confidence=0.96 (near-maximum), because the formula only ever
    compared against the stale absolute constant, never against how many
    OTHER continuations the same source token had. Concretely diagnosed
    via inference.py's disagreement-abstention check: a hyper-common token
    (433 outgoing edges) had a top edge with confidence=0.9625 but
    weight=0.00000196 -- internally contradictory, and it was vetoing
    genuinely reasonable neural predictions this way on 9/10 real
    benchmark prompts. P(tgt | src) fixes this without a magic constant:
    a token with 433 diffuse possible continuations naturally gets LOW
    confidence on any single one, honestly reflecting that it isn't a
    reliable, specific prediction -- no arbitrary scale-dependent cutoff
    needed. max_freq upper bound removed for the same reason: it used to
    exclude genuinely common, reliable high-frequency patterns while
    keeping noisy borderline ones through the confidence formula's back
    door; a low min_freq noise floor is enough now that confidence itself
    correctly downweights diffuse/generic edges.
    """
    counts = Counter(zip(ids[:-1], ids[1:]))
    total = sum(counts.values())
    src_totals: Counter = Counter()
    for (src, _tgt), freq in counts.items():
        src_totals[src] += freq
    edges = []
    for (src, tgt), freq in counts.items():
        if freq >= min_freq:
            confidence = freq / src_totals[src]
            edges.append((src, tgt, dict(relation_type="co_occurrence", weight=freq / total,
                                          confidence=confidence, frequency=freq,
                                          provenance="training_corpus")))
    return edges


def add_model_prediction_edges(graph: TokenGraph, model, ids, confidence_threshold: float = 0.95,
                                n_samples: int = 500, block_size: int = 128) -> int:
    """Third knowledge source: the model's own high-confidence predictions,
    added back as graph edges (swarm.md's mechanism, minus the "expert
    agreement" part -- no swarm, so the gate is just this single model's
    own softmax confidence). Only genuinely novel edges are added -- if the
    model is highly confident about something the graph already has an
    edge for, that's confirmation, not new knowledge, so it's skipped.
    Confidence-gated at a high bar (0.95, matching swarm.md) specifically
    because this is the riskiest of the three sources: a confidently wrong
    model prediction, added uncritically, becomes a self-reinforcing error
    the next time the graph is queried (the echo-chamber risk flagged in
    the conversation record's weaknesses list). Nothing here cross-checks
    against semantic facts before adding -- that override still only
    happens at query/add_edge time via confidence comparison.
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    added = 0
    ids_t = ids if torch.is_tensor(ids) else torch.tensor(ids)
    with torch.no_grad():
        for _ in range(n_samples):
            start = torch.randint(0, len(ids_t) - block_size - 1, (1,)).item()
            chunk = ids_t[start : start + block_size].unsqueeze(0)
            logits, _, _, _ = model(chunk)
            probs = F.softmax(logits[0, -1], dim=-1)
            conf, pred = probs.max(dim=-1)
            src = ids_t[start + block_size - 1].item()
            tgt = pred.item()
            if conf.item() < confidence_threshold:
                continue
            if tgt in graph.successors(src):
                continue  # already known -- confirmation, not new knowledge
            graph.add_edge(src, tgt, relation_type="model_prediction", weight=conf.item(),
                            confidence=conf.item(), provenance="model_inference")
            added += 1
    return added


def add_user_correction(graph: TokenGraph, context_last_token: int, correct_token: int,
                         incorrect_token: int = None) -> None:
    """Post-hoc fix, no retraining: if the graph (or by extension the model)
    was about to suggest something wrong, this corrects it immediately --
    the next query sees the fix. Highest-confidence provenance, so it wins
    over any existing edge for this (src, correct_token) pair, and if an
    incorrect edge is named, it's downweighted (not deleted -- keeps the
    record that it was once believed, at lower confidence) rather than
    silently erased.
    """
    if incorrect_token is not None and incorrect_token in graph.successors(context_last_token):
        edge = graph.edges[context_last_token][incorrect_token]
        edge["weight"] *= 0.5
        edge["confidence"] *= 0.5
    graph.add_edge(context_last_token, correct_token, relation_type="semantic_fact", weight=1.0,
                    confidence=1.0, provenance="user_correction")


def build_graph(tok, code_source: str, *id_tensors) -> TokenGraph:
    """AST-fact edges from code_source (skipped if empty -- no code corpus
    in scope) plus co-occurrence edges from every id tensor passed in.
    Generalized from a hardcoded (rj_ids, code_ids) pair so a single-corpus
    caller (Ducky's current text-only focus) isn't forced to invent a
    second, unused corpus just to satisfy the signature.
    """
    graph = TokenGraph()
    if code_source:
        for src, tgt, meta in build_ast_fact_edges(tok, code_source):
            graph.add_edge(src, tgt, **meta)
    for ids in id_tensors:
        ids = ids.tolist() if hasattr(ids, "tolist") else ids
        for src, tgt, meta in build_cooccurrence_edges(ids):
            graph.add_edge(src, tgt, **meta)
    return graph


if __name__ == "__main__":
    import torch

    from tokenizer import Tokenizer

    tok = Tokenizer()
    code_source = (ROOT / "data" / "code" / "corpus.txt").read_text()
    rj_ids = torch.load(ROOT / "data" / "cache" / "rj.pt")
    code_ids = torch.load(ROOT / "data" / "cache" / "code.pt")

    graph = build_graph(tok, code_source, rj_ids, code_ids)
    ast_pairs = graph.edges_by_provenance("ast_extraction")
    print(f"graph: {graph.num_edges()} total edges "
          f"({len(ast_pairs)} AST facts, {graph.num_edges() - len(ast_pairs)} co-occurrence)")

    print("\nspot-check sample (AST facts, for manual precision check):")
    for s, t in random.sample(ast_pairs, min(10, len(ast_pairs))):
        print(f"  {tok.decode([s])!r} -> {tok.decode([t])!r}")
