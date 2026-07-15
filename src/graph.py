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


def build_cooccurrence_edges(ids: list, min_freq: int = 3, max_freq: int = 80) -> list:
    """Consecutive-token counts. Thresholds scaled down from swarm.md's
    (5, 100) -- those assumed 100K-1M tokens, ours are ~50K.
    """
    counts = Counter(zip(ids[:-1], ids[1:]))
    total = sum(counts.values())
    edges = []
    for (src, tgt), freq in counts.items():
        if min_freq <= freq <= max_freq:
            confidence = min(0.99, freq / max_freq)
            edges.append((src, tgt, dict(relation_type="co_occurrence", weight=freq / total,
                                          confidence=confidence, frequency=freq,
                                          provenance="training_corpus")))
    return edges


def build_graph(tok, code_source: str, rj_ids, code_ids) -> TokenGraph:
    graph = TokenGraph()
    for src, tgt, meta in build_ast_fact_edges(tok, code_source):
        graph.add_edge(src, tgt, **meta)
    for ids in (rj_ids, code_ids):
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
