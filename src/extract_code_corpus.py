"""Build the code corpus from a deliberate, fixed set of CPython stdlib
modules: chosen for being pure-Python, idiomatic, and well-docstringed
(PSF license) -- not a scrape, a hand-picked list, same as the original 5.

Grown from 5 to 10 modules (~150KB -> ~425KB) to reduce signal-starvation
in a few diagnosed places (AST-fact fragmentation, sparse model-prediction
edges, thin confidence distributions) -- deliberately still small and
hand-picked, not a switch to broad/scraped data.

The existing tokenizer/vocab is NOT retrained here -- growing the corpus
this way keeps every checkpoint trained so far loadable and comparable
(same vocab_size, same token-id meanings). Only the `code` token cache
(data/cache/code.pt) needs regenerating, and only for future runs that
load it fresh.

Produces:
  data/code/corpus.txt   - concatenated raw source, for `base`/`mtp` arms
  data/code/pairs.jsonl  - {doc, code} pairs per function, for `jepa-aux`
"""
import ast
import json
import sysconfig
from pathlib import Path

MODULES = [
    "statistics.py", "textwrap.py", "heapq.py", "bisect.py", "fractions.py",
    "enum.py", "dataclasses.py", "contextlib.py", "functools.py", "pathlib/__init__.py",
]
MIN_DOC_CHARS = 20

stdlib = Path(sysconfig.get_path("stdlib"))
out_dir = Path(__file__).resolve().parent.parent / "data" / "code"
out_dir.mkdir(parents=True, exist_ok=True)

raw_chunks = []
pairs = []

for name in MODULES:
    src_path = stdlib / name
    source = src_path.read_text()
    raw_chunks.append(f"# --- {name} ---\n{source}")

    tree = ast.parse(source, filename=name)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node)
        if not doc or len(doc) < MIN_DOC_CHARS:
            continue
        code = ast.get_source_segment(source, node)
        if not code:
            continue
        pairs.append({"doc": doc.strip(), "code": code, "module": name, "name": node.name})

(out_dir / "corpus.txt").write_text("\n\n".join(raw_chunks))
with (out_dir / "pairs.jsonl").open("w") as f:
    for p in pairs:
        f.write(json.dumps(p) + "\n")

print(f"corpus.txt: {sum(len(c) for c in raw_chunks)} chars from {len(MODULES)} modules")
print(f"pairs.jsonl: {len(pairs)} (doc, code) pairs")
