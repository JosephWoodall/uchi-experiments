"""Build the code corpus from the full local CPython stdlib (excluding a
curated denylist), rather than a hand-picked module list -- approved
relaxation of "hand-picked, not scraped" to "curated bulk, still
license-clean" (PSF license, already installed locally, zero download).

Grown four times now: 5 -> 10 -> 49 -> full stdlib (~150KB -> ~2MB ->
~11MB). Earlier growths were a hand-picked list; this one auto-discovers
every .py file under sysconfig's stdlib path and excludes only a small,
principled denylist: test suites (not representative code, mostly
assertions against internal APIs), GUI-heavy modules (tkinter/idlelib/
turtledemo -- overwhelmingly boilerplate widget wiring, low density of
generally-useful patterns), legacy/vendored code (lib2to3, ensurepip's
bundled wheels), and easter eggs (this.py, antigravity.py). Still
excludes __pycache__ and any file that fails to parse (encoding issues,
version-specific syntax) -- logged, not silently dropped.

Produces:
  data/code/corpus.txt   - concatenated raw source, for `base`/`mtp` arms
  data/code/pairs.jsonl  - {doc, code} pairs per function, for `jepa-aux`
"""
import ast
import json
import sysconfig
from pathlib import Path

DENYLIST_DIRS = {
    "test", "tests", "idlelib", "turtledemo", "tkinter", "lib2to3",
    "ensurepip", "__pycache__", "site-packages",
}
DENYLIST_FILES = {"this.py", "antigravity.py"}
MIN_DOC_CHARS = 20

stdlib = Path(sysconfig.get_path("stdlib"))
out_dir = Path(__file__).resolve().parent.parent / "data" / "code"
out_dir.mkdir(parents=True, exist_ok=True)


def eligible(path: Path) -> bool:
    if path.name in DENYLIST_FILES:
        return False
    return not any(part in DENYLIST_DIRS for part in path.relative_to(stdlib).parts)


files = sorted(p for p in stdlib.rglob("*.py") if eligible(p))

raw_chunks = []
pairs = []
n_parse_failed = 0

for src_path in files:
    name = str(src_path.relative_to(stdlib))
    try:
        source = src_path.read_text()
        tree = ast.parse(source, filename=name)
    except (SyntaxError, UnicodeDecodeError, ValueError):
        n_parse_failed += 1
        continue

    raw_chunks.append(f"# --- {name} ---\n{source}")
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

print(f"corpus.txt: {sum(len(c) for c in raw_chunks):,} chars from {len(raw_chunks)} modules "
      f"({n_parse_failed} skipped -- failed to parse)")
print(f"pairs.jsonl: {len(pairs)} (doc, code) pairs")
