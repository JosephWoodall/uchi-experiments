"""Build the code corpus from the full local CPython stdlib plus a curated
set of already-installed, permissively-licensed third-party libraries --
same "curated bulk, still license-clean, zero download" principle as the
stdlib-only growth, extended to close most of the remaining code-side
Chinchilla data deficit for free.

Grown five times now: 5 -> 10 -> 49 -> full stdlib (~11MB) -> full stdlib
+ curated site-packages (~180MB+). The site-packages libraries below were
individually checked for license before inclusion (all BSD/MIT/Apache/
PSF-style permissive, none GPL/copyleft): torch, sympy, scipy, jax, pandas,
scikit-learn, mne, matplotlib, jaxlib, nltk, pygments, fonttools, networkx,
numpy, textual, moabb, orbax, setuptools, flax, docutils, mpmath, optax,
pytest, rich, Pillow. Same exclusions as the stdlib pass (tests, parse
failures) plus one more specific to third-party packages: vendored
third-party code bundled *inside* another package (a "_vendor"/"vendor"
subdirectory) is excluded, since its own license may differ from the
parent package's and wasn't independently checked.

Produces:
  data/code/corpus.txt        - everything concatenated (call-graph/AST
                                 tooling reads this single file)
  data/code/corpus_core.txt   - stdlib only: simple, idiomatic utility-style
                                 code, the style actually tested by
                                 bench_ducky.py's held-out tasks
  data/code/corpus_breadth.txt - site-packages libraries only: real code,
                                 but dominated by large ML/scientific
                                 library internals, a different style
  data/code/pairs.jsonl       - {doc, code} pairs per function, for `jepa-aux`

Split into core/breadth because they're wildly imbalanced by raw size
(core ~11MB, breadth ~150MB) -- uniform sampling over the concatenated
corpus would make core's simple-utility style a rounding error during
training. get_weighted_code_batch (data.py) samples from these as two
separate weighted pools instead, the same pattern get_joint_batch already
uses to keep small pixel/audio pools from being drowned out by much larger
text/code ones.
"""
import ast
import json
import site
import sysconfig
from pathlib import Path

STDLIB_DENYLIST_DIRS = {
    "test", "tests", "idlelib", "turtledemo", "tkinter", "lib2to3",
    "ensurepip", "__pycache__", "site-packages",
}
STDLIB_DENYLIST_FILES = {"this.py", "antigravity.py"}

SITE_PACKAGES_DENYLIST_DIRS = {
    "test", "tests", "testing", "__pycache__",
    "_vendor", "vendor", "vendored", "_vendored",
}
ALLOWED_THIRD_PARTY_PACKAGES = [
    "torch", "sympy", "scipy", "jax", "pandas", "sklearn", "mne",
    "matplotlib", "jaxlib", "nltk", "pygments", "fontTools", "networkx",
    "numpy", "textual", "moabb", "orbax", "setuptools", "flax", "docutils",
    "mpmath", "optax", "_pytest", "rich", "PIL",
    # Second batch -- this is the realistic zero-download ceiling for code
    # given what's installed locally (measured: ~19.3M more chars across
    # 92 packages before this list, all checked as well-known, permissively
    # licensed projects, same diligence as the first batch): fastapi (MIT),
    # requests (Apache-2.0), click (BSD), urllib3 (MIT), joblib (BSD),
    # beautifulsoup4 (MIT), PyYAML (MIT), httpx/httpcore/anyio/starlette/
    # uvicorn (BSD/MIT), coverage (Apache-2.0), python-dateutil
    # (Apache/BSD dual), idna (BSD-style), markdown-it-py (MIT),
    # pyparsing (MIT), seaborn (BSD), absl-py/google-grain/etils/treescope
    # (Apache-2.0, Google), triton (MIT), torchgen (BSD, pytorch tooling),
    # fsspec (BSD), mpl_toolkits (matplotlib's own license), pyriemann/
    # mne_bids (BSD).
    "triton", "torchgen", "grain", "seaborn", "google", "fastapi", "fsspec",
    "coverage", "joblib", "pyriemann", "treescope", "mne_bids", "anyio",
    "gemma", "mpl_toolkits", "absl", "pyparsing", "urllib3", "click", "bs4",
    "etils", "idna", "httpx", "dateutil", "httpcore", "starlette", "uvicorn",
    "yaml", "requests", "markdown_it",
]

MIN_DOC_CHARS = 20

stdlib = Path(sysconfig.get_path("stdlib"))
# The installed libraries (torch, sympy, etc.) live in the user site-packages
# (pip --user / venv-less local install), not sysconfig's "purelib" (system
# site-packages, which is a different, mostly-empty directory here).
purelib = Path(site.getusersitepackages())
out_dir = Path(__file__).resolve().parent.parent / "data" / "code"
out_dir.mkdir(parents=True, exist_ok=True)


def stdlib_eligible(path: Path) -> bool:
    if path.name in STDLIB_DENYLIST_FILES:
        return False
    return not any(part in STDLIB_DENYLIST_DIRS for part in path.relative_to(stdlib).parts)


def site_pkg_eligible(path: Path, pkg_root: Path) -> bool:
    return not any(part in SITE_PACKAGES_DENYLIST_DIRS for part in path.relative_to(pkg_root).parts)


core_files = [(p, stdlib) for p in sorted(stdlib.rglob("*.py")) if stdlib_eligible(p)]
breadth_files = []
for pkg_name in ALLOWED_THIRD_PARTY_PACKAGES:
    pkg_root = purelib / pkg_name
    if not pkg_root.is_dir():
        print(f"  (skipping {pkg_name}: not found at {pkg_root})")
        continue
    breadth_files.extend(
        (p, purelib) for p in sorted(pkg_root.rglob("*.py")) if site_pkg_eligible(p, purelib)
    )

pairs = []
n_parse_failed = 0


def extract(files_with_roots: list) -> list:
    global n_parse_failed
    chunks = []
    for src_path, root in files_with_roots:
        name = str(src_path.relative_to(root))
        try:
            source = src_path.read_text()
            tree = ast.parse(source, filename=name)
        except (SyntaxError, UnicodeDecodeError, ValueError):
            n_parse_failed += 1
            continue

        chunks.append(f"# --- {name} ---\n{source}")
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
    return chunks


core_chunks = extract(core_files)
breadth_chunks = extract(breadth_files)

(out_dir / "corpus_core.txt").write_text("\n\n".join(core_chunks))
(out_dir / "corpus_breadth.txt").write_text("\n\n".join(breadth_chunks))
(out_dir / "corpus.txt").write_text("\n\n".join(core_chunks + breadth_chunks))
with (out_dir / "pairs.jsonl").open("w") as f:
    for p in pairs:
        f.write(json.dumps(p) + "\n")

core_chars = sum(len(c) for c in core_chunks)
breadth_chars = sum(len(c) for c in breadth_chunks)
print(f"corpus_core.txt (stdlib): {core_chars:,} chars from {len(core_chunks)} modules")
print(f"corpus_breadth.txt (site-packages): {breadth_chars:,} chars from {len(breadth_chunks)} modules")
print(f"corpus.txt (combined): {core_chars + breadth_chars:,} chars "
      f"({n_parse_failed} skipped -- failed to parse)")
print(f"pairs.jsonl: {len(pairs)} (doc, code) pairs")
