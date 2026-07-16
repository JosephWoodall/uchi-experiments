"""Build the code corpus from a deliberate, fixed set of CPython stdlib
modules: chosen for being pure-Python, idiomatic, and well-docstringed
(PSF license) -- not a scrape, a hand-picked list, same as the original 5.

Grown three times now: 5 -> 10 -> 49 modules (~150KB -> ~425KB -> ~2MB).
This third growth is deliberately bigger (~4.7x) than the second (~2.8x) --
motivated by "make Ducky denser" running into the Chinchilla mismatch
(940K params was already ~125x over-parameterized for the 149K-token
corpus): the honest fix is proportional data+param growth, not params
alone. Still a hand-picked, general-purpose, non-niche subset of the
stdlib (avoiding debugging-tool modules like pdb/pickletools and giant
doc/data dumps like pydoc_data), not a switch to scraped/broad data.

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
    "argparse.py", "difflib.py", "ipaddress.py", "inspect.py", "subprocess.py",
    "doctest.py", "json/decoder.py", "json/encoder.py", "json/__init__.py",
    "csv.py", "configparser.py", "logging/__init__.py", "unittest/case.py",
    "collections/__init__.py", "shutil.py", "tempfile.py",
    "decimal.py", "queue.py", "threading.py", "datetime.py", "calendar.py",
    "random.py", "uuid.py", "base64.py", "gzip.py", "glob.py",
    "fnmatch.py", "shlex.py", "pprint.py", "reprlib.py", "weakref.py",
    "copy.py", "numbers.py", "traceback.py", "warnings.py", "abc.py",
    "graphlib.py", "typing.py", "ast.py",
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
