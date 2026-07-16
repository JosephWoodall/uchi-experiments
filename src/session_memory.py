"""Session-scoped working memory -- deliberately NOT graph.py's TokenGraph.
The graph is corpus-level, static: the same edges regardless of where you
are in a generation, so it can't answer "did I already say something
different for this exact context a moment ago." This trie tracks only
what happened earlier in the CURRENT generation, built fresh at the start
of generate_with_grounding/generate_with_resampling and discarded when it
returns -- it never touches disk, so there's nothing to secure at rest by
design (see tasks/ducky.md for why BPTT's carried hidden state can't be
trusted to do this implicitly: three rounds of testing found no evidence
it retains anything across chunks, so this makes memory explicit instead).

Keys are blake2b digests chained through the whole path (parent digest ->
child digest), not a hash of one token in isolation -- with only 8192
possible tokens, hashing a single token is trivially reversible by
building an 8192-entry lookup table once. Chaining means guessing the
digest at depth d requires guessing the full d-token prefix that produced
it (8192^d possibilities), which is a meaningfully larger space -- but
this is obfuscation, not cryptographic security: there is no secret key,
and nothing here defends against a motivated attacker with the model and
enough compute. Its actual job is efficiency (fixed 8-byte dict keys
instead of variable-length tuples) and not keeping raw token sequences
lying around in an obviously-readable form, not confidentiality.
"""
import hashlib
from typing import Optional

_ROOT_SALT = b"ducky-session-trie-v1"


def _chain_hash(parent_digest: bytes, token_id: int) -> bytes:
    h = hashlib.blake2b(digest_size=8)
    h.update(parent_digest)
    h.update(token_id.to_bytes(4, "little", signed=False))
    return h.digest()


class _Node:
    __slots__ = ("children", "next_tokens")

    def __init__(self):
        self.children: dict[bytes, "_Node"] = {}
        self.next_tokens: dict[int, int] = {}


class SessionTrie:
    """max_depth bounds both memory and per-token insert cost: observe()
    only ever walks/creates at most max_depth nodes regardless of how long
    the generation has run, so total work across a generation is
    O(max_depth * tokens_generated) -- linear in what Ducky has generated,
    not quadratic, matching the actual growth requirement.
    """

    def __init__(self, max_depth: int = 8):
        self.max_depth = max_depth
        self.root = _Node()

    def observe(self, context: list[int], next_token: int) -> None:
        """context = tokens BEFORE next_token. Records next_token as a
        continuation at every depth along the trailing window, so a
        lookup later can match whatever prefix length is actually
        available (short early in a generation, up to max_depth later).
        """
        window = context[-self.max_depth :]
        node = self.root
        digest = _ROOT_SALT
        for tok in window:
            digest = _chain_hash(digest, tok)
            node = node.children.setdefault(digest, _Node())
            node.next_tokens[next_token] = node.next_tokens.get(next_token, 0) + 1

    def lookup(self, context: list[int]) -> Optional[dict[int, int]]:
        """Walks as deep as the trie has data for this exact trailing
        context, returns the deepest (most specific) match's
        {token: count} of what followed it before in this generation, or
        None if this context has never occurred before at any depth.
        """
        window = context[-self.max_depth :]
        node = self.root
        digest = _ROOT_SALT
        best: Optional[dict[int, int]] = None
        for tok in window:
            digest = _chain_hash(digest, tok)
            if digest not in node.children:
                break
            node = node.children[digest]
            best = node.next_tokens
        return best
