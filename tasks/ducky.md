# Ducky: This Architecture's North Star

**Core idea:** A next-token predictor whose backbone is mostly linear-recurrent
(unlimited context, O(1) memory) with periodic attention for in-window
quality, backed by a single updatable knowledge graph and a single model's
own calibrated confidence — not a swarm of experts. Ducky's mission is to
**predict tokens well**; grounding and abstention are robustness layers on
top of that mission, not a replacement of it.

**Why this configuration, not the obvious ones:**

Dense attention-only Transformers (this repo's original baseline) cost
grows with context and hard-caps at `block_size`. Pure RWKV (linear
recurrence throughout) removes that cap but underperformed the dense
baseline on held-out loss at toy scale (rj: 4.74 vs 4.36; code: 5.22 vs
4.81) — a fixed-size recurrent state is a more constrained mechanism than
full attention over the window, and real RWKV models need more scale/
tuning to close that gap. The hybrid (3 RWKV blocks + 1 attention block,
matching uchi's own SSM-plus-periodic-attention precedent) beat *both*:
rj 4.351, code 4.636, reproducible on both corpora at matched (~941K)
params. Same budget, better loss than pure attention, unlimited context on
3 of 4 blocks. This is not a hedge between two options — it's evidence the
combination is better than either extreme.

**State-of-the-art grounding:**
- RWKV: Peng et al. 2023 (arXiv:2305.13048) — linear-time, O(1)-memory
  recurrence as an attention replacement.
- BitNet 1.58-bit quantization: Ma et al. 2024 (arXiv:2402.17764) — ported
  from uchi (`uchi/uchi/flux/bitnet.py`), currently wired only into the
  (abandoned) MoE experts, not yet into Ducky's own blocks.
- MoE foundations: Shazeer et al. 2017 (arXiv:1701.06538); DeepSeekMoE
  2024 (arXiv:2401.06066) — tested and **rejected** for this project (see
  Alternatives Rejected).
- Hallucination is not solvable to zero: Kalai & Vempala 2024
  (arXiv:2311.14648) — abstention is measured as verified-accuracy +
  abstention-rate, never "hallucination-free."
- uchi's own precedent (`uchi/README.md`): 17 SSM layers + 3 attention
  layers at positions 5/11/17 ("SSM handles everything else, attention
  gives long-range recall checkpoints") — the design pattern Ducky's
  hybrid directly mirrors, at a much smaller scale.

**Alternatives rejected, with evidence, not assumption:**
1. **Swarm of specialized experts querying a graph differently**
   (`tasks/swarm.md`'s original proposal). Rejected after direct testing:
   routing specialization JS-divergence was 0.0000-0.0002 at *every*
   layer (0 through 3, checked exhaustively, not just layer 0) on the two
   most different domains available (Shakespeare dialogue vs. Python
   stdlib). Without specialization, "swarm" is N redundant experts voting
   on near-identical outputs — cost without benefit.
2. **MoE (learned routing, no swarm)** — tested independently of the
   swarm mechanism. Still lost to dense on held-out loss at every matched
   param count, on both corpora, and cost 30-60% more wall-clock despite
   matched active FLOPs (naive per-expert masking loop, not a batched
   kernel). Rejected at this scale; not proven impossible at larger scale.
3. **Full JEPA-as-primary-objective** (no token decoder) — solves a
   different problem than "predicts the next token"; the auxiliary-loss
   version (`jepa-aux`) is still an open, under-tested arm (86 training
   pairs was too few, since expanded to ~258 with the corpus growth —
   worth revisiting, not rejected outright).
4. **Multi-token prediction (mtp)** — consistently worse than base at
   every matched param count on both corpora; more steps (150->500) did
   not close the gap. Needs the training-scale regime the original paper
   used, not proven dead at toy scale.

**The graph — three knowledge types, one structure, confidence-gated:**
`graph.py`'s `TokenGraph` unifies AST-grounded facts (code only — text
dialogue isn't factual prose, so IE-style extraction was skipped entirely,
not attempted and failed), co-occurrence statistics (domain-agnostic), and
the model's own high-confidence discoveries (`add_model_prediction_edges`,
>0.95 confidence, novelty-gated so confirmation isn't mistaken for new
knowledge). Facts always win over statistics on a conflicting edge
(confidence-based override, not a special case). `add_user_correction`
takes effect on the next query with zero retraining — a different, more
practical notion of hallucination-resistance than "the model never
confabulates": *can wrong things be fixed cheaply after the fact.*

**Known, honest limitations, not hidden:**
- AST-fact precision degrades at this vocab/corpus scale — rare
  identifiers fragment to near-character-level BPE tokens, producing
  low-semantic-value edges. Not fully fixed; worked around (not solved)
  by `grounding.py`'s `identifier_grounded`, which checks decoded strings
  against a real symbol table instead of token boundaries.
- The unlimited-context mechanism is structurally proven (constant
  memory, linear time, verified directly on the Ducky checkpoint, not
  just standalone RWKV) but this checkpoint still doesn't *use* it for
  long-range recall. Cross-chunk BPTT training (`train_bptt.py`, K=4
  chunks, state not detached across them) was built and run specifically
  to fix this — result: **negative, and decisively so**. KL divergence
  between carried-state and fresh-state predictions is 0.0000 at every
  horizon checked (128 through 640 tokens), including 512 tokens, exactly
  the span BPTT training was designed to cover. Diagnosed cause, not a
  mystery: 700 steps is a small budget to shift learned decay rates, and
  the hybrid's attention layer gives the model an escape hatch — it can
  hit low loss per chunk via local attention alone, with no pressure to
  ever rely on the carried RWKV state. Open, untested question: would a
  pure-RWKV BPTT run (no attention escape hatch) or a much larger step
  budget actually induce retention? Ducky's unlimited-context property
  remains real and unused, not real and exploited.
  **Both candidate explanations now tested and resolved.** (1) Escape
  hatch: ruled out -- pure RWKV (attention removed entirely) gave the
  identical null result (KL~0.0000 at every horizon). (2) Insufficient
  steps: also resolved, in the opposite direction than expected -- 5000
  steps (7x the original budget) made things *worse*, not better. Best
  val loss occurred at step 500 (4.382), then exploded to 8.283 by step
  5000 (worse than random guessing) while train loss collapsed to 0.040 --
  severe memorization, far past any point where long-range retention
  could plausibly be rewarded. (Also found: `train_bptt.py` saves the
  *final* state to a file named `..._best.pt`, not the true best --
  same misleading-name bug class as before; not worth a rerun to fix
  given the conclusion below.) **Conclusion: this is the same data-ceiling
  theme that has dominated the whole session.** The corpus is too small
  for "use distant context" to ever out-compete "memorize what's already
  been seen," at any BPTT step count tested. Not an architecture failure
  -- a fixed, small-corpus ceiling that this specific mechanism can't get
  past. Would need a genuinely larger corpus, not more steps, to test
  fairly.
  **Follow-up done, decisive**: reran BPTT with attention_layers=()
  (pure RWKV, escape hatch fully removed). KL still ~0.0000 at every
  horizon (128-640 tokens) — identical to the hybrid result. This rules
  out the escape-hatch hypothesis directly rather than leaving it
  unresolved: removing the one plausible confound changed nothing, so the
  bottleneck is the other hypothesis (700 BPTT steps is too little signal
  to shift learned decay rates), by elimination, not by assumption. A much
  larger BPTT step budget is the remaining untested lever.
  **Third round, "big data + big budget together," also decisive, also
  negative**: reran on the fully-grown 49-module corpus (387K tokens
  available, vs the much smaller corpus every earlier round used) at
  vocab=8192, 3000-step budget with early stopping (best@1250, val loss
  5.0054, stopped at 2750 once it started rising again). New eval script
  (`eval_bptt_retention.py`, generalizes test_unlimited_context.py's KL
  check across 5 horizons instead of one) still found KL=0.0000-0.0001 at
  every horizon (128 through 640 tokens) — the carried-state and
  fresh-state predictions pick the identical top token every time, not
  just a similar one. Bigger data and enough steps to reach a real,
  early-stopped optimum still didn't move this. Also fixed a real bug
  found while re-running this: `train_bptt.py` saved the *final* checkpoint
  to a file misleadingly named `..._best.pt` (flagged but left unfixed
  after the second round); now tracks and saves the actual best checkpoint
  with early stopping, matching train.py's pattern. **Conclusion, now
  three-for-three across data scale, step budget, and architecture
  (hybrid vs pure-RWKV): this BPTT training setup does not induce
  cross-chunk retention in this model, under any tested combination of
  those three levers.** Not fully explained -- the training loss itself
  does improve substantially (loss 5.33 -> 2.23 over the run), so the
  model is learning something real from the objective, just not
  "carry information forward across chunk boundaries in a way a fresh
  state wouldn't already produce." Remaining untested candidate causes:
  the loss may be achievable entirely by local (within-chunk) pattern
  matching with no gradient signal ever favoring cross-chunk information
  flow specifically, or the state-carrying mechanism's gradient path may
  need an explicit loss term that rewards it (e.g. only computing loss on
  the final chunk, forcing all earlier chunks to earn their keep purely
  through the carried state) rather than the current per-chunk-averaged
  loss, where every chunk can independently minimize its own loss without
  ever needing the others.
- Grounding/abstention validated as a genuine net positive, not just
  mechanically sound: selective-prediction check shows accuracy on
  answered (non-abstained) tokens is meaningfully higher than the
  unconditional baseline on both domains (rj +5.3 points, code +3.3
  points, `eval_grounding.py`). This closes the "difference vs.
  improvement" gap the swarm.md postmortem originally flagged.
- Abstention thresholds are now calibrated against this checkpoint's own
  confidence distribution (not borrowed production-scale numbers), and
  the fast/slow-path split behaves sensibly (mostly answers, abstains
  ~20% of the time on real data, not by default). The four grounding
  signals in `grounding.py` are wired in (`predict_next` for the per-token
  n-gram check, `generate_with_grounding` for the span-level syntax/self-
  critique/identifier checks) and tested end-to-end — current behavior at
  700 steps is to abstain within 1-4 tokens on most real prompts, which is
  honest calibration for an undertrained toy model, not a defect.
- BitLinear convergence check done: extended to 3000 steps on the grown
  code corpus, found its ceiling at step 1875 (val loss 3.8519) — beats
  dense's ceiling (3.921, step 1250) but doesn't fully close the gap to
  the unquantized hybrid (3.8267, step 1250), despite needing 50% more
  steps to get there. Matches expectations: quantization costs some
  quality, mostly recoverable with more training, not free.
- Repeat-seed check done (3 seeds, both corpora): **hybrid beats dense
  6/6.** On code, a strong, reliable win (mean margin 0.152 nats, std
  0.035 — small relative to the mean). On rj, the win is real but the
  *size* varies a lot seed to seed (mean margin 0.035, std 0.023, nearly
  as large as the mean) — direction is trustworthy, a precise margin
  number on rj specifically is not. Also confirmed the code corpus's
  growth (51K->149K tokens) mattered independent of architecture — dense's
  own baseline moved more from that (4.815->~4.06) than the hybrid-vs-dense
  gap did.

**Training efficiency:** the WKV recurrence's Python `for`-loop was the
concrete, measured bottleneck (hybrid: 0.39-0.58s/step vs dense's 0.15s/step)
— the same one uchi already solved for their own SSM scan
(`UCHI_FUSE_SSM_SCAN=1`, README). Applying the identical fix
(`torch.compile` over the whole scan, not a switch to a parallel scan,
which uchi tried and reverted for memory-bandwidth reasons) gives a
measured **2.65x steady-state speedup** (0.5757s/step eager -> 0.2174s/step
compiled, controlled same-script comparison), at a one-time ~169s
compilation cost that pays for itself within a single 700-step run.
Verified numerically identical to the eager version (max diff ~1e-7,
float32 rounding, not a real discrepancy) — same math, just fused. Env var
`UCHI_FUSE_SSM_SCAN=0` disables it if `torch.compile` misbehaves in a given
environment, same escape-hatch name uchi itself uses.

Went further: `--compile-full-model` (`train.py`) compiles the whole
forward pass, not just the scan. Measured 0.146s/step steady-state —
actually faster than plain dense's *uncompiled* 0.15s/step, a 3.95x
speedup over the original eager baseline (beats uchi's own reported 3x).
Verified numerically correct (diff ~7e-7). Real tradeoff, not free: ~480s
one-time compile cost vs ~170s for scan-only, breakeven at ~4300 extra
steps — opt-in for long runs, not the default for quick experiments.
Also found and fixed a real bug while testing this (an overly broad
`replace_all` renamed a function parameter but not its body reference —
caught immediately by a crash, not silently wrong). And a genuine, honest
limitation surfaced, present in the *default* scan-only compilation too,
not just full-model: `generate()`'s token-by-token growing context length
means the compiled scan gets recompiled for each new shape it sees during
sampling, capped at 8 recompiles before falling back to eager. Bounded,
doesn't affect correctness or the fixed-length training loop, but a real
one-time cost during sample generation specifically — not hidden.
CPU thread count also tuned: 8 threads measured optimal (0.209s/step) vs
the PyTorch default of 10 (0.252s/step) vs 16-20 (0.91-1.31s/step, 4-6x
slower from thread-sync overhead) — now the default in `train.py`.

bfloat16 tested and **rejected**, not adopted: only ~13% speedup
(0.0133s->0.0116s/call) against real, non-trivial precision loss (up to
8.5% of a standard deviation, worst case) in a recurrence run sequentially
across 128 timesteps, where per-step error can compound. Risk/reward
doesn't clear the bar next to `torch.compile`'s exact-correctness win.

Automated early stopping added (`train.py`, `--patience`/`--min-delta`,
default disabled to match every run so far): verified stopping at step 900
instead of running the full 2000-step budget, correctly identifying the
same best checkpoint (step 700) every extended sweep this session had to
discover by running long and reading off the best point afterward.

Ducky SDK (`ducky.py`) setup caching added: graph-building (300 forward
passes) + threshold calibration (500+ more) cached to disk, keyed by
checkpoint mtime so a retrained/overwritten checkpoint invalidates the
cache automatically rather than serving stale data. Verified: 10.71s cold
-> 2.24s warm, identical thresholds/graph/output confirmed, not just faster.

Ducky SDK also now supports `backbone="dense"` alongside the default
`"hybrid"` -- same domain, same API, either backbone, so the dense-vs-Ducky
comparison can be run directly through the SDK, not just read from a table.

**New-generation Ducky trained** (vocab=8192, 12 layers, rank-64 factored
embedding, no BitLinear, grown 49-module corpus): hybrid xl beat dense xl
again -- best val 5.1643 (step 1750) vs 5.2873 (step 2500), the same
architecture win reproduced at the new scale. (These loss values aren't
comparable to earlier sub-3.1 numbers -- an 8192-way softmax has a much
higher cross-entropy floor than 1024-way, especially early in training;
same-vocab comparisons only.) `ducky.py`'s `DEFAULT_RUNS` now points
code/hybrid and code/dense at these new checkpoints; rj/* stays on the old
(vocab=1024) generation since rj hasn't been retrained at the new scale yet.

Getting both generations to load correctly through one SDK surfaced three
real bugs, each fixed:
1. `config.json` never recorded which vocab size a checkpoint was trained
   under -- `train.py` now saves it explicitly; the two xl checkpoints
   trained just before that fix landed were patched by hand.
2. `data.py`'s per-corpus tokenize cache (`data/cache/{name}.pt`) wasn't
   versioned by vocab size either -- the vocab-8192 training run had
   already silently overwritten it with new-vocab ids mid-session, which
   would have broken loading any 1024-vocab checkpoint through that path.
   Fixed the same way: `{name}_{vocab_size}.pt`.
3. `ducky.py` assumed that cache file already existed (raw `torch.load`)
   instead of calling `load_lm_corpus` (which creates it) -- broke the
   first time a checkpoint used a vocab size nothing had tokenized the
   full corpus under yet. Fixed by calling `load_lm_corpus` directly.

End-to-end regression check confirms all of this holds together: the new
vocab=8192 xl checkpoints (hybrid and dense) and the old vocab=1024 rj
checkpoint all load and answer correctly through the same `Ducky()`
constructor in the same process, each picking its own correct vocab size
from its own config -- old and new generations coexist, neither broke the
other.

Reject-and-resample built (`inference.py`'s `generate_with_resampling`, wired
into `ducky.py`'s `ask(n_candidates=N)`): the grounding signals were
previously a smoke detector (reported after the fact) — this makes them a
sprinkler system (used to pick the output). Required first fixing that
`predict_next` was fully deterministic (always argmax): a `temperature`
parameter was added so repeated calls on the same prompt can genuinely
differ, while the abstain/don't-abstain decision itself still always uses
the greedy-argmax confidence (temperature only affects which token gets
returned once the model doesn't abstain) — sampling diversity shouldn't be
allowed to destabilize the calibrated abstention behavior itself. Verified
on a real checkpoint (`code_base_l`, dense, best val 3.027): for one prompt,
the deterministic path (temperature=0) returned a syntactically-invalid
completion (self-critique 0.149); resampling 8 candidates at temperature=0.8
selected a different candidate with *lower* self-critique (0.024) but valid
syntax, correctly outscoring it via the decisive syntax-validity bonus
(1.024 vs 0.149) — proof the selection logic, not just the sampling, is
doing real work. Abstained candidates score `-inf` internally and are never
selected over a candidate that produced anything.
Also found and fixed, while testing this: `data.py`'s per-corpus tokenize
cache (`data/cache/{name}.pt`) wasn't versioned by vocab size, so the
vocab-8192 training run silently overwrote it with new-vocab ids, breaking
any future load of a 1024-vocab checkpoint through that path — same bug
class as the tokenizer `MODEL_PREFIX` issue, fixed the same way
(`{name}_{vocab_size}.pt`).

Session-scoped working memory built (`session_memory.py`'s `SessionTrie`,
wired into `inference.py` as a third grounding signal): distinct from
`TokenGraph` (corpus-level, static -- same regardless of position in a
generation) and from `ngram_index` (also corpus-level). This tracks only
what THIS generation has already produced, catching self-contradiction
(a different token chosen for an identical trailing context within one
generation) that neither of the other two signals can see. Built fresh per
`generate_with_grounding`/`generate_with_resampling` call, discarded when
it returns -- deliberately never persisted to disk (explicit scope
decision: this is a per-generation working memory, not a cross-session
one). Keys are blake2b digests chained through the whole path (parent
digest -> child digest), not a hash of one token in isolation -- with only
8192 possible tokens a single-token hash would be trivially reversible by
brute force; chaining means guessing a node at depth d requires guessing
the whole d-token prefix (honest caveat: this is obfuscation, not real
cryptographic security -- no secret key, no adversary model, nothing here
is ever persisted or transmitted). `max_depth` bounds both memory and
per-token insert cost, so total work grows linearly with tokens generated,
not quadratically. Observability only so far, same incremental pattern as
self-critique/syntax-validity before reject-and-resample gave them teeth:
annotates `session_consistent` in the info dict, doesn't yet override the
chosen token or trigger abstention. Verified with unit tests (repeat
context -> count increments; different continuation for an identical
context -> both recorded, distinguishable; never-seen context -> None) and
a real end-to-end SDK call with no regressions.

**Four uchi-inspired reasoning mechanisms built and benchmarked, honestly.**
Ultimate goal stated by the user: scale Ducky enough to eventually score
well on MMLU/SWE-bench. Reality check given before building anything: at
~10M params / ~2M training characters, Ducky is 6-7 orders of magnitude
below the scale/data needed for either benchmark to be a meaningful target
today -- MMLU needs broad world knowledge across 57 subjects never in this
corpus, SWE-bench needs correct patches against large unfamiliar repos far
beyond stdlib-snippet scale. User's response: this is intentional -- Ducky
is the small-scale testbed for hallucination-resistance and training
efficiency, meant to be scaled up later, matching the project's original
premise. So: build the mechanisms now, validate honestly at Ducky's real
scale, benchmark against a realistic (not predetermined-zero) task set.

Built: (1) `mcts_lite.py` -- value-guided PUCT/UCB search over chunk-level
generation branches, using self_critique_score as an honest proxy value
function (uchi's own WorldModel.ValueHead is trained; nothing here is,
same "no second model" discipline). (2) `repair_loop.py` -- sequential,
feedback-informed retry (real syntax errors spliced into the next
attempt's prompt, max_attempts=4 matching uchi's own retry cap), tri-state
PASS/FAIL/ABSTAIN outcome. (3) `session_history.py` -- extractive,
verbatim-only cross-call memory for one Ducky() instance's lifetime
(RAM-only, opt-in via `track_history=True`). (4) `check_call_arity_consistency`
in `grounding.py` -- narrow, deterministic, additive-only veto (same scope
discipline as uchi's relational_reasoning.py), using the session-memory
trie's underlying idea at the AST level instead of the token level. Every
one of these was individually validated with real, not cherry-picked,
test output before the combined benchmark: MCTS demonstrably branches and
explores differently than best-of-N (different completion, different
self-critique score); the repair loop demonstrably splices the real
SyntaxError text into each retry; session history demonstrably persists
and correctly compacts across calls; the arity veto demonstrably catches
a synthetic inconsistency and correctly abstains on unparseable code.

**Benchmark harness** (`bench_ducky.py`): 10 held-out docstring ->
function-body tasks (clamp, is_palindrome, gcd, is_prime, etc.), graded by
executing the generated code against real assert statements in a
restricted, timeout-guarded namespace -- same shape as SWE-bench's "does
the patch pass tests," sized to what a 10M-param model could plausibly do
at least a little of. Harness itself verified correct first: known-correct
canned solutions score 100%, a deliberately wrong one fails, a deliberately
infinite loop times out via SIGALRM rather than hanging the run.

**Real result, run against the actual trained checkpoint: 0/10 on all
four mechanisms** (baseline, resample, MCTS-lite, repair loop) -- not a
bug, a real ceiling. Every completion across every mechanism tops out at
1-2 tokens ('return', 'if', short fragments) before abstaining; there is
no multi-line completion for any mechanism to select among, retry into, or
check consistency across. This concretely confirms the reality check given
before building any of this: reasoning scaffolding amplifies existing
generation capability, it cannot manufacture capability the base model
doesn't have. All four mechanisms are correctly built and ready -- the
remaining, real bottleneck is base-model scale and training data, the same
theme that has dominated this entire project (vocab size, corpus size,
BPTT retention all traced back to the same ceiling). Next real lever,
consistent with the Chinchilla-ratio scaling already used for the
proportional dense/hybrid comparison: meaningfully more parameters
*and* proportionally more assertion-style/function-body training data,
not more search/retry compute on top of the current checkpoint.

**Streaming BPTT + gulp-size sweep + loss-shaping: rounds 4 and 5, both
decisive, both negative.** Prior BPTT rounds (1-3, above) all used
train_bptt.py's random-restart K-chunk windows -- every training example
resets state to None at a random corpus position, so no example ever has
real information before its own window boundary that would help predict
inside it. Streaming (`train_stream_bptt.py`) fixes exactly that: batch_size
parallel sequential streams through the real corpus, state carried
continuously across the whole corpus, gradients truncated only every
`gulp` steps (Transformer-XL's segment-level recurrence, Dai et al. 2019,
arXiv:1901.02860 -- "stop gradient at the segment boundary, keep the
memory"). A genuinely different regime, not a rerun.

Swept gulp in {4, 8, 16, 32} (gradient spans 512-4096 tokens) at fixed
total compute (~8000 chunks each, so larger gulps get proportionally fewer
optimizer steps -- a real confound noted honestly, not hidden). Val loss
was monotonically worse at larger gulp (5.09 -> 5.58), consistent with
fewer steps, not necessarily meaningful about retention itself. Retention
(`eval_bptt_retention.py`, extended to horizons up to 8192 tokens -- the
whole point of testing streaming, since old BPTT was capped at 640):
**KL=0.0000 at every horizon, every gulp size, including gulp=32's
4096-token direct-gradient-flow window.** Fourth decisive negative round.

Round 5: hypothesized the real cause was the *objective*, not data/steps/
regime -- per-chunk-averaged loss lets every chunk minimize its own loss
locally, so nothing ever requires relying on carried state. Added
`--final-chunk-only-loss` to train_stream_bptt.py: score only the last
chunk of each gulp; earlier chunks get real gradient only by shaping a
useful state for it, no independent local reward. Reswept the same 4 gulp
sizes with this objective. **Still KL=0.0000 at every horizon, every gulp
size.** Fifth decisive negative round, now the most targeted test
possible -- the loss literally cannot be reduced without using the
carried state, and retention still didn't emerge.

**Conclusion, now five-for-five across corpus size, step budget,
architecture, training regime, and the objective itself: this is very
unlikely to be a training-recipe problem at all.** The consistent
remaining explanation is a capacity/gradient-flow limitation at this
model's scale (~1.86-10M params): gradient backpropagating through
4096+ steps of recurrence very plausibly vanishes to near-zero by the
time it reaches the earliest chunks, regardless of what the objective
rewards in principle -- the same well-documented failure mode that
motivated LSTMs/GRUs over vanilla RNNs historically. Not fixable by
another training-script variant at this scale; the real lever is model
scale itself (more capacity, better-conditioned recurrence), the same
theme that has dominated this entire project. Recommendation: stop
active pursuit of BPTT-induced retention at toy scale -- the
unlimited-context property remains real and structurally verified
(constant memory, linear time), just not something this size of model can
be trained to exploit.

**Major data expansion + retrain (this round): real progress, honest limits.**
Motivated by the 0/10 benchmark result and its diagnosed Chinchilla deficit
(10M params, ~387K training tokens, ~570x under the ~20-tokens/param
target). Expanded both domains substantially:
- Code: full local stdlib (565 modules) + curated, individually
  license-checked site-packages libraries (torch, sympy, scipy, jax,
  pandas, sklearn, matplotlib, numpy, networkx, PIL, etc. -- all BSD/MIT/
  Apache, none GPL), zero download. 2.8M -> 47.2M tokens.
- Text: rj + 1,255 curated Gutenberg texts (sampled from Gutenberg's real
  public catalog, not hand-listed -- 957 general + 298 specifically
  drama/dialogue, filtered by the catalog's own Subjects/Bookshelves
  metadata) + NLTK's nps_chat corpus (10,567 real chat posts). 14.7M ->
  135M tokens. Drama/chat added specifically to address a structural gap:
  plain prose teaches fluency, not conversational turn-taking.
- Vocabulary regrown 8192 -> 32768 to match the far more diverse corpus
  (verified real compression gains: 17-25% fewer tokens on ML code and
  philosophical prose specifically, the content that motivated the growth).
- Code corpus split into `corpus_core.txt` (stdlib, simple utility style)
  and `corpus_breadth.txt` (site-packages, ML-library-internals style) with
  a weighted per-example sampler (`get_weighted_code_batch`, same pattern
  as the existing `get_joint_batch`) so core's style -- the style
  `bench_ducky.py`'s held-out tasks actually test -- isn't diluted to a
  ~6.6%-by-volume rounding error by the much larger breadth pool.
- Synthetic relational data folded in as a third weighted pool, with a
  validated redundancy-pruning step: measured that 43.8%/46.3% of raw
  chains had a Python builtin (`open`, `len`, `isinstance`, ...) as the
  bridge/endpoint -- generic, low-signal chains. Filtering them out
  (`filter_generic_chains`) cut the set to 27.4% of its original size
  *and improved* held-out generalization (63.7% vs 61.1% accuracy, 0.836
  vs 0.769 mean margin) -- compressing out redundant signal measurably
  helped, exactly the hypothesis it was built to test (single seed, same
  caveat as always).

**Retrain result, both domains, xl size (11.6M params, vocab=32768):**
text best val 5.1040 (full 8000-step budget, never early-stopped) --
and, for the first time this entire project, a genuinely coherent
generated sample: *"ROMEO: Your little voice seems to me more interesting
than the scene. I know your words, and the eyes of my mind, who are
merely delightful--the heavy-lined-aged, proudly-filled man..."* -- real
grammar, real punctuation, not word-salad. code best val 3.4822
(early-stopped at step 5750/8000) -- lower than the previous
8192-vocab-generation's 5.1643 despite a 4x harder (bigger) vocab, meaning
the data expansion more than offset the harder softmax.

**Real scaling bug found and fixed while testing this:** `ducky.py`'s SDK
setup did three separate full-corpus `ast.parse` calls (`build_symbol_table`,
`build_call_graph`, `build_graph`'s AST-fact extraction) against the full
~162MB combined corpus.txt -- fine at the old, smaller scale, but now slow
enough to hit real timeout/resource limits (measured: killed after several
minutes, repeatedly). Fixed the same way as `synthetic_relations.py`'s call
graph: scoped to `corpus_core.txt` (stdlib, ~11MB) instead -- cleaner,
more idiomatic source for identifier/AST grounding anyway, not just faster.
Setup time: indefinite/killed -> 23.6s (warm) / ~177s (cold, first
tokenization pass). Token-level signals (n-gram index, co-occurrence
edges) were confirmed already fast at the new scale (12s for 39M tokens)
and stayed on the full corpus.

**Honest benchmark result: still 0/10 on all four mechanisms**, rerun
against the new xl checkpoint. Not sugarcoating the headline number. But
the *character* of the failures shifted in a way worth being precise
about: every failure before this round was an empty completion or pure
gibberish; now resample/MCTS produce plausible, contextually-sensible
partial fragments (`obj`, `words,`, `paths`, `text`, `values`) that fail
via `NameError` (an undefined variable used, not nonsense) because
generation gets cut short by calibrated abstention before completing a
valid statement -- not because the tokens themselves are wrong. Real,
measurable local-token-quality improvement, consistent with the text
domain's coherent sample. But the actual capability the benchmark
tests -- sustaining a complete, valid, multi-line function body -- still
isn't there. Don't conflate "more coherent" with "more capable": this
round's data/vocab work moved the model from incoherent to coherent-but-
short-winded, not from incoherent to capable. The next lever is still
scale (this is 11.6M params; real coding-capable models are 3-4+ orders
of magnitude larger), not another data/mechanism iteration at this size.

**Root cause of premature generation cutoff, found and fixed.** Diagnosed
by instrumenting `predict_next` step-by-step on real benchmark prompts:
on 9/10 tasks, the model's own reasonable top prediction (e.g. "if",
confidence 0.27-0.31 -- a meaningfully high probability out of 32,768
tokens) was vetoed by the slow-path disagreement check. Root cause was in
`graph.py`'s `build_cooccurrence_edges`: confidence was computed as
`min(0.99, freq/max_freq)` with `max_freq=80`, a constant calibrated for
the original ~50K-token corpus ("scaled down from swarm.md's (5, 100)").
Once the corpus grew to 47M+ tokens (~940x), this silently broke: a token
occurring ~77 times -- statistical noise at this scale -- still computed
confidence=0.9625 (near-maximum), regardless of how many OTHER
continuations that same source token had. Concretely: a hyper-common
token (433 outgoing edges) had a top edge with confidence=0.9625 but
weight=0.00000196 -- internally contradictory -- and its generic,
low-value suggestion was vetoing the model's own reasonable prediction on
9 of 10 real benchmark prompts via the disagreement-abstention rule.

Fixed by replacing the absolute-frequency clamp with the empirical
conditional probability P(tgt | src) (freq of this pair / total outgoing
frequency of src) -- scale-invariant, no magic constant needed. A token
with 433 diffuse continuations now honestly gets low confidence on any
single one, rather than a stale threshold making it look artificially
certain. Also removed the `max_freq` upper-bound filter, which had been
excluding genuinely reliable high-frequency patterns while letting noisy
borderline ones through via the broken confidence formula.

**Verified with real before/after evidence, not just the fix landing:**
before, both diagnosed prompts abstained at step 0. After, `is_prime`
generated 5+ real tokens with no abstention (`if n == 0: raise` -- a
genuinely sensible edge-case check), `clamp` reached step 4. Rerunning
the full 10-task benchmark confirmed this generalizes: baseline now
produces substantially longer, more syntactically-coherent attempts
across the board (e.g. `"if n == 0: raise ValueError('n must be a
integer')"`), not the immediate 1-token abstentions from before.

**Honest result: still 0/10 on the strict pass/fail metric.** Fixing
premature cutoff exposed a *different*, previously-invisible failure
mode: degenerate repetition loops (the same phrase repeated 8+ times
verbatim) plus smaller syntax gaps (missing colons after `if`, unclosed
string literals). This is a classic small-model autoregressive failure,
not a grounding/abstention problem -- it was always latent, just never
reachable before because generation used to stop after 1-2 tokens. Real,
verified progress on the diagnosed question (why generation cut short);
a distinct, newly-visible problem (why longer generation degenerates)
is the next one to solve, not yet attempted.

**Non-negotiable scope discipline:** Ducky's job is next-token prediction
quality first. Every grounding/abstention addition earns its place by
being cheap and checkable against something real (parse validity, a real
symbol, a real n-gram, this checkpoint's own recalibrated confidence) —
never by adding a second model, a vote, or an unverified heuristic dressed
up as intelligence. If a future addition can't point at what it's checked
against, it doesn't belong in Ducky.
