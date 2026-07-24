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

**Repetition loop: root cause found, fixed, and honestly resolved --
doesn't move the benchmark, and that itself is the real finding.**
Instrumented raw greedy generation step-by-step on the is_prime prompt: the
model's confidence in repeating an already-generated span climbs
*monotonically* with every cycle (measured at matched positions: 0.54 ->
0.82 -> 0.88 -> 0.90 for "if"->"n"; 0.16 -> 0.81 -> 0.89 -> 0.92 for
"integer"->")"), converging toward ~0.97 -- the exact self-reinforcing
mechanism Holtzman et al. 2019 (arXiv:1904.09751) describes for greedy/
deterministic decoding. By the 2nd-3rd cycle, confidence exceeds the
fast-path threshold, meaning the model bypasses grounding entirely --
nothing confidence-based can break out once it starts, since confidence
*increases* each cycle rather than decreasing.

Fixed with the standard technique (HuggingFace's `no_repeat_ngram_size`,
widely used in production decoding): `_block_repeat_ngrams` in
inference.py masks out (logit=-inf) any candidate that would recreate an
n-gram (default n=4) already generated in this context, applied before
confidence is computed so the rest of the abstention machinery naturally
reconsiders whatever's left. Verified directly: the exact verbatim loop
("if n == 0: raise ValueError('n must be a integer')" repeated 8+ times)
is gone, replaced by forced divergence ("if n == 1: raise ValueError('i
must be a" -- a genuinely different, if imperfect, continuation).

**Reran the full benchmark with the fix active: still 0/10, and this is
the complete, honest answer, not a loose end.** Without the easy
repeat-based continuation available, completions got *shorter* again
(back to fragments like "value += 1", "count_v", "self._") -- the
model's genuine confidence on truly novel content was never actually
high; the repetition loop was masking that low confidence behind a
decoding artifact that looked confident. Removing the artifact doesn't
unlock capability that wasn't there -- it just stops hiding its absence
behind a repeat loop. Concise, honest abstention is a strictly better
failure mode than a 50-word verbatim loop even though neither passes the
benchmark, so the fix is kept regardless. This closes the loop on both
"why does generation cut short" (a real, fixed calibration bug) and "why
does longer generation degenerate" (a real, fixed decoding artifact) --
both diagnosed to their actual root cause, both fixed, neither was ever
the true bottleneck. That remains base-model scale.

**Non-negotiable scope discipline:** Ducky's job is next-token prediction
quality first. Every grounding/abstention addition earns its place by
being cheap and checkable against something real (parse validity, a real
symbol, a real n-gram, this checkpoint's own recalibrated confidence) —
never by adding a second model, a vote, or an unverified heuristic dressed
up as intelligence. If a future addition can't point at what it's checked
against, it doesn't belong in Ducky.

**Architecture critique: what's exhausted vs. what's genuinely untested.**
Prompted by a direct challenge -- is the dense hybrid backbone worth it,
and is MoE actually the alternative -- applied against evidence already in
this file rather than fresh experiments.

Two things are exhausted, not worth revisiting without a scale change:
1. *Recurrence-block-type shopping* (RWKV -> Mamba-2/RetNet/Griffin/xLSTM
   as a drop-in replacement), for base-loss reasons. Hybrid beats dense
   6/6 seeds, but the margin (code 0.152±0.035 nats, rj 0.035±0.023 nats)
   is dwarfed by data-scale effects measured in this same file: code
   corpus growth alone (51K->149K tokens) moved *dense's own* baseline
   4.815->~4.06 nats, a ~0.76 nat swing -- roughly 5x bigger than the
   architecture's win over dense. At this scale, data currently dominates
   block-type choice by a wide margin; swapping recurrence flavors reads
   as real architecture research but the numbers say it's noise-level
   next to corpus size.
2. *Reopening MoE or swarm in any shape.* Both failed for the identical,
   already-diagnosed reason: routing specialization JS-divergence was
   0.0000-0.0002 at every layer on the two most different domains
   available. That's a data-diversity problem, not a router-quality
   problem -- no new MoE variant (Switch, DeepSeekMoE, soft-MoE, a
   differently-shaped swarm) fixes a corpus that doesn't have enough
   distinct token distributions for an expert to specialize on.

One honest nuance on the retention finding: RWKV's time-decay
(`time_decay`/`time_first` in `rwkv_model.py`) is a fixed, *content-
independent* per-channel parameter -- the same decay regardless of what
token is flowing through. Mamba/S6-style selective SSMs (Gu & Dao 2023,
arXiv:2312.00752) exist specifically to make decay *input-dependent*, the
exact mechanism RWKV lacks. The five-for-five "not a training-recipe
problem" conclusion above was diagnosed on RWKV specifically -- it rules
out training recipe for *this* content-independent-decay design, not for
every linear-recurrent design. Real distinction, not backbone-flavor
noise, but only load-bearing if cross-chunk retention specifically is
still a goal (see ranking below -- backlog, not next, given the cost of
testing it properly).

What's genuinely untested and worth ranking, highest leverage first:
1. **Conditional/adaptive compute gated on difficulty, not learned
   content-specialization** -- Mixture-of-Depths (Raposo et al. 2024,
   arXiv:2404.02258) or a simpler confidence-gated early-exit. Routes on a
   scalar (predicted difficulty / this checkpoint's own confidence)
   instead of requiring a router to learn "which domain is this," so it
   doesn't hit MoE's proven failure mode -- difficulty variance (some
   tokens are near-deterministic: closing brackets, common keywords;
   others wide open) exists even in a small, non-diverse corpus.
   Architecturally the most coherent option too: Ducky already computes
   calibrated per-token confidence for abstention (`grounding.py`,
   `predict_next`'s fast/slow path split); extending that same signal to
   decide how many blocks a token gets is a natural extension of an
   existing mechanism, not a second model or a bolted-on vote.
2. **Layer weight-sharing** (Universal-Transformer/ALBERT-style tied
   blocks) -- newly relevant because code's realistic zero-download data
   ceiling (~52M tokens) was just hit (the latest "Scale-up round"
   commit), while text still has Gutenberg headroom. For code
   specifically, "more data" is no longer a free lever; weight-sharing
   turns "more layers" into "more effective passes through fewer physical
   parameter sets," freeing param budget for width/rank growth at fixed
   total params. A parameter-efficiency question, distinct from
   recurrence-type.
3. **Input-selective decay (Mamba/S6-style)**, as a real test of the
   retention question specifically -- mechanistically distinct from
   RWKV's fixed decay, so not covered by the five-for-five negative
   result. Backlog, not next: needs a real associative/chunked-scan
   implementation to be trustworthy at any real sequence length (not
   another Python-loop toy version), and only pays off if long-range
   retention is still a goal -- the dominant finding across this whole
   project remains that data scale outweighs every architecture lever
   tested so far.

The comfort tax: swapping RWKV for a fancier recurrence reads as "doing
real architecture research"; wiring the existing confidence signal into a
depth-router reads as "just plumbing." The evidence says the plumbing move
has a live, untested hypothesis and a principled reason to succeed exactly
where MoE failed (no specialization required), while the glamorous move
is already shown to be noise-level. No code changes or experiments were
run for this critique -- see `tasks/todo.md`'s new backlog phase for the
ranked next steps.

**Ranked ideas #1 and #2, actually implemented and tested (idea #3 stays
backlog).** Kept to a tight memory/time budget deliberately (a concurrent
process was consuming real RAM): `rj` domain only (~150KB raw text),
`vocab_size=1024`, `"m"` size (940,800 params dense) -- same toy scale
`runs/rj_base_m` already used, no code/text corpora touched.

**Real bug found while setting this up, not fixed (out of scope, flagged
honestly):** `runs/rj_base_m` was trained against the *original*
unversioned tokenizer file (`data/tokenizer/spm.model`, predating the
vocab-versioning fix earlier in this file), not today's
`data/tokenizer/spm_1024.model` -- same vocab size, different BPE merges
(the versioned file was trained later, against a different snapshot of the
combined corpus), so different token-ID meanings. Loading that checkpoint
with today's tokenizer is silently wrong, not a crash: measured val loss
~8.6 nats (worse than the ln(1024)=6.93 random-guessing floor) instead of
the recorded 4.3746; the actual original tokenizer file reproduces 4.385,
confirming the mismatch, not a training regression. This exact fallback
pattern (`Tokenizer(vocab_size=cfg_dict.get("vocab_size", 1024))`) is also
in `ducky.py` -- meaning any pre-versioning checkpoint loaded through the
SDK today likely gets silently wrong token IDs, not an error. Not fixed
this round (broader than the two ranked ideas, touches every pre-versioning
checkpoint); routed around here by training fresh, consistently-tokenized
checkpoints (`rj_base_m_seed57`, `rj_base_m_tied_seed57`) instead of relying
on the stale one.

**Idea #1: confidence-gated conditional compute, cheapest test first
(`eval_early_exit.py`).** Zero training, zero new parameters: probes
`rj_base_m_seed57` (freshly trained for this purpose, dense 4-layer,
940,800 params, best_val=3.6623 @ step 900) by reusing its own already-
trained `ln_f`+`lm_head` at every intermediate depth ("logit lens,"
nostalgebraist 2020) instead of only at the end. Confidence and accuracy
both rise with depth as expected (layer 1: conf 0.113/acc 6.0%; layer 4:
conf 0.258/acc 23.0%). Swept exit thresholds: at 0.5, 2.7% average compute
saved for a -0.4pp accuracy cost (noise-level on 500 samples); at 0.3,
13.15% saved but a real -3.8pp accuracy cost; at 0.7/0.85, negligible
savings either way. **Honest conclusion: a zero-training probe doesn't
cleanly separate "safe to exit early" from "needs full depth" enough to
buy meaningful compute savings without a real accuracy cost, at this
checkpoint's scale.** This does not kill the idea -- CALM/Mixture-of-Depths
train the exit decision jointly with the task loss rather than reusing an
untrained-for-this-purpose head, a materially different (likely
better-calibrated) test this round deliberately didn't attempt, matching
the "cheapest test that could kill the idea, first" discipline. What this
round does establish: the mechanism needs to be trained, not just probed
post-hoc, before it's worth judging.

**Idea #2: layer weight-sharing (`tie_layers`, `model.py`/`train.py`).**
`GPTConfig.tie_layers`: one shared `Block` instance referenced `n_layer`
times in the `ModuleList`; `nn.Module.parameters()` de-duplicates by tensor
identity, verified directly (940,800 -> 345,984 params on construction,
exact arithmetic match with the expected embedding+head overhead, no
double-counting bug). Trained on `rj`/base/m, vocab=1024, seed=57,
2000-step budget/patience=2 (same convention as `rj_base_m`): **tied
best_val=3.6845 @ step 1400 vs. untied (freshly retrained, identical seed/
settings) best_val=3.6623 @ step 900 -- a real but small gap (0.0222 nats)
for a 63.2% cut in block-stack parameters.** Real, honest caveat: this is
a parameter-*count* win, not a compute win -- the tied model still runs all
4 block-applications per forward pass (identical FLOPs/token, same weights
reused), and needed *more* steps to reach its best checkpoint (1400 vs 900)
and more wall-clock (343.5s vs 144.6s), not fewer. The untested follow-up
this opens (not attempted this round): does reinvesting the freed param
budget into width recover or beat the untied baseline at the *same* total
param count -- that's the real test of "more effective passes through
fewer parameter sets," this round only tested "fewer parameters, same
width." Single-seed, `rj`-domain result (not `code`, where the actual
motivating data-ceiling problem lives) -- same caveat this file applies
everywhere else to single-seed numbers; needs a repeat-seed check and a
`code`-domain run before being treated as decided.

**Memory, measured not assumed:** peak RSS during both training runs was
~1.04GB each (`resource.getrusage().ru_maxrss`); the early-exit probe
peaked at ~751MB. Comfortably small next to the concurrent process's
~5.7GB RSS this was budgeted around.

**Token efficiency: tokenizer-fairness diagnostic + candidate comparison
(`extract_terminal_corpus.py`, `eval_tokenizer_fairness.py`,
`build_candidate_tokenizers.py`).** Follow-up to the "token efficiency"
line of the critique -- is the shared text+code BPE vocab (`tokenizer.py`)
actually fair across domains, and would a terminal-command domain make
that worse. Audio/pixel were confirmed out of scope for this question:
`codec.py`'s separate VQ-VAE codebooks + `data.py`'s offset ranges already
match current unified-multimodal-tokenizer practice (UniTok, UGen,
MM-Tokenizer all converge on modality-specific quantizers feeding one
discrete ID space), so the domination risk is specific to BPE-shared
symbolic domains (text/code/terminal).

Sourced a real, tiny, near-zero-cost domain corpus first: NL2Bash (Lin et
al. 2018, arXiv:1802.08979) -- 12,607 real bash one-liners scraped from
StackOverflow, fetched directly from GitHub (`data/terminal/
nl2bash_corpus.txt`, 574,351 chars, exact expected count, confirming a
clean fetch). Built a fertility (tokens/char) + Gini-coefficient fairness
diagnostic (`eval_tokenizer_fairness.py`, small tail slices per domain via
seek+read -- never loads `gutenberg_corpus.txt`'s 1.3GB whole) that works
against any SentencePiece `.model` file, so the same script scores both
the production tokenizer and any candidate.

**Baseline result, current production tokenizer (`spm_32768.model`,
vocab=32768), measured for the first time against terminal-command
text it has never seen:** text fertility 0.2467, code 0.2395 (comparable
to each other), terminal 0.4070 -- ~65-70% more tokens per char than
text/code. Gini across the three domains: 0.125. Confirms the diagnosed
problem directly: a domain absent from tokenizer training pays a real,
measurable compression penalty, not a hypothetical one.

**Three small candidates trained for comparison** (vocab=8192, a few MB
of input each, `data/tokenizer/candidates/`, production `spm_32768.model`
untouched) -- naive-concat BPE (today's approach, but now including
terminal data), resampled BPE (text/code/terminal upsampled/truncated to
~1.5M chars each before training), resampled Unigram (identical resampled
corpus, `model_type="unigram"`). Peak RSS for all three combined: ~209MB.

| tokenizer | vocab | text | code | terminal | Gini |
|---|---|---|---|---|---|
| spm_32768 (production) | 32768 | 0.2467 | 0.2395 | 0.4070 | 0.1250 |
| naive_bpe (small-scale, +terminal) | 8192 | 0.3113 | 0.2577 | 0.3692 | 0.0792 |
| resampled_bpe | 8192 | 0.3179 | 0.2583 | 0.3559 | 0.0698 |
| resampled_unigram | 8192 | 0.3410 | 0.2635 | 0.3659 | 0.0703 |

**Honest reading, three separate findings, not one:**
1. **The single biggest lever is simply including terminal data in
   training at all.** `naive_bpe` (small-scale, but WITH terminal data)
   already cuts Gini from 0.125 to 0.079 and terminal fertility from
   0.407 to 0.369 -- most of the "fairness" gain comes from the domain
   existing in the training mix, not from resampling. This is confounded
   with vocab size (8192 vs. production's 32768) and training-corpus size
   (a few MB vs. hundreds of MB to 1.3GB) -- not a same-scale controlled
   comparison, flagged honestly rather than overclaiming the exact
   magnitude would hold at production scale.
2. **Resampling gives a further, real but modest improvement, with a real
   trade-off.** `naive_bpe` -> `resampled_bpe`: Gini 0.0792 -> 0.0698
   (~12% relative reduction), terminal fertility improves slightly
   (0.3692 -> 0.3559), but text fertility gets *slightly worse* (0.3113 ->
   0.3179) -- resampling helps the underrepresented domains at a small,
   real cost to the previously-dominant one, exactly the mechanism
   working as intended, not a free lunch.
3. **Unigram does not help here -- a real, measured negative result, not
   an assumption.** On the *identical* resampled corpus, `resampled_unigram`
   has marginally worse fairness (Gini 0.0703 vs. 0.0698) and worse
   absolute compression on every single domain (text 0.341 vs. 0.318,
   code 0.2635 vs. 0.2583, terminal 0.3659 vs. 0.3559) than
   `resampled_bpe`. Matches this round's research finding that Unigram vs.
   BPE is domain-dependent, not universally better -- here, on Ducky's
   actual domains, BPE wins outright. Reject Unigram for this project;
   keep BPE.

**Side finding, not a new bug, a pre-existing shared characteristic:**
round-trip encode/decode was verified correct for text and terminal
samples on all three candidates, but *not* for a code sample (indentation/
newlines collapse to single spaces on decode) -- confirmed this is
identical behavior on the production `spm_32768.model` too (SentencePiece's
default `nmt_nfkc` normalizer's whitespace collapsing, not something these
candidates introduced). Not a differentiator in this comparison; a real,
separate improvement opportunity (`normalization_rule_tsv` override or
`remove_extra_whitespaces=False`) if code round-trip fidelity ever matters
for a downstream use case, not pursued this round.

**Explicit non-decision:** migrating the production tokenizer was
deliberately not attempted this round. That requires retraining every
downstream checkpoint -- the same costly migration already paid for twice
in this project (vocab 1024->8192->32768) -- and this comparison, while
directionally clear (include terminal data + resample + keep BPE, don't
switch to Unigram), used small-scale candidates specifically to stay cheap
under concurrent CPU/GPU load, not a production-ready tokenizer. A real
migration would need: the resampling ratio tuned at full corpus scale, the
production vocab size (32768) re-validated at that scale rather than
inherited from the 8192 comparison, and -- given the tokenizer-versioning
bug already flagged in `tasks/todo.md`'s Phase N -- deliberate versioning
this time so it doesn't silently break old checkpoints the way the
pre-versioning tokenizer already did.

**Real, full-corpus-scale "balanced" tokenizer built and validated
(`spm_32768_balanced.model`) -- tokenizer only, no checkpoint retraining,
by explicit user choice given the CPU/GPU contention and the ~3.8-hour
historical cost of a single production retrain (`code_base_xl_rwkv_rank64`'s
own `metrics.json`).**

**Versioning fix landed first, additively:** `tokenizer.py`'s
`Tokenizer`/`_model_prefix`/`train_if_missing` gained an optional
`variant: str = ""` -- two tokenizers can now share a `vocab_size` but
differ in training recipe (`spm_32768.model` vs.
`spm_32768_balanced.model`) without colliding on the same filename.
Default `variant=""` reproduces every existing filename exactly, verified
directly (`_model_prefix(32768)` and `_model_prefix(1024)` unchanged;
`Tokenizer(vocab_size=1024)` still loads and encodes correctly). This is
the deliberate-versioning fix the earlier stale-tokenizer bug (this
file's own tokenizer-fairness section, and `tasks/todo.md`'s Phase N)
flagged as needed the next time a new tokenizer generation is built --
landed now, not deferred again.

**Alpha picked empirically, then corrected for scale, not assumed twice
over.** Extended `build_candidate_tokenizers.py` with alpha-weighted
resampling (`weight_domain = size_domain**alpha`) and swept alpha in
{1.0, 0.5, 0.3} on the existing small samples: alpha=1.0 exactly
reproduced the earlier `naive_bpe` numbers (Gini 0.0792 -- a real
consistency check that the new alpha-weighted code path is correct, not
just plausible), alpha=0.5 gave 0.0734, alpha=0.3 gave 0.0719 -- a
monotonic small-scale improvement as alpha decreases. But small-scale
domain sizes (~4:4:1) don't reflect real ones (~2300:300:1): computed the
actual full-scale repeat count each alpha would imply for the terminal
domain -- alpha=0.3 implies repeating NL2Bash's 574KB corpus ~156x to hit
its target share (pathological, mostly-duplicate-driven merges); alpha=0.5
implies ~39x (still real upsampling of a genuinely diverse small corpus --
206 unique flags, 102 utilities -- not exact-duplicate noise at that
multiple). Chose **alpha=0.5** over the small-scale sweep's marginally
better 0.3, explicitly trading a small measured Gini difference for
avoiding a repetition regime with no precedent for being safe.

**Streaming corpus construction, not `tokenizer.py`'s existing whole-file-
read approach.** `build_production_tokenizer.py` reads every source file
(`romeo_and_juliet.txt`, `gutenberg_corpus.txt`, `chat_corpus.txt`,
`corpus_core.txt`, `corpus_breadth.txt`, `nl2bash_corpus.txt`)
line-by-line, applying a per-domain keep-probability (text: drop lines
randomly, since its alpha-weighted target is below its natural size) or
repeat-count (code, terminal: below target, so lines get rewritten
`repeat` times, capped at `MAX_REPEAT=50` regardless of what the formula
implies) -- never holds a full domain's text in one Python string, unlike
`tokenizer.py`'s current `train_if_missing`. Real computed values, not
estimated: text size=1,327,244,929 bytes, target=1,085,648,526
(keep_prob=0.818); code size=177,945,415, target=397,518,286 (repeat=2);
terminal size=575,091, target=22,598,624 (repeat=39, matching the
alpha=0.5 estimate above almost exactly). Trained with `num_threads=4`
(vs. SentencePiece's default of 16 used in the small-scale run) --
deliberately less aggressive given the concurrent CPU load.

**Real, measured cost: 1,886MB peak RSS** (`resource.getrusage`) for the
full build (streaming ~1.5GB combined corpus through SentencePiece
training) -- an order of magnitude more than the small-scale sweep's
~132-209MB, exactly as expected given the corpus size, but still small
next to available RAM and the concurrent process's own footprint.

**Full-scale validation result, the number that actually matters:**

| tokenizer | vocab | text | code | terminal | Gini |
|---|---|---|---|---|---|
| `spm_32768` (production, unchanged) | 32768 | 0.2467 | 0.2395 | 0.4070 | 0.1250 |
| `spm_32768_balanced` (new) | 32768 | 0.2474 | 0.2386 | 0.3561 | 0.0930 |

**The small-scale finding holds at real scale, and better than the small
comparison's own trade-off pattern suggested.** Terminal fertility
improved substantially (0.4070 -> 0.3561, ~12.5% relative reduction);
Gini dropped 25.6% (0.1250 -> 0.0930) -- a real, large fairness gain. Text
fertility is *unchanged within noise* (0.2467 -> 0.2474, +0.0007) rather
than measurably worse as it was in the small-scale comparison, because at
real scale text only gets trimmed to 81.8% of its 1.3GB (still enormous,
diverse remaining text) rather than forced toward rough equality with a
much smaller domain the way the small toy comparison did. Code fertility
even improved slightly (0.2395 -> 0.2386) from its 2x upweighting. This is
a cleaner win than the small-scale round predicted: at production scale,
the fairness gain came essentially for free, not at a measurable cost to
the previously-dominant domain.

**Explicit non-decision, unchanged, by the user's own choice:** no
checkpoint has been retrained on `spm_32768_balanced.model`.
`code_base_xl_rwkv_rank64`, `text_base_xl_rwkv_rank64`, and `rj_base_m*`
all still load and run on `spm_32768.model` exactly as before -- confirmed
untouched (`git status` on `data/tokenizer/spm_32768.*` shows nothing).
The new tokenizer exists, is validated at real scale, and is ready for a
future retraining decision; it is not yet in use by anything.

**Clearing the non-retraining backlog, all at toy scale (per the new
small-scale-first rule in `tasks/core_principle.md`).** User's direction:
do everything left except retraining production checkpoints.

**Real bug found and fixed: `ducky.py`'s stale-tokenizer mismatch, plus a
second, deeper one it exposed.** `Tokenizer` gained a `model_path`
override (bypasses the `spm_{vocab_size}[_variant].model` naming
convention entirely) and `ducky.py` now routes checkpoints with no
recorded `vocab_size` to the preserved original `data/tokenizer/spm.model`
instead of guessing `vocab_size=1024` against today's (different)
`spm_1024.model`. Verifying this surfaced a second, previously-invisible
bug: `data.py`'s tokenize cache was keyed only by `vocab_size`, so the
legacy `spm.model` and the retrained `spm_1024.model` -- same vocab size,
different token-ID mappings -- collided on the same cache file
(`rj_1024.pt`). First fix attempt still reproduced the broken ~8.6-nat
loss because of this collision; added `Tokenizer.cache_key` (equals
`str(vocab_size)` for every existing plain caller -- verified zero
disruption to `code`/`text` production paths, whose `cache_key` still
exactly equals their `vocab_size`) and rekeyed `data.py`'s cache on it
instead. **Verified end-to-end after both fixes:** `rj_base_m` reproduces
val loss 4.3855 (matches the recorded 4.3746) instead of 8.638;
`Ducky(domain="rj").ask(...)` runs correctly end-to-end (empty-string
abstention on this toy checkpoint is expected behavior, already documented
elsewhere in this file, not a new symptom). One real, one-time cost from
this fix, measured not assumed: the first `Ducky(domain="rj")` call after
the fix landed had to build a fresh `code_spm.pt`/`rj_spm.pt` cache (never
existed under this cache key before) -- peak RSS 19.5GB for that one cold
call, dropping to 2.3GB warm on the next call. `--tokenizer-variant` also
wired into `train.py` (mirrors `--vocab-size`) -- plumbing only, enables
but doesn't perform a future `spm_32768_balanced.model` retrain.

**`tie_layers` width-reallocation: does reinvesting the freed params into
width recover the untied baseline? No -- slightly worse, not better.**
Searched `d_model` for a tied config landing near the untied baseline's
940,800 params: `d_model=224` (`n_head=7`) gives 863,520 tied params, the
closest achievable match. Result: best_val=3.7121 (early-stopped at step
700, best@500) -- *worse* than the narrower tied variant's 3.6845 (345,984
params), despite 2.5x more parameters, and worse than untied's 3.6623.
Honest caveat: also converged faster/stopped earlier (700 steps vs
narrow-tied's 1600), so this may be undertrained relative to its own
capacity rather than a clean negative on the "width helps" hypothesis --
single seed, not a settled answer, but the naive "just widen it" version
of the idea does not work at this scale.

**`tie_layers` repeat-seed (3 seeds) + code-domain check: rj result holds,
but the domain-dependence is large and important.** Three seeds on `rj`/m,
untied vs. tied, same steps/patience:

| seed | untied best_val | tied best_val | gap |
|---|---|---|---|
| 57 | 3.6623 | 3.6845 | +0.0222 |
| 58 | 3.6284 | 3.6337 | +0.0053 |
| 59 | 3.6471 | 3.6801 | +0.0330 |

Untied wins all 3 seeds, gap mean~0.0202 (std~0.0114) -- direction
reproduces, consistent with the original single-seed finding. **Code
domain (small already-cached 640,903-token corpus, `code_1024.pt` --
predates the recent corpus expansions, reused here specifically to stay
toy-scale): untied 3.6691 vs. tied 3.8856, gap +0.2165 -- roughly 10x
larger than any `rj`-domain gap measured.** This is the load-bearing
correction to last round's finding: `tie_layers`' cost is domain-dependent,
and it costs substantially more on code -- the exact domain the idea was
originally proposed for (code's data ceiling, not `rj`'s). Plausible
reading: code likely benefits more than prose from genuinely different
per-depth transformations (syntax-level patterns early, compositional
structure later), so forcing all 4 depth-steps to share one transformation
hurts it more. **Practical takeaway, corrected from last round: weight-
sharing is not a free lunch for code specifically -- the domain it was
meant to help pays the largest cost.** Single-seed on code; a repeat-seed
code check is a natural follow-up, not done this round.

**Idea #1 for real: training the halting head beats reusing an untrained
one, clearly.** Added `GPTConfig.use_halting` (`model.py`): a per-block
`Linear(d_model, 1)` halt head per depth step (negligible params, 516
total for `n_layer=4`), trained via an auxiliary BCE loss
(`TinyGPT.halting_loss`, `HALT_AUX_WEIGHT=0.1` in `train.py`) predicting
"does this layer's own logit-lens prediction already match the final-depth
prediction" -- the identical signal `eval_early_exit.py`'s untrained probe
checked empirically, now actually learned. Trained at `rj`/m scale
(941,316 params, seed 57): best_val=3.7639, a real +0.1016-nat cost versus
the untied baseline's 3.6623 -- the auxiliary objective genuinely competes
with the primary task at this toy scale (`HALT_AUX_WEIGHT` is a real,
untuned lever, not validated at other values this round).

New `eval_halting.py` (mirrors `eval_early_exit.py`'s exact methodology --
same `THRESHOLDS`, same `n_samples=500` -- for direct comparability) shows
the trained signal is far more decisive and far more useful:

| | threshold | avg_exit_depth | compute_saved | accuracy_delta |
|---|---|---|---|---|
| untrained probe | 0.5 | 3.892 | 2.7% | -0.004 |
| untrained probe | 0.3 | 3.474 | 13.15% | -0.038 |
| **trained halting** | **0.7** | **3.42** | **14.5%** | **-0.002** |
| **trained halting** | **0.5** | **2.102** | **47.45%** | **-0.056** |

At a near-zero accuracy cost, trained halting buys ~14.5% compute savings
where the untrained probe's best near-zero-cost point only ever managed
0.75% -- roughly an order of magnitude better. At a moderate cost budget
(~5-6pp), trained halting buys 47.45% savings, far past anything the
untrained probe reached at *any* threshold (its most aggressive setting,
threshold=0.3, only reached 13.15%). **Real, positive result: the
mechanism needed to be trained, not just probed, and once trained it
clearly works at toy scale** -- confirms the prediction from last round's
inconclusive result rather than leaving it open. Real cost to weigh
against the gain: the +0.10-nat hit to the primary LM loss from training
the auxiliary objective at all.

**Idea #3 first-rung check: `SelectiveTimeMixing` (Mamba/S6-style
input-dependent decay) beats plain RWKV, a genuinely new positive result.**
New `mamba_lite.py`, isolated (mirrors `rwkv_model.py`'s own pattern):
`time_decay` becomes `-softplus(Linear(x))` (per-token, per-channel,
input-dependent) instead of `TimeMixing`'s fixed `nn.Parameter` -- the
actual mechanistic distinction the critique flagged, RWKV's decay being
content-independent regardless of what token is flowing through. Wired
into `model.py` as `GPTConfig.use_selective_decay` (only valid alongside
`use_rwkv_hybrid`). Sanity-verified first: forward/backward pass correct,
gradients reach `decay_proj`, no NaNs.

Trained matched (`rj`/m, seed 57, `attention_layers=(2,)`, same steps/
patience as every other comparison this round):

| variant | params | best_val |
|---|---|---|
| dense (untied baseline) | 940,800 | 3.6623 |
| RWKV hybrid | 941,184 | 3.5886 |
| **selective-decay hybrid** | 990,336 | **3.5635** |

Selective decay beats plain RWKV hybrid (3.5635 vs. 3.5886, a real
0.0251-nat win) and both beat dense, at toy scale, on the first try.
Honest caveat: not perfectly matched params (selective decay has 49,536
more, from `decay_proj`'s extra weights on the 3 non-attention blocks) --
a real, if modest, confound. **Explicitly not attempted, per
`tasks/core_principle.md`'s small-scale-first rule:** rebuilding the
cross-chunk BPTT training + KL-divergence retention-eval harness (deleted
in an earlier repo cleanup) to test whether selective decay actually fixes
the retention question RWKV failed five-for-five. This first-rung result
is promising enough to justify that larger investment as a deliberate next
step -- not promising enough to skip straight to it without this check,
which is exactly the point of the rule just written down.

**Scaling-law simulation tool built, and it reversed the selective-decay
verdict above -- exactly the failure mode it exists to catch.** User's top
priority: a genuine way to "simulate large scale" cheaply rather than
guess from one toy point. `src/fit_scaling_law.py` fits `L(N) = a *
N^(-alpha)` via log-log linear regression (Kaplan et al. 2020; Hoffmann et
al./Chinchilla 2022) -- verified first against a synthetic known power law
(recovers `a=10, alpha=0.3` exactly) before trusting it on real data.

**Real training use of `spm_32768_balanced.model`, without a production
retrain** (per the user's explicit choice not to retrain anything this
round): `src/run_scaling_sweep.py`, 12 configs -- 2 sizes (`xs`, `s`,
deliberately the cheapest presets) x 3 domains (`rj`, `code_core`,
`terminal` -- the last two via direct `data._tokenize_corpus` calls,
bypassing `load_lm_corpus`'s fixed domain set rather than changing it) x 2
architectures (RWKV-hybrid, selective-hybrid; dense excluded, already
well-established as losing to hybrid). `--embedding-rank 32` throughout so
vocab=32768's embedding table doesn't dwarf `xs`'s tiny block stack; the
fit uses **block-stack-only params**, not total params, as `N`, for
exactly that reason.

| domain | size | rwkv best_val | selective best_val |
|---|---|---|---|
| rj | xs | 6.7824 | 6.7875 |
| rj | s | 6.7171 | **6.7600** |
| code_core | xs | 7.1058 | **7.0876** |
| code_core | s | **7.0639** | 7.1187 |
| terminal | xs | 5.6461 | **5.6218** |
| terminal | s | **5.6093** | 5.6448 |

**The pattern that matters: selective decay leads at `xs` on 2 of 3
domains, then loses on all 3 at `s`.** Its relative edge shrinks or
reverses as capacity grows even slightly -- the opposite of what would be
needed to justify scaling it up further. Fitted extrapolation (10x past
`s`'s block-stack size) picks RWKV as the winner on **all three domains**:

| domain | rwkv alpha | selective alpha | winner |
|---|---|---|---|
| rj | 0.00545 | 0.00227 | rwkv |
| code_core | 0.00333 | **-0.00245** | rwkv |
| terminal | 0.00368 | **-0.00228** | rwkv |

Selective decay's fitted `alpha` is *negative* on both `code_core` and
`terminal` -- its own fit says loss gets *worse*, not better, with more
capacity in this range, while RWKV's stays (barely) positive on all three.

**FINAL VERDICT: promote selective decay = False.** This reverses last
round's tentative recommendation, which was based on a single (vocab=1024,
"m"-size, `rj`-only) point that happened to favor it. That's precisely
the failure mode this tool and the multi-domain decision rule exist to
catch -- and did.

**Honest limitations, not hidden:** only 2 size points per curve (a
secant slope, not a robust regression), and both are very close together
in absolute capacity (25K-158K block-stack params) against a vocab=32768
softmax whose entropy floor likely still dominates the loss at this
scale -- every fitted `alpha` is tiny (near-flat), consistent with the
project's own repeated Chinchilla-ratio theme (still deep in an
undertrained-relative-to-vocab regime, not yet in a capacity-limited one).
The *direction* of the reversal (xs favors selective, s favors RWKV, on
the identical domains) is a real, structural signal regardless of the
extrapolation's precision -- but a firm verdict at true production scale
would need `m`/`l` points too, where the block stack has enough capacity
to actually separate from the vocab floor. Not done this round, per the
same resource-conscious choice that kept this sweep to `xs`/`s`. Peak RSS
for the full 12-run sweep: 3,516.8MB.

**Explicit non-decision, unchanged:** no checkpoint was retrained this
round. The new tokenizer got its first real training use (this sweep) but
`code_base_xl_rwkv_rank64`/`text_base_xl_rwkv_rank64`/`rj_base_m*` are
still on `spm_32768.model`. Selective decay is not recommended for
promotion at this scale -- the next production retrain, whenever it
happens, should use RWKV (unchanged) + the new tokenizer, not selective
decay.

**Three follow-ups this round: halting+tokenizer combined, a genuinely new
mechanism (confidence-gated width), and selective decay restricted to
attention-adjacent layers.** All toy-scale, resource-conscious as asked;
two of the three reuse existing validated pieces rather than re-deriving
them.

**Part 1: halting under the new tokenizer.** Single-variable change from
the recorded `rj_base_m_halt_seed57` result (vocab=1024, dense,
best_val=3.7639): rerun identical (`rj`/m/dense) with `--tokenizer-variant
balanced --vocab-size 32768` (no `--embedding-rank`, so the plain tied
embedding grows to 4,194,304 params -- a real, expected cost of the
32x-bigger vocab, not a bug; total `n_params=5,004,548`). **best_val=5.7857
-- not directly comparable to 3.7639** (vocab=32768's much higher entropy
floor, consistent with the scaling sweep's own 5.6-7.1 range at this
vocab). `eval_halting.py` against the new checkpoint: `full_depth_accuracy`
craters to 0.4% (2/500 samples) -- far too sparse for the accuracy-delta
metric to mean anything at this vocab/scale, unlike the cleaner vocab=1024
result. What *does* still look sensible: per-layer confidence (0.60 ->
0.80 -> 0.78 -> 1.0) and agreement-with-final-layer (0.376 -> 0.794 ->
0.874 -> 1.0) both still rise with depth, roughly monotonically --
the halting head is still learning something real about the model's own
depth-wise confidence trajectory, even though "is the final answer
correct" is too rare an event here to validate the compute/accuracy trade
the way the vocab=1024 result did. Honest verdict: inconclusive, not
negative -- the mechanism's internal signal still looks coherent; the
validation metric itself needs more training or a bigger model to be
meaningful at vocab=32768.

**Part 2: confidence-gated width -- a new mechanism, implemented and
tested for the first time.** `WidthGatedMLP` (`model.py`): same
`Linear(d,4d) -> GELU -> Linear(4d,d)` shape as the plain MLP, but the
**second half** of the 4d hidden activations gets scaled by a per-token
gate `g=sigmoid(Linear(d_model,1)(x))` before the second projection -- the
first half is always fully active (guaranteed baseline capacity, same
idea as halting always running at least one block); `TinyGPT.
width_sparsity_loss()` reads `last_gate_mean` off every width-gated block
as a side effect (no change to `Block.forward`'s return signature, same
non-invasive pattern `MoEFFN.last_aux_loss` already uses), and
`train.py`'s new `WIDTH_SPARSITY_WEIGHT=0.05` is the only pressure pushing
gates below 1.0. Sanity-verified (forward/backward correct, gate
gradients flow) before training.

Trained matched exactly against `rj_base_m_seed57`'s baseline (`rj`, m,
dense, vocab=1024, 940,800 params): width-gated model has 941,316 params
(516 extra for 4 gate heads), **best_val=3.7242 -- a modest +0.062-nat
cost**, smaller than halting's +0.10 cost and much smaller than
`tie_layers`' code-domain cost. The sparsity regularizer worked as
intended: average gate value on held-out data is 0.163 (mostly closed,
real sparsity pressure succeeded). **But the width-axis analog of
halting's "does this correlate with real difficulty" check comes back
negative:** average gate on correctly-predicted tokens (0.1696, n=119) vs.
incorrectly-predicted tokens (0.1605, n=381) -- a 0.009 difference,
noise-level at these sample sizes, not the kind of clear separation
halting's per-layer confidence showed. **Honest read: the mechanism learns
to be sparse, but not to be selectively sparse in a way that tracks token
difficulty** -- a genuinely different (weaker) outcome than halting's
validated win, not just an untested idea anymore. Worth noting for any
future attempt: the soft multiplicative gate used here never actually
skips compute (the second half is still computed, just scaled down), so
even a working version of this mechanism wouldn't yield real FLOP savings
without a harder, indexed/sparse implementation -- a further reason this
is a weaker candidate than halting's literal early-exit.

**Part 3: selective decay restricted to attention-adjacent layers --
doesn't reverse the verdict, but is genuinely informative.** Generalized
`use_selective_decay` (all-or-nothing) with `GPTConfig.
selective_decay_layers: tuple` -- non-empty overrides the boolean with an
exact layer list; empty (default) preserves today's behavior exactly,
verified via a direct sanity check (`selective_decay_layers=(1,)` at
`n_layer=3, attention_layers=(2,)` correctly builds `TimeMixing` at layer
0, `SelectiveTimeMixing` at layer 1, attention at layer 2). Two new runs
(`rj`/s and `code_core`/s, same vocab=32768/balanced + embedding_rank=32
setup as the scaling sweep, `selective_decay_layers=(1,)` -- the one
non-attention layer directly adjacent to attention), compared directly
against that sweep's own recorded rows for the identical domain/size:

| domain | rwkv (all layers) | selective (all layers) | adjacent-only |
|---|---|---|---|
| rj | 6.7171 | 6.7600 | 6.7640 |
| code_core | 7.0639 | 7.1187 | **7.0988** |

RWKV still wins both domains. Restricting scope to the attention-adjacent
layer gives a real, if partial, recovery on `code_core` (closes about half
the gap between uniform-selective and plain RWKV) but is slightly *worse*
than uniform selective decay on `rj` (6.7640 vs. 6.7600) -- another
domain-dependent result, consistent with this whole round's theme.
**Verdict unchanged from the scaling sweep: selective decay, in any form
tested so far (uniform or attention-adjacent), does not beat plain RWKV.**
The remaining untested question is whether a more complete Mamba-2/S6
implementation (selective B/C, not just decay) would fare differently --
still open, still bigger than this round's scope.

**Ensemble check (Part A) + a grounded self-training "flywheel" test
(Part B), both genuinely small this time (`s`-size, 223,808 params, not
`m`).** User's framing: "what if multiple small models shared outputs to
build on one another" -- distinct from the already-rejected swarm/MoE
idea (that needed routing specialization that never materialized,
JS-divergence 0.0000-0.0002; a plain ensemble needs only decorrelation,
which the repeat-seed check already showed exists here).

**Part A: probability-averaging 3 fresh, differently-seeded `s`-size
models (seeds 57/58/59) beats the best single model, on both domains
tested, in different ways.**

| domain | best-single accuracy | ensemble accuracy | best-single NLL | ensemble NLL |
|---|---|---|---|---|
| `rj` | 0.212 | 0.210 (flat) | 3.817 | **3.7357** (real win) |
| `code` | 0.166 | **0.174** (real win) | 4.4661 | **4.4529** (real win) |

On `rj`, top-1 accuracy doesn't move but calibration (NLL) genuinely
improves -- the ensemble doesn't change which token wins, but its
probability estimates are measurably better, which matters more for
Ducky's own confidence-gated design than raw accuracy does. On `code`,
both metrics improve. New `eval_ensemble.py` (probability-averaging, not
logit-averaging -- the correct way to combine independently-calibrated
models) is reusable for any future N-model comparison.

**Part B: the flywheel produced a real methodological finding before it
produced a real result.** Round 0 baseline (`bench_ducky.py`, greedy,
ensemble and all 3 single members): **0/10**, matching the historical
pattern at this scale exactly. Generation+verification (5 resampled
ensemble candidates per task, temperature=0.8, "share outputs" realized
literally -- every token decision already blends all 3 models'
probabilities, not decided by one model and handed to the next): Tier 1
(real assert pass) found 0; Tier 2 (`verify_code_syntax` +
`check_call_arity_consistency`) found 6/10 -- looked like real, usable
fuel.

**It wasn't.** Reading the actual pooled text before retraining on it
(never skip this step) showed all 6 "verified" examples were comment-only
gibberish after the docstring (e.g. `# # 0 input_es = None: # Cestr_cont:
the # """ any for a...`) -- `verify_code_syntax` passed them because a `#`
line produces no AST node at all and isn't a syntax error, and
`check_call_arity_consistency` trivially reports no conflict when there
are zero function calls to check. **"Parses" was never the same claim as
"contains real code," and the gate was accidentally checking the former
while believed to be checking the latter.** Fixed with a new,
genuinely-reusable grounding signal: `grounding.has_real_statement` --
does the function body contain at least one statement beyond its own
docstring (rules out comment-degenerated and bare-`pass` "bodies").
Regenerated with the corrected gate (Tier 2 now requires
`verify_code_syntax AND has_real_statement`): **0 Tier-1, 0 Tier-2, 0
total verified fuel.** No retrain was performed -- there was nothing to
retrain on.

**Honest verdict: the flywheel has nothing to spin on at this scale, once
verification is actually checking what it claims to.** Consistent with
this project's most repeated finding -- reasoning/bootstrapping scaffolding
amplifies existing capability, it cannot manufacture capability the base
model doesn't have, and at `s`-scale (223,808 params) on code, there isn't
yet real capability to bootstrap from. This is a stronger, more honest
version of that same conclusion than the mcts_lite/repair_loop/resample
round reached, because this time the verification gate that would have
said otherwise was caught and fixed before being trusted, not after.
**Net result of this whole round: ensembling (Part A) is a real, modest,
reusable win; a self-training flywheel (Part B) is not viable yet at this
scale, and the honest reason why is now backed by a corrected grounding
signal available for future use.**

**Trained code's model to its Chinchilla-matched "minimum viable" size --
real data, real tokenizer, real result.** Direct follow-up to "how does
the flywheel scale": the identified prerequisite was a real model past
zero capability, reached by matching params to the *actual* data budget,
computed rather than guessed, and done safely under an explicit
resource constraint.

**Safe tokenization first, given a real prior close call.** This session
already hit a ~19.5GB transient RSS spike once, tokenizing a large corpus
under a new tokenizer identity for the first time in one `encode()` call.
`corpus_breadth.txt` (167,234,835 bytes) had never been tokenized under
`spm_32768_balanced.model` -- new `safe_tokenize_breadth.py` streams it in
~10MB line-bounded chunks instead, encoding each separately and
concatenating token-id lists. **Result: peak RSS 2,917.9MB** -- an order
of magnitude safer than the earlier spike, confirming the chunked
approach directly rather than assuming it would help.

**Real numbers, not estimates, drove the config.** Combined real code
corpus (`corpus_core.txt` + `corpus_breadth.txt`) under the balanced
tokenizer: **43,263,940 tokens** (2,528,834 + 40,735,106) -- close to the
~52M-token ceiling already recorded elsewhere in this file. Chinchilla-
optimal (20 tokens/param): **2,163,197 params**. Searched real
`GPTConfig`s (not guessed) for a close match:
**d_model=128, n_layer=6, n_head=4, embedding_rank=32, RWKV hybrid,
attention_layers=(5,)** -> **2,259,584 total params** (within 4.5% of
the target), same `d_model=128` shape as every `"m"`-family toy
comparison this session, just deeper and with a wider factored-embedding
rank for the real 32768 vocab.

**Trained one model** (not an ensemble -- this round was about size, not
re-litigating Part A), `--dataset code --tokenizer-variant balanced`,
weighted `code_core`/`code_breadth` sampling (existing
`load_weighted_code_corpus`, same as every production code run), step
ceiling 11,000 with patience=6. **Took ~2.3 hours wall-clock -- longer
than the ~25-50 minute estimate**, because the estimate was extrapolated
from toy `"s"`/`"m"`-size dense/small-vocab runs and under-weighted RWKV
hybrid's real per-step recurrence overhead (already documented elsewhere
in this file as 2-3x dense's per-step cost) compounding with the bigger
vocab/embedding-rank compute at this size. Peak RSS stayed healthy and
stable throughout (checked directly at multiple points: 2.75GB -> 2.3GB
-> 2.85GB -> 1.75GB, normal fluctuation, no growth trend) -- the resource
constraint was respected in practice, not just in the plan.

**Result: n_params=2,259,584, best_val=3.6779, best_step=11000 (the full
ceiling -- patience never triggered, meaning it was still improving when
the budget ran out, an honest caveat: this may not be the true best
achievable within this data/param budget, just the best found within this
step budget).** For comparison: the toy vocab=32768 scaling-sweep points
(`xs`/`s`, 25,440-158,272 block-stack params, same tokenizer) scored
5.6-7.1 on the same kind of loss. **This model, with meaningfully more
(but still Chinchilla-modest) capacity matched to the real data volume,
scores 3.6779** -- a dramatic improvement, direct, measured evidence for
the Chinchilla-ratio thesis this whole project has repeated: matching
params to data moves the needle far more than any architecture change
tested this session.

**`bench_ducky.py`: still 0/10, but a real, qualitative step up in
*how* it fails.** New `eval_chinchilla_min.py` (greedy decode,
no-repeat-ngram blocking, same discipline as every other generation path
this project uses). Every completion is now built from genuinely
plausible Python idioms -- `isinstance(n, (int, int))`-style type checks,
`raise ValueError(f"Expected {x}...")`-style error handling, `np.array`/
`np.sqrt`-style calls -- not gibberish, not single-token fragments, not
comment noise. What fails: malformed f-strings (unterminated), unbalanced
brackets, and generation drifting past the intended function's natural
end into unrelated new `def`s. **This is the same "coherent but not yet
capable" pattern already documented in this file for the much bigger
production xl checkpoint (11.6M+ params) -- reached here at 2,259,584
params, roughly 5x fewer**, via matching size to real data rather than
brute-force scale. Genuine, measured evidence that Chinchilla-matching is
a more efficient path to fluency than just training a bigger
arbitrarily-sized model on the same data.

**Honest framing, not overclaimed:** this is still 0/10 on the strict
pass/fail metric this project has always reported honestly, and
`max_new_tokens=48` cutting generation off mid-structure is a real,
uncontrolled confound in some of these completions (some "drift into a
new def" failures may be generation running past where a real function
would have naturally ended, not a distinct failure mode). Not retested at
a longer generation budget this round -- a cheap, natural follow-up.

**Follow-up round: a self-distillation "dreaming" phase (real negative,
directly confirming a named risk) + 3 actionable weakness fixes.**

**Part A: self-distillation dreaming made things worse, and probably for
the reason the plan warned about before running it.** New
`run_dreaming.py`: a frozen teacher (last round's Chinchilla-matched
checkpoint) generates 30 "dreamed" continuations (temperature=0.8,
no-repeat-ngram blocking) from real 40-token seed contexts, and a student
copy is trained via KL-divergence to match the teacher's own *softened*
(temperature=2.0, Hinton et al. 2015) distribution on those dreams --
Furlanello et al. 2018's Born-Again Networks pattern, adapted to use
self-generated rather than real re-labeled data. No hard pseudo-labels,
no verification gate needed (unlike the flywheel), since the target is
the teacher's own smoothed belief, not a ground-truth claim.

**Result: held-out loss got measurably *worse* (4.0813 -> 4.2888, +0.2075
nats), `bench_ducky.py` stayed 0/10 -> 0/10.** Plausible, specific cause:
400 distillation steps drawn from only 30 dream sequences means each
sequence was revisited ~107 times on average (400*8/30) -- the student
almost certainly overfit to a narrow, repetitive, self-generated set
rather than genuinely "consolidating" anything. This is a small-scale,
direct confirmation of exactly the risk this file's own architecture-
critique section named before this experiment ran (Shumailov et al. 2024,
"The Curse of Recursion," arXiv:2305.17493) -- even one modest round of
self-training on a narrow self-generated set measurably degrades general
performance, not just at the large scales that paper studied. **Honest
verdict: this specific dreaming design doesn't work at this scale/
configuration.** A natural, cheap follow-up (not done this round): far
more dream diversity (hundreds, not 30) relative to distillation steps,
so no single dreamed sequence dominates training -- untested, not assumed
to fix it.

**Part B2: stopping criterion for generation -- implemented correctly,
and it directly falsified last round's "over-generation" theory.** New
`grounding.is_complete_statement` (parses + a blank line just emitted --
neither alone is a real signal, together they're a checkable proxy for
"this looks done"), wired into `generate_with_grounding` as
`stop_when_complete` (default on). Tested two ways: through `Ducky.ask()`'s
real confidence-gated path, abstention already cuts every generation
short (4-40 tokens) before the new check ever gets a chance to fire.
Through the plain greedy decode that originally showed the "drift into an
unrelated `def`" pattern: **`stopped_complete=False` on all 10
`bench_ducky.py` tasks** -- the model never reaches a valid-parse-plus-
blank-line state within the 48-token budget. This directly rejects the
hypothesis that some 0/10 failures were near-misses corrupted by
over-generation: the completions are syntactically broken throughout,
not valid-but-overrun. A real, useful, correctly-implemented mechanism
that isn't the fix for the current gap -- confirms the gap is genuinely
about capability, stated more precisely than before.

**Part B3: grounding-signal audit -- one function already honestly
scoped, one had a real, previously-unnoticed gap.** Tested both remaining
signals against real adversarial input, not just reasoned about them.
`check_call_arity_consistency`: correctly returns "consistent" for both
the comment-garbage text and any code with zero function calls -- but its
own docstring already discloses this ("can only flag inconsistency, never
confirm correctness") -- not a hidden bug, an honestly-scoped limitation.
`identifier_grounded`: tested against the *real* symbol table (22,537
identifiers from `corpus_core.txt`) and found single-letter identifiers
(`x`, `a`, `n`, ...) present near-universally -- `identifier_grounded(
"return x", real_symtable)` was `True` regardless of whether anything
meaningful was verified, while a genuinely novel identifier
(`flatten_one_level`) correctly failed. **Fixed with a `min_length=3`
threshold** (verified: single letters now correctly return `False`,
specific identifiers unaffected) -- the check is well-calibrated for
specific/rare identifiers and was silently meaningless for generic short
ones.

**Part B4: two real bugs fixed, both blocking already-validated work from
being usable through the actual SDK, not new features.**
1. `Ducky.__init__` read `cfg_dict["vocab_size"]` but never
   `cfg_dict.get("tokenizer_variant", "")` -- meant
   `Ducky(run_name="code_base_chinchilla_min_rwkv_rank32_tokbalanced")`
   would have silently loaded the *wrong* tokenizer (`spm_32768.model`
   instead of `spm_32768_balanced.model`), the identical bug class already
   fixed once this session for the legacy pre-versioning case. Fixed;
   verified end-to-end (`d.tok.cache_key == "32768_balanced"`, real `ask()`
   output). Also registered `"chinchilla_min"` as a permanent `SIZES`
   preset in `train.py` (was only a runtime monkey-patch before) and
   corrected that file's own now-outdated comment claiming `"xl"` was
   "already reasonably matched" to code's real data ceiling -- it isn't,
   by about 5x.
2. **`EnsembleModel`** (new class, `ducky.py`): wraps N loaded checkpoints
   behind the identical `model(idx) -> (logits, extra_logits, aux_loss,
   new_states)` call contract `TinyGPT` presents, averaging softmax
   probabilities (`eval_ensemble.py`'s validated approach) before
   returning log-probs as "logits" -- every existing caller
   (`predict_next`, `self_critique_score`, `calibrate_thresholds`,
   `add_model_prediction_edges`, and therefore every `generate_with_*`
   function) needed zero changes, since they only ever call `model(idx)`
   and read `model.cfg.block_size`. New `Ducky.__init__(ensemble_run_names=
   [...])` param. Verified end-to-end with 3 real seed-varied checkpoints:
   `EnsembleModel` built correctly, `.ask()` worked through the full
   graph/threshold/generation pipeline unchanged.

## Ducky is a mimicking device: calculator-grounded arithmetic + a
## step-sequencer alternative, tested honestly

User's framing, stated directly: Ducky mimics, it doesn't compute. Correct,
and expected -- TinyGPT is cross-entropy-trained next-token prediction,
nothing in that objective rewards "get the arithmetic right," only
"produce a plausible continuation." The ask: keep the mimicry for
reasoning *structure* (which operation, in what order), make the actual
values real. Investigated `/home/redleadr/workspace/uchi` (the separate,
more mature project this repo already ports pieces from) before writing
any code, per the user's explicit "nail the concept first":

- `uchi/predictor.py`'s `UniversalPredictor` -- a non-neural, trie-based
  Credibility-Weighted Context Tree (CTW-style, multiplicative-weights
  credibility updates) -- is almost certainly "uchi's old sequence
  predictor." Genuinely different mechanism from TinyGPT (interpretable,
  auditable), but still fundamentally a predictor: it recalls/blends
  patterns already seen, it computes nothing.
- **uchi's own codebase already tested and rejected this exact predictor
  for numeric-value judgments, in writing** (`uchi/numeric_plausibility.py`'s
  docstring): tried reusing it to judge whether a claimed number was
  plausible; failed, since a static/scattered pool of numeric facts has no
  temporal order for a sequence predictor to exploit (55 and 50,000 scored
  equally "surprising"). Real, load-bearing precedent carried into this
  round: this predictor class must never be the thing producing or
  validating a numeric value.
- **uchi's actual "not mimicry" mechanism is `tool_calling.py` +
  `scratchpad.py`**: a `<|tool_call|> run_python(...)` grammar hands real
  code to a subprocess-sandboxed interpreter and splices genuine stdout
  back into the text -- the PAL (Gao et al. 2022, arXiv:2211.10435) /
  Toolformer (Schick et al. 2023, arXiv:2302.04761) pattern: the model
  decides *what*/*when* to compute, never emits the digits itself.

User's confirmed scope: pure numeric expressions only (not word problems
-- no corpus exists, and prose-to-operand parsing is a separate, harder,
unscoped capability); `UniversalPredictor` gets one cheap test as an
alternative step-*sequencer* only, never as a value-producer.

### Part A: calculator-grounded generation -- real, strong, and honestly bounded

**Mechanism** (`grounding.py`, `inference.py`), directly extending Ducky's
own established "verify against something real, abstain otherwise"
discipline to a new domain, no `model.py` changes at all:
- `evaluate_arithmetic(expr)`: a restricted AST walker (whitelist
  `BinOp`/`UnaryOp`/numeric `Constant`, `+-*/**`) -- never `eval()`.
  Verified it refuses every disallowed construct tried (`__import__(...)`,
  bare names, calls, attributes, `bool`-as-`int`) by returning `None`, not
  executing anything; verified correct on real precedence (`12 + 5*3 - 4`
  -> 23, matches Python's own `eval`).
- `find_arithmetic_expression`/`arithmetic_grounded`: detect a
  just-completed literal expression right before `=`, and a post-hoc
  claim-vs-real check, respectively -- both abstain (`None`) when
  inapplicable, same convention as `check_call_arity_consistency`.
- `generate_with_calculator` (`inference.py`): identical token-by-token
  loop to `generate_with_grounding` (reuses `predict_next`, `SessionTrie`,
  no-repeat-ngram blocking -- zero duplicated logic), but every time
  generation completes an expression right before `=`, the model's own
  digit prediction for the result is skipped entirely: the real value is
  computed and its tokens are spliced in directly. Records
  `model_would_have_generated` (a non-deciding peek, for honest
  before/after comparison) and `splice_verified` (a re-decode self-check
  for BPE-boundary safety) on every splice.

**Real bug found and fixed while testing, not after**: the first version
only checked for the trigger *after* a freshly-generated token, so a
prompt that already ends in "expr =" (e.g. a direct query, or this round's
own benchmark prompts) could have its intervention point silently skipped
-- if the tokenizer fuses "=" together with the start of the answer into
one BPE step, the moment right after a bare "=" never occurs as its own
check point. Confirmed empirically (not assumed): several benchmark rows
showed identical plain/calculator output with `n_splices: 0` despite the
prompt containing a clean expression. Fixed with one additional check
against the raw prompt before the generation loop starts at all.

**New toy checkpoint** (`train_arithmetic.py`, `runs/arithmetic_base_s_rwkv`):
new synthetic, real-by-construction corpus (`generate_arithmetic_corpus.py`
-- 4000 step-by-step reduction traces, real operator precedence,
cross-validated against Python's own `eval()` on the exact generated
expression, not just asserted correct). `s`-size RWKV hybrid,
223,936 params, best_val=0.6130 at step 2600, **peak RSS 1,150MB**. The
raw sample from training is a perfect, unprompted illustration of the
exact problem this round targets: `'Step 1: 1 * 3 = 3 Step 2: 3 * 18 = 62
Step 3: 62 + 8 = 82 ...'` -- flawless step-trace *format*, wrong
arithmetic on every multi-digit step (`3*18` isn't 62, it's 54).

**Benchmark** (`bench_arithmetic.py`, 20 fresh held-out single-expression
tasks, seed distinct from training): **plain (mimicry-only) generation:
45% correct. Calculator-grounded generation: 100% correct.** A real,
complete fix for the well-posed scope this round committed to.

**Honest boundary, found (not hidden) while building the benchmark**: the
mechanism guarantees every *individual detected expression* is computed
correctly -- it does not make the model correctly *carry a prior real
result forward* as the next step's operand in a freely-generated
multi-step chain. Manually observed before narrowing the benchmark's
scope: given only "Step 1:" (no fixed expression), the model invented its
own operands throughout; the calculator correctly computed whatever
expression appeared at each step, but a later step's operands sometimes
had no relation to the previous step's real (correctly-spliced) answer.
This is a distinct, harder capability (something like "attend to and
reuse your own prior real output"), honestly out of scope this round --
the benchmark was deliberately scoped to single, well-posed expressions
precisely because multi-step chains in this corpus have independently-
random later operands no model could predict from context anyway.

### Part B: `UniversalPredictor` step-sequencer test -- inconclusive-to-negative, reported honestly

Ported `uchi/predictor.py` verbatim into `src/sequence_predictor.py`
(dependency-clean, stdlib-only, attributed) per the user's explicit
request, scoped by the uchi precedent above: tested only as an
alternative to Ducky's own generation for predicting *which operation
comes next* in a reduction trace (`ADD`/`SUB`/`MUL`/`DIV`) -- a genuinely
sequential/categorical question, never a numeric value.

Real structure exists in this task: operator precedence always groups
all `*`/`/` before all `+`/`-` regardless of the original random draw
order, so it's not pure noise. `eval_step_sequencer.py` compared
`UniversalPredictor` against a tiny freshly-trained Ducky model
(6,736 params) on the identical data/split (3600 train / 400 val
sequences, from `generate_arithmetic_corpus.py`'s own op-order labels),
**and, critically, against a majority-class ("always guess ADD")
baseline** -- essential context, not decoration, since precedence
grouping alone makes later positions disproportionately ADD/SUB.

**Results: majority-class baseline 43.6% (283/649). `UniversalPredictor`
40.4% (262/649) -- *below* the trivial baseline. Tiny Ducky 44.8%
(291/649) -- only +1.2 points above the trivial baseline.** Neither
predictor demonstrates real sequential learning beyond what a constant
prediction already captures on this specific task; `UniversalPredictor`
does measurably worse than guessing. Plausible, stated honestly rather
than explained away: sequences here are short (1-4 steps) and each is its
own fresh `.history` context, leaving little room for genuine
cross-position sequential signal to accumulate before the group-boundary
structure is already absorbed into the marginal (position-conditioned)
distribution a trivial baseline exploits for free. **Verdict: this
specific test doesn't show a case for `UniversalPredictor` over Ducky's
own generation as a step-sequencer** -- a longer/deeper chain task might
give a genuinely sequential predictor more room to show an edge, untested
here, not assumed to change the outcome.

## Three weaknesses, tested small-scale-first -- a new standing rule

Follow-up to "what's preventing Ducky from performing exceptionally
well" (four weaknesses ranked: scale/data, no internally-grounded
computation, RWKV retention unexploited, self-improvement doesn't
substitute for data). **New standing rule, elevated to a hard
requirement, not just this round's scope** (saved to memory as
`no_scaleup_without_proof`): never commit to expensive/large training to
test whether the architecture can produce good results -- always prove
cheaply, at small scale with representative/diverse data, that the
architecture (not lack of scale) is the real ceiling, before any
scale-up is even proposed. Skipped weakness #1 (scale) this round by
explicit instruction; did the other three.

**#4 (self-improvement) re-evaluated, not re-attempted.** Both dreaming
and the flywheel were already toy-scale, honest, diagnosed negatives.
The seemingly-new angle -- Part A's calculator-grounding now makes
verified-correct arithmetic fuel abundant -- turned out not to add new
information on inspection: `generate_arithmetic_corpus.py`'s training
data was already 100% correct by construction, yet the base model still
only reached 45% plain-mode accuracy (see the arithmetic section above)
-- that gap is data-efficiency/capacity (weakness #1 again), not
something self-training on verified data could fix. Folded its one
genuinely new angle (operand-chain coherence) into the grounded-
computation work below instead of re-running self-training a third time
without a new hypothesis.

### Part 1: real execution-based grounding for code

`grounding.verify_code_syntax` only checked `ast.parse` validity --
"parses" isn't "runs without error." Moved `bench_ducky.py`'s already-
validated sandboxed-execution primitive (`run_task`: restricted-
builtins `exec()`, SIGALRM timeout) into `grounding.run_sandboxed`
(behavior-preserving generalization -- `extra_statements` replaces the
hardcoded `asserts` param) so it's reusable as a generation-time signal,
not just this one benchmark's grading mechanism. New
`grounding.executes_without_error(code)`: `None` if it doesn't parse,
`True`/`False` (did it raise) otherwise -- the honest middle rung
between syntax validity and full assert-based grading (no assertions
exist at generation time). Wired into `generate_with_grounding`'s
code-domain result dict as `executes`, alongside `syntax_valid`.

**Verified, not assumed**: known-good code -> `True`, code that raises
-> `False`, syntax-invalid -> `None`, infinite loop -> `False` (times
out at the configured `timeout_s`) -- all four cases distinguished
correctly. **Regression check on the refactor itself**: re-ran
`bench_ducky.py`'s original validation criteria (canned-correct
solutions for all 10 tasks, a wrong answer, an infinite loop) --
**10/10, fail, timeout, byte-for-byte identical to Phase L's original
numbers.** Reran the full benchmark against the real Chinchilla-min
checkpoint: **0/10, unchanged** from the historical record -- the
refactor changed nothing about behavior, only where the safety
primitive lives.

### Part 2: operand-chain coherence for arithmetic

Part A's own honestly-flagged boundary: calculator-splicing guarantees
each *individual* expression is computed correctly, but the base model
was never taught to carry a real prior result forward as its own next
operand, because the original corpus draws every operand independently
-- there's no chain-continuity signal in the training data at all. New
`generate_arithmetic_corpus.make_chained_expression`: each step's FIRST
operand IS the previous step's real result (second operand freshly
random) -- verified by direct construction check (chain continuity +
arithmetic correctness cross-checked programmatically across multiple
generated chains, including division). New corpus
(`data/arithmetic/chained_*.txt`, 4000 examples) + one toy training run
(`runs/arithmetic_chained_base_s_rwkv`, 223,936 params, best_val=0.5337
at step 3000 -- ran the full budget, patience never triggered, unlike
the original run's 2600-step early stop -- peak RSS 1,135.5MB).

New `bench_arithmetic.bench_chain_coherence`: measures **continuity**
(does a freely-generated next step's first operand match the real prior
result) and **final-answer self-consistency** (does the stated
"Answer: N" match the model's own last real computed step) -- both
well-posed regardless of which operations the model freely chooses,
unlike trying to match one predetermined multi-step chain (not
well-posed here, since later operands are inherently unpredictable from
context by the corpus's own random-draw design).

**Real methodological bug found and fixed before trusting the first
result**: there's no stopping criterion for this domain (unlike code's
`is_complete_statement`), so generation regularly ran past "Answer: N"
into an unrelated new trace within the same token budget -- the first
version of the analysis compared a trace's stated answer against a
*later, unrelated* trace's last step, producing a nonsensical result
(chained checkpoint looked catastrophically worse than it was). Fixed
by scoping step extraction to the text up to and including the first
"Answer:" only.

**Corrected result: original checkpoint -- continuity 90.5% (42
checked), final-answer self-consistency 89.5% (19 checked). Chained-
trained checkpoint -- continuity 93.75% (32 checked, +3.25pp, likely
within noise at this sample size), final-answer self-consistency 65.0%
(20 checked, -24.5pp).** Doesn't confirm the hypothesis: continuity was
already fairly high in the ORIGINAL checkpoint without any chain-aware
training data at all (plausibly a generic "copy the most recently
written number" pattern learnable from the shared step-trace text
layout regardless of whether the corpus enforces true chain semantics),
and the chained corpus's self-consistency was measurably *worse*, not
better -- an honest, real, negative-leaning result, not the fix Part A's
flagged gap seemed to call for. Not chased further with additional
training variants this round (would risk becoming unbounded
re-attempts without a new hypothesis, the same discipline that ruled
out re-trying self-improvement above).

### Part 3: RWKV retention via a synthetic associative-recall task -- a sharp capacity cliff, not a data-pressure problem

Five prior BPTT tests on real corpora (rj, code -- Phase M) were all
negative, diagnosed as "natural data never forces long-range
dependency." Per the new standing rule: test with data specifically
*engineered* to require retention, at toy scale, before assuming real-
corpus scale-up is the missing ingredient -- the standard SSM-literature
diagnostic (associative recall / induction heads, e.g. Gu & Dao 2023's
own Mamba diagnostics), not improvised.

New `generate_recall_corpus.py`: custom small integer vocabulary (no
BPE), sequences `[filler]*L, KEY, VALUE, [filler]*L, QUERY -> predict
VALUE`. 4-condition sweep (pure RWKV / hybrid x short L=10 / long
L=100): **every single condition landed at or below chance (10%, 10
possible VALUE tokens)** -- pure_rwkv/short 0.100, pure_rwkv/long 0.085,
hybrid/short 0.100, hybrid/long 0.085. n_params 39,936-45,728 (peak RSS
1,511MB for the sweep).

**A uniformly-at-chance result across every condition, including
"short," was suspicious enough to distrust before writing it up --
distrusting your own suspicious-looking negative result is exactly
this project's own established discipline.** Three real controls run
before trusting the finding:
1. **Loss-trajectory check** (3000 steps, 2x the sweep's budget, same
   config): loss stayed completely flat (~2.59) the entire time, with
   `recall_accuracy` bouncing at chance (7-14%) throughout -- a genuine
   plateau, not a slow-but-positive trend that just needed
   extrapolation.
2. **Width check** (d_model 32 -> 64, still toy scale): identical flat
   plateau, ruling out "just too narrow."
3. **L=0 control** (KEY and QUERY adjacent, zero gap -- the copy task
   with no recall challenge at all): **100% accuracy within 200 steps.**
   This is the decisive one -- it proves the task, training loop, loss,
   and eval logic are all correctly implemented and the mechanism *can*
   learn the lookup in principle. The L=10/L=100 chance-level results
   are real, not a setup bug.

**Cliff precisely located**: swept L in {0,1,2,3,5,10} (hybrid,
d_model=32, 1500 steps each). **Perfect (100%) at L=0,1,2. Collapses to
chance (13%, 8.5%, 10%) at L=3,5,10.** A sharp cliff between 2 and 3
intervening filler tokens, not a gradual falloff -- and identical
regardless of architecture (pure RWKV and hybrid failed identically at
L=10/100 in the original sweep) or width (32 vs. 64).

**This meaningfully sharpens, not just repeats, the Phase M finding.**
The earlier "5-for-5 negative, diagnosed as no training pressure on
natural data" story could always be read as "maybe natural data just
never needed it." This round's data was built specifically so the task
*cannot* be solved without retention -- and it still fails almost
immediately, at a hard capacity wall around 2-3 tokens, regardless of
width or architecture variant. That's real, controlled evidence pointing
toward toy-scale capacity itself (not merely lack of data pressure)
being the bottleneck -- which is exactly the kind of finding that would
justify testing at real scale next, reached cheaply and diagnostically
first, per the new standing rule, not assumed or jumped to.

## Phase AA — Cashing in the proof: real production checkpoints, code and text, on GPU
Direct follow-through on Phase Z's verdict: RWKV-hybrid validated across
4 real size points, now used for an actual production-scale commitment
(user's explicit choice, "let's do option 1, go ahead" after being shown
the alternatives plainly). `train.py` itself had zero device placement
(same gap as `run_scaling_sweep.py` before Phase Z) -- added `--device`
autodetect + `.to(device)` at every tensor site (batches, prompts,
jepa/joint paths), plus `.cpu()` on both checkpoint saves so they stay
portable regardless of training device.

**Code**: re-ran the existing Chinchilla-matched `chinchilla_min` config
(128d/6L, 2.26M params, matched to the real 43,263,940-token corpus) with
the recipe Phase X validated but never applied to a full run
(`--nanogpt-recipe --lr-schedule plateau`). First attempt (best_val
3.7559) was **worse** than the old pre-recipe checkpoint (3.6779) --
diagnosed, not accepted: `plateau_stall` and the early-stopping
`patience_counter` share the same clock and `patience_counter` never
resets when a decay fires, so with this project's own defaults
(`--plateau-patience 5` vs `--patience 6`) early stopping fires ~1
checkpoint after every decay, before the new LR can possibly help.
**Fixed** (`patience_counter = 0` alongside `plateau_stall = 0` when a
decay actually fires) and re-ran: **best_val 3.6423** at step 11500 (two
real decays, 567.6s total on GPU vs. ~2.3 hours on CPU for the original) --
now genuinely beats the old checkpoint, confirming Phase X's recipe
prediction was right all along; the bug had been masking it.

**Text**: no valid production checkpoint existed at all --
`text_base_xxl_rwkv_rank96`'s `train.log` was a 0-byte file, no
`config.json`. Root cause found before touching anything: `train.py`'s
`_load_text_domain` concatenates rj + gutenberg_corpus.txt (grown to
~1.3GB by the earlier Gutenberg expansion) + chat, then calls
`_tokenize_corpus` with **no chunking** -- the same single-giant-`encode()`
pattern that spiked to ~19.5GB peak RSS on a 167MB corpus
(`safe_tokenize_breadth.py`'s own documented incident). At ~8x that
corpus size on a 39GB-RAM machine, this would OOM-kill near-instantly --
exactly matching the observed empty log. New `safe_tokenize_text.py`
mirrors the breadth-corpus fix (streamed, line-bounded ~10MB chunks) but
improves on it: per-chunk tensors concatenated via `torch.cat` at the end
instead of one flat growing `list[int]`, since a ~300M-token Python list
of ints (~30-40 bytes/token overhead) would itself have risked ~10GB+
just in list overhead. **Result: 305,451,284 real tokens, peak RSS only
5,635MB** -- safely tokenized for the first time.

Computed (not guessed) Chinchilla-optimal size for that real count
(305,451,284 / 20 = 15,272,564) and searched real configs: `d_model=320,
n_layer=10, n_head=8, embedding_rank=80` -> 15,021,120 params, within 2%.
Registered as `chinchilla_text` in `train.py`'s `SIZES` -- supersedes
`xxl`'s ~428M-token projection, which turned out to over-estimate the
real post-expansion count (28.08M params would have been ~1.8x
over-parameterized for what the corpus actually contains).

**GPU contention discovered mid-launch**: a real, independent, currently-
running production job (`uchi.flux.react_warmup_train`, part of the
actual `uchi` project, 2+ hours elapsed) was using 8.1GB/12.2GB VRAM --
the earlier "GPU is free" reading (826MiB, 5% util) had caught a lull
between phases of that same job, not a genuinely idle GPU. The full-batch
(32) `chinchilla_text` config OOM'd against it. User's explicit choice:
shrink the micro-batch and use gradient accumulation to keep the same
effective batch size, rather than wait or fall back to CPU. Added
`--grad-accum-steps` to `train.py` (default 1, zero behavior change for
every existing run) -- `--batch-size 8 --grad-accum-steps 4` reproduces
the original effective batch of 32 while fitting in the ~3GB actually
free, verified via smoke test before the real launch (steady-state
0.24s/step once past a ~250s one-time compile cost for the new shape).

**Result: best_val 5.0030** at step 7,500 (of a 75,000-step ceiling,
patience=6 stopped it there; 2,278.9s ≈ 38 minutes total, not the ~5-hour
worst case estimated before the run actually converged early) -- beats
the old, differently-sized `text_base_xl_rwkv_rank64` checkpoint (5.1040),
the first complete, real, Chinchilla-matched text checkpoint this project
has produced. Honest caveat, not smoothed over: `eval_step` reuses
`args.batch_size` (now 8, reduced for the grad-accum fix) for validation
too, so this run's val-loss estimate is averaged over 5x8=40 samples
instead of the usual 5x32=160 -- a real, noisier measurement than the
code run's, not a wrong one. Decoupling eval batch size from the training
micro-batch is a flagged follow-up, not done this round.

**Wired into the SDK and verified for real** (user's explicit request,
matching Phase X's own precedent of never trusting a promoted default
until it's exercised through `Ducky()` itself): `ducky.py`'s
`DEFAULT_RUNS` updated to both new checkpoints. `Ducky(domain="code")`
verified clean. `Ducky(domain="text")` found a real, serious bug on
first use, not a cosmetic one: `build_ngram_index` (one Python tuple per
token position, in a `set`) and `build_cooccurrence_edges` (called via
`build_graph`) were only ever measured safe at code's ~43M-token scale --
nobody had re-checked them since the "text" domain's real corpus grew to
305M tokens. A live call spiked past 25GB RSS and had to be killed by
hand before it OOM'd the whole machine (39GB RAM) and risked taking the
concurrent `uchi.flux.react_warmup_train` GPU job down with it. Fixed by
capping both `rj_ids`/`code_ids` to the same already-proven-safe order of
magnitude (50M tokens) before they reach either structure -- verified
safe on re-run (`text ask(): 'The'`, no crash, RSS stayed well under the
15GB watchdog threshold this time).

**bench_ducky re-run against the new code default: still 0/10**, but a
real qualitative shift, not the same failure as before. Completions are
now built from genuine (if ultimately wrong) Python idioms -- repeated
`isinstance` type-checks, `raise ValueError`, real control-flow shapes --
failing on malformed syntax/logic that never actually solves the task,
not on incoherence or the old "1-2 tokens then abstain" pattern smaller
checkpoints showed. Same "coherent but not capable" finding this project
has now confirmed at four separate scale points (Phase L's original
~10M-param checkpoint, Phase T's chinchilla_min on the old recipe, and
now this properly-recipe'd, GPU-trained rerun) -- architecture and
recipe improvements measurably improve the base model (loss, coherence,
idiom-correctness) without touching this specific capability ceiling.
Confirms rather than overturns the project's most-repeated finding:
scaffolding and better training amplify what the model already knows,
they don't manufacture task-solving capability it doesn't have.

## Phase AB — Abstention removed, then graph-blending removed too; single-corpus SDK
User's explicit sequence of calls, each a real product decision, not a bug
fix: (1) abstention was "clearly stifling" output quality, remove it;
(2) generate long output first, revisit a hallucination gate later; (3)
Ducky should be single-domain (rj) and single-backbone (RWKV-hybrid), no
domain=/backbone= arguments; (4) the TokenGraph blend was "causing Ducky
to give bad results," remove that too.

- [x] Abstention removed from `inference.py`: `predict_next` can no
      longer return `ABSTAIN` -- always a real token, fast path (high
      neural confidence) or slow path (graph-blended, at the time).
      `calibrate_thresholds` simplified from 3 percentile thresholds to 1.
      Deleted `eval_grounding.py` and `eval_predictor_paths.py` (both
      measured/depended on the now-removed mechanism, not worth keeping
      half-broken).
- [x] `Ducky()` collapsed to single-domain/single-backbone: no `domain=`/
      `backbone=` params, hardcoded to rj + RWKV-hybrid
      (`rj_base_m_rwkv_lrplateau_nanogpt_seed57`). Code-only machinery
      (symbol_table, call_graph, AST-fact injection, `use_retrieval`)
      removed rather than left half-disabled -- `DEFAULT_RUNS`'s (domain,
      backbone) dict collapsed to one `DEFAULT_RUN` constant.
      `generate_with_grounding` gained a real `temperature` parameter
      (previously missing entirely) after finding `ask()`'s declared
      `temperature=0.8` default silently did nothing on the primary
      (n_candidates=1) path.
- [x] Added nucleus (top-p) sampling to `_choose`. **Measured, not
      guessed, the actual sweet spot**: temperature=0.8/top_p=0.9 (the
      first thing tried) produced fluent-*sounding* but frequently garbled
      non-words ("shadn", "wcup", "penk") -- this checkpoint's per-token
      confidence isn't peaked enough for 2nd/3rd-choice subword pieces to
      reliably compose into real words. A direct side-by-side swept down
      to temperature=0.3-0.5/top_p=0.5-0.7: legibility came back sharply.
      temperature=0.5/top_p=0.5 set as the new default.
- [x] Real gaps found testing MCTS/repair-loop at the new 300-token
      length (never validated past the old short abstain-truncated
      completions): MCTS's `n_simulations=6` default only reaches ~16
      tokens deep before running out of simulation budget (needs budget
      scaled to `max_new_tokens / chunk_size`, not fixed) -- **not yet
      fixed, flagged**. repair-loop's pass/fail check
      (`domain != "code" or syntax_valid`) is trivially true for
      non-code domains, so it "passes" on attempt 1 regardless of
      quality -- no real text-domain quality gate exists yet, also
      **not yet fixed, flagged**.
- [x] **Graph-blending removed** (`predict_next`'s `alpha*neural_logits +
      beta*graph_scores` slow-path mix): every token is now 100% the
      model's own computation, no `graph` parameter left anywhere in
      `inference.py`/`mcts_lite.py`/`repair_loop.py`. `fast_threshold`/
      `calibrate_thresholds`/`measure_confidence_distribution` deleted
      entirely (nothing left to gate a fast/slow split between). Fixed
      the resulting break in `bench_arithmetic.py` (real, previously-
      working calls that passed a now-nonexistent `graph` argument).
- [x] **Real, stated capability loss, not glossed over: `Ducky.learn()`
      is gone.** It only ever worked by adding edges to the TokenGraph --
      "no retraining, graph update instead" WAS the mechanism, not one
      option among several. Removing the graph removed the only thing
      `learn()` had to act on. Ducky currently has no way to incorporate
      new information short of retraining. The entire setup-cache
      mechanism (`SETUP_CACHE_DIR`, pickled graph edges + thresholds) was
      removed alongside it -- nothing expensive is left to cache once
      graph-building (300 forward passes) and threshold calibration
      (500+ more) are both gone; building the n-gram index for rj's
      ~50K-token corpus is fast enough on its own.
- [x] Verified end-to-end after the full removal: `Ducky()` loads,
      `ask()` generates 300 real tokens with comparable legibility to
      before graph removal, and all four generation modes (plain,
      resample, MCTS, repair) run without error.

## Phase AC — Word-level garbling diagnosed to its root cause and fixed: tokenizer, not architecture
User's report: samples through the SDK showed frequent garbled non-words
("ambs'd", "blesh", "spless") even at low temperature. Diagnosed, not
guessed: the shared vocab=1024 tokenizer fragments every character name
into 4-7 BPE pieces (`ROMEO -> ['R','O','ME','O']`, `BENVOLIO` -> 7
pieces). Getting every piece right, in order, across that many
autoregressive steps is genuinely hard for a ~1M-param model -- confirmed
this is the actual mechanism, not a decoding-randomness artifact: garbled
character names (`ROMEome`, `ROETER`) still appeared even at
temperature=0.15 (near-greedy), meaning the model's own single most
likely prediction is sometimes wrong at exactly these fragment
boundaries.

Two retrain attempts to reduce fragmentation via a bigger *shared*
vocabulary both made things **worse**, a real negative result: vocab=32768
(rank-32 factored embedding) degenerated into near-total gibberish
("ROMEOME I.sIO"); vocab=8192 (full embedding) was better but still
badly garbled ("bygy", "fis'd", "goler"). Diagnosed why: spreading rj's
tiny ~38-66K-token corpus across 8-32x more distinct token types leaves
too few repetitions per token for a model this small to learn reliable
transitions -- the fragmentation-reduction benefit was real but
outweighed by the training-signal-sparsity cost at this corpus size.

**Real fix: a tokenizer trained ONLY on romeo_and_juliet.txt** (not the
shared multi-domain vocab), sized to the corpus's own actual vocabulary
(3,574 unique words -> vocab=2000 BPE, `spm_rj_only_2000.model`).
Verified before training anything: every character name became a single
token (`ROMEO`, `JULIET`, `MERCUTIO`, `BENVOLIO` -- previously 4-7 pieces
each). Added `--tokenizer-model-path` to `train.py` (loads an exact
`.model` file via `Tokenizer(model_path=...)`, bypassing the shared
vocab_size/variant convention) and `ducky.py`'s `_load_single_model`
reads it back the same way. Retrained the rj/m/RWKV-hybrid/nanogpt-recipe
config under this tokenizer (1,066,112 params) -- **real, verified fix**:
a 6-prompt side-by-side test showed every single `ROMEO.`/`JULIET.`
character-name occurrence spelled correctly, zero fragmentation, versus
frequent garbling before. Promoted to `DEFAULT_RUN`. Still trained on
Romeo & Juliet alone -- the tokenizer's own training data doesn't change
what the model learns to predict, only how text gets encoded, so this
stays within the user's explicit R&J-only scope.

Remaining, different, not-yet-solved issue found by the same test: the
fixed-tokenizer checkpoint still degrades into repetitive phrasing
("I'll not, I's my lady? ROMEO. I'st thou not...") over a long
generation. That's the model's actual capacity/data-scale ceiling
surfacing, not a tokenization artifact -- the same "coherent but not
capable" finding this project has hit repeatedly, now isolated cleanly
from the (now-fixed) word-fragmentation problem instead of being
tangled up with it.

## Phase AG — Conversational data added: real structural learning, same capacity ceiling as content
User's goal: "introduce conversational data into Ducky so it can respond
naturally." The only conversational data already in the repo
(chat_corpus.txt) is anonymized IRC chatroom logs (~22.5% pure
`JOIN`/`PART`/`ACTION` protocol noise, the rest crude unstructured
multi-user chatter, no real turn structure) -- flagged as a poor fit
before touching it, per the user's own choice: write a small, clean,
curated set instead, matching this project's established "one deliberate
input" discipline (rj itself, the hand-picked stdlib corpus).

- [x] Wrote `data/text/conversation_corpus.txt`, `User:`/`Ducky:` turns
      reusing rj's own "NAME: dialogue" structure Ducky already handles
      correctly. First pass: 1,585 words. Extended to 6,492 words (a real
      4x expansion, genuinely varied topics -- feelings, books/rj
      tie-ins, animals, hobbies, travel, technology, cooking, gratitude,
      humor, quick trivia, advice) after the first pass showed the
      structure was learnable but too small to move content coherence.
- [x] Real gap found and fixed before training: the dedicated
      rj-only tokenizer (Phase AC) fragmented conversational vocabulary
      badly -- even "Ducky" (the model's own name) split into 3 pieces
      (`D`+`uck`+`y`). Built a new combined tokenizer
      (`spm_rj_conv_2300.model`, vocab sized to the real combined unique-
      word count, 4,046) trained on rj + conversation_corpus.txt
      together -- verified "Ducky", "conversation", "kindness" and every
      character name (ROMEO/JULIET/MERCUTIO/BENVOLIO) all single tokens.
- [x] Added weighted per-example pool sampling for rj vs. conversation
      (`load_weighted_rj_corpus` in `data.py`, `--rj-conversation-weight`
      in `train.py`) -- reused `get_weighted_code_batch` as-is (already
      generic despite its name, no code-specific logic in it) rather than
      duplicating the mechanism. Without this, conversation's ~6%-by-word
      share would round to near-zero training exposure, same reasoning
      as code's core/breadth weighting.
- [x] **Trained and tested at weight=0.3 (first, small corpus) and
      weight=0.3/weight=0.15 (after the 4x expansion). Real, honest
      result at every setting: the turn-taking STRUCTURE is genuinely
      learned** (correct `User:`/`Ducky:` labels appear reliably, real
      conversational vocabulary and phrasing patterns show up) **but
      response CONTENT never coherently addresses the specific question
      asked**, and cross-contamination into rj prompts is real and
      inconsistent -- `ROMEO:` sometimes stays in Shakespearean register,
      sometimes doesn't, unpredictably, even at the same weight. Lowering
      the weight (0.3 -> 0.15) didn't reliably fix the contamination
      (JULIET: recovered proper register in one test, ROMEO: still didn't
      in the same run).
- [x] **Diagnosis: this is the same capability ceiling this project has
      hit at every prior scale/recipe/architecture combination
      (`bench_ducky` 0/10 throughout), not a new or separately-fixable
      bug.** A ~1.1M-parameter model doesn't have the capacity to both
      cleanly separate two registers by prompt cue AND produce
      consistently coherent, on-topic content in either -- more weight-
      tuning on the same small model doesn't cross that ceiling, it just
      moves where the inconsistency shows up.
- [x] Real bug found and fixed as a byproduct: `--rj-conversation-weight`
      wasn't part of `train.py`'s run-name suffix, so two different
      weight values (0.3, then 0.15) silently overwrote the same run
      directory in sequence -- the same overwrite-collision class this
      project has hit repeatedly. Fixed additively (`_convw{weight}`
      suffix), verified going forward only, not retroactively (the 0.3
      checkpoint's specific weights are gone, but its measured behavior
      is recorded here).

## Phase AH — Code added as a third weighted pool: mechanism validated at toy scale
Direct follow-through on Phase AG, per the user's own stated plan
("I will expand to code once this has been nailed down"). User's
explicit request: add code the same deliberate way conversation was
added, toy-scale first, before any real scale-up.

- [x] Extended `load_weighted_rj_corpus` (`data.py`) to optionally load
      `corpus_core.txt` (this project's existing hand-picked stdlib
      extraction, not newly scraped) as a third pool, opt-in via
      `include_code`. Added `--rj-code-weight` to `train.py`, composed
      the same way `--code-synthetic-weight` sits on top of
      `--code-core-weight`: code's weight taken off the top, remainder
      split between rj/conversation per the existing
      `--rj-conversation-weight`. Added the missing run-name suffix
      (`_codew{weight}`) proactively this time, alongside fixing the
      still-missing `_convw{weight}` suffix from Phase AG -- no
      overwrite collision this round.
- [x] Built a third combined tokenizer (`spm_rj_conv_code_8192.model`,
      vocab=8192 -- sized up from 2300 to match the real combined
      unique-identifier count once code enters the mix, 49,924 vs. the
      rj+conversation-only 4,046). Verified before training: character
      names, "Ducky", and common Python keywords (`def`, `return`,
      `self`, `import`, `class`) all single tokens.
- [x] Trained at "l" size (6.87M params, vocab=8192, no embedding-rank
      factoring -- the difference vs. rank=64 wasn't large enough to
      bother at this scale), weights rj=0.425/conversation=0.075/code=0.5.
      Converged at step 700 (best_val=4.9325) -- notably later than the
      2-pool version's step 200, consistent with code adding real
      additional complexity for the model to work through before
      overfitting sets in.
- [x] **Real, encouraging result: register separation held for all
      three registers in the same test that surfaced 2-pool
      contamination before.** ROMEO:/JULIET: stayed cleanly Shakespearean
      (correct names, real vocabulary, zero conversational or code
      bleed-through in this sample). Code prompts (`def add(a, b):`,
      `import os`) produced genuine Python *shape* -- docstrings,
      `return`, `raise ValueError(...)`, `isinstance()` checks -- not
      valid/executable code, but recognizably code-structured rather than
      prose or Shakespearean verse. Same content-coherence ceiling as
      conversational responses, now showing up as invalid/incomplete
      syntax instead of off-topic answers -- not a new problem, the same
      one in a third shape.
- [x] **Conclusion: the three-way weighted-pool + shared-tokenizer
      mechanism is validated at toy scale.** This was the explicit
      precondition ("only after adding in coding... like we did with
      conversational data") before discussing a real scale-up -- that
      precondition is now met.

## Phase AI — The real scale-up: literary + conversation + code, Chinchilla-matched
Direct follow-through on Phase AH's validated mechanism, per the user's
explicit sequencing ("lets scale, but only after adding in coding").
Scope confirmed with the user first: code = corpus_core + corpus_breadth
(full ~43.26M-token real code corpus); text = expanded to rj + Gutenberg
("literary" pool, deliberately excluding chat_corpus.txt -- an explicit
quality decision from Phase AG, not a volume one); conversation = kept as
the existing small hand-curated set.

- [x] Found the same tokenizer-dilution problem in a new, bigger shape:
      even a tokenizer trained ON the combined real corpus still
      fragmented character names (ROMEO -> 3 pieces) because rj+
      conversation are only ~185KB against gutenberg+code's ~1.48GB (an
      ~8000:1 ratio) -- R&J's own vocabulary was too rare to earn
      dedicated BPE merges. Fixed with the exact discipline this project
      already used once before (Phase O's tokenizer-fairness stratified
      resampling): repeated the small rj+conversation pool 50x in the
      tokenizer's own training input. Verified: every character name,
      "Ducky", and common Python keywords are single tokens again.
- [x] Safely tokenized all four real pools before touching training:
      `safe_tokenize_literary.py` (new, rj+gutenberg only, reusing
      safe_tokenize_text.py's chunked mechanism) -- 300,326,357 tokens,
      peak RSS 5.6GB; code_breadth via the existing safe chunked
      tokenizer -- 41,576,969 tokens; code_core, safely chunked the same
      way on principle even though small -- 2,587,556 tokens;
      conversation -- 9,015 tokens (small enough to tokenize directly).
      **Total: 344,499,897 real tokens.**
- [x] Computed (not guessed) Chinchilla-optimal size: 344,499,897 / 20 =
      17,224,995. Registered `chinchilla_scaleup`
      (d_model=320, n_layer=12, n_head=8, embedding_rank=80) in
      `train.py`'s `SIZES` -- 17,487,680 params, within 2%.
- [x] Extended the data-loading/training-CLI surface additively:
      `load_scale_up_corpus` (`data.py`, reads all four pre-cached pools,
      1.2s load time, zero raw re-tokenization) and `--scale-up`/
      `--scaleup-code-weight` (0.4 default)/`--scaleup-conversation-weight`
      (0.2 default) in `train.py`, composed the same way
      `--code-synthetic-weight` sits on top of `--code-core-weight`.
      Fixed a real bug caught by re-deriving the lazy-loading logic
      before running it: `_tokenize_corpus` takes a plain string, not a
      callable -- passing `gutenberg_path.read_text()` directly (matching
      existing code's own eager convention) would have read the full
      1.3GB file into memory on every call even when the cache already
      existed. Added `_tokenize_corpus_lazy` (checks the cache first,
      only calls the text-producing function on a genuine miss) instead.
- [x] Real GPU contention found at smoke-test time (not assumed): a
      Space Engineers 2 game process (~7.5GB) plus the recurring
      `uchi.flux.react_warmup_train` job together left too little free
      VRAM for the full config -- caught via a real OOM on the very
      first smoke test, not guessed at. Reused the established
      `--batch-size 8 --grad-accum-steps 4` fix from Phase AA (same
      effective batch of 32, fits in what's actually free).
- [x] Measured real steady-state throughput (~0.30s/step) via two smoke
      tests before committing to the full run -- the first one
      accidentally dropped `--tokenizer-model-path`/`--scale-up` (caught
      by checking the run's own recorded name, not assumed correct) and
      had to be redone properly. Estimated ~7-8 hours for a ~90,000-step
      ceiling (roughly one real pass over the combined corpus at
      Chinchilla ratio) -- flagged as a real, multi-hour commitment and
      confirmed with the user explicitly before launching, given this
      is qualitatively different from every prior toy-scale run this
      session.
- [x] **Completed: early-stopped at step 21,500 of the 90,000-step
      ceiling, best_val=4.1542, 8,710.7s (~2.4 hours) total** --
      dramatically faster than the ~7-8 hour estimate, since it
      converged and stopped rather than running the full budget.
      Promoted to `DEFAULT_RUN`.
- [x] **Verified across all three registers -- a genuine, real
      qualitative jump, not just a bigger number.** Conversational
      responses are coherently ON-TOPIC for the first time in this
      project's history: "What is your name?" -> "My name is Ducky.
      What's yours?"; "What do you think about friendship?" -> "I think
      friendship is one of the best things two people can share." Code
      prompts produce real structure -- proper docstrings with doctest-
      style examples (`>>> ExtendedContext.add(...)`), plausible control
      flow (`if value in self._values:`). This is the first checkpoint in
      the entire Ducky history to show genuine content coherence rather
      than plausible-but-empty phrasing -- real scale (17.5M params, 344M
      real tokens) crossed a threshold no toy-scale combination did.
- [x] **Real, disclosed tradeoff, not smoothed over**: ROMEO:/JULIET:
      prompts now produce fluent prose, but it's drifted from strict
      Shakespearean verse-drama toward general 19th-century novel style
      -- the "literary" pool blends rj with ~2,000 Gutenberg books, and
      at real scale that broader literary register measurably dilutes
      rj's own specifically dramatic voice. An inherent cost of
      expanding text to Gutenberg, not a bug to fix.
- [x] **Confirms this project's own repeated finding from the other
      direction**: every toy-scale combination (up to ~17M params on
      curated-but-small corpora) hit the same "coherent structure, not
      coherent content" ceiling; real scale (comparable params, but 344M
      real tokens instead of tens of thousands) crossed it. Capability
      really was a scale/data problem, not an architecture problem --
      exactly what this project's own scaling-law work (Phase Q) and
      repeated small-scale-first discipline predicted before ever
      committing to this run.

## Phase AF — Recency-weighted repetition penalty: tested properly, real negative result
Picking back up the recency-weighted repetition penalty (paused mid-test
in Phase AD to fix the context-window limit first). Single-sample
eyeballing across a few decay values hadn't given a clear signal either
way -- exactly the kind of premature read this project's own discipline
warns against -- so this was finished with a real, quantitative,
multi-seed test instead of continuing to read text by eye.

- [x] Measured distinct-2/distinct-3 diversity (Li et al. 2016) across 5
      seeds, 250 tokens each, repetition_penalty=1.3, sweeping
      recency_decay in {1.0 (flat), 0.99, 0.98, 0.95, 0.9}. **Clean,
      monotonic, unambiguous result: flat (decay=1.0) wins outright**
      (distinct-2=0.8008, distinct-3=0.9411), and diversity gets steadily
      *worse* as decay drops (0.9: distinct-2=0.5542, distinct-3=0.8032).
      The hypothesis behind adding decay was real (a flat penalty treats
      a necessary common word used 200 tokens ago the same as one used 2
      tokens ago) -- but the fix doesn't work as designed: letting old
      tokens' penalties fade doesn't selectively free up necessary common
      words, it just as freely lets old repeated PHRASES resurface too,
      and at this checkpoint's scale that effect dominates.
- [x] **Real bug caught and fixed as a direct result of this test**:
      `_apply_repetition_penalty`'s `recency_decay` and `predict_next`'s
      `repetition_penalty_decay` both defaulted to 0.95 (the
      experimental, now-measured-worse value) -- meaning `Ducky.ask()`'s
      actual default behavior was silently using the rejected setting the
      whole time, not the validated flat one, since nothing overrode it
      downstream. Fixed both defaults to 1.0 (flat, matching Phase AD's
      original validated behavior) and re-verified: `predict_next` called
      with no explicit decay argument now reproduces the best-measured
      distinct-2 score (0.8008) exactly.
- [x] Kept `recency_decay`/`repetition_penalty_decay` as available,
      off-by-default parameters (not deleted) -- a real, informative
      negative result worth being able to reproduce or re-examine later,
      same discipline as every other tested-and-rejected mechanism in
      this file (selective decay, tie_layers, etc.), not silently erased.

## Phase AE — The real context-window limit: RWKV state never actually carried at inference time
User's question ("what is the limit for the amount Ducky can generate")
surfaced a genuine architectural gap, distinct from max_new_tokens (which
was never limited -- a plain loop counter). `predict_next` always did
`model(idx[:, -block_size:])`, a fresh forward pass over just the
trailing 128 tokens every step, and never passed or captured
`rwkv_states` -- despite `model.py`'s own `hidden_states()` already
documenting exactly this as "the unlimited-context mechanism." The
project had separately, structurally verified RWKV's O(1)-state
long-context capability in earlier phases, but Ducky's actual generation
loop never used it: in practice, past ~128 tokens the model could not
see any earlier part of its own generation at all, not by choice but
by construction (`pos_emb` is `nn.Embedding(block_size, d_model)`, a
table with exactly block_size rows -- an absolute position past
block_size-1 is out of bounds, which is *why* the crop existed).

- [x] Added `ChunkedState` (`inference.py`): carries per-block RWKV state
      across `block_size`-token chunk boundaries, threaded through
      `predict_next` (`chunk_state` param, default None = old behavior
      unchanged for any caller not updated) and on by default in
      `generate_with_grounding`/`generate_with_resampling`
      (`use_chunked_state=True`).
- [x] **A real bug caught and fixed before ever running it**, by
      re-deriving the invariant carefully: the first version rolled the
      chunk buffer over to fully empty at each block_size boundary, which
      would crash `next_logits()` (a forward pass needs at least one real
      token; carried_states alone can't produce output from zero input).
      Fixed by always leaving the newest token active after a rollover
      (fold everything *except* the last token into carried_states, keep
      the last token as the new chunk's first element) -- verified this
      maintains a correct, gap-free, non-duplicated split between
      carried_states and the active chunk at every step.
- [x] **Verified correct, not just crash-free**: the base case (short
      context, no rollover) produces byte-identical logits to the old
      crop-and-forward path (max diff 0.0). A direct 300-token, same-seed
      A/B test showed the chunked and non-chunked paths produce
      *identical* output through token ~128, then genuinely diverge right
      at the chunk boundary -- exactly the expected signature of the old
      path losing access to early tokens while the new path retains them.
- [x] **Honest result on quality, consistent with (not contradicting)
      Phase M's own finding**: the fix is real and verified -- the model
      now genuinely processes information beyond 128 tokens back, not
      just in principle but confirmed by the outputs actually differing.
      It does not produce an obviously more coherent 300-token sample at
      this checkpoint's scale, matching Phase M's five-for-five BPTT
      finding: the architecture can carry long-range state, but this
      checkpoint was never trained with any pressure to actually
      *exploit* carried state (ordinary short-context training, not
      cross-chunk BPTT) -- carrying it correctly at inference time was
      never going to manufacture a capability training never taught.
      The fix closes the described limit honestly either way: Ducky no
      longer has a hidden 128-token amnesia wall, whether or not this
      checkpoint currently has learned to make full use of that.

## Phase AD — Repetitive phrasing fixed: a real repetition penalty, distinct from no-repeat-ngram blocking
The fixed-tokenizer checkpoint (Phase AC) still degraded into cyclic
phrasing over long generations ("I'll not, I's my lady? I'st thou
not..."). Diagnosed as a genuinely different failure mode from what
`no_repeat_ngram_size` already handles: that mechanism only blocks an
*exact* repeated n-gram, and does nothing when the same handful of
tokens ("I'll", "not", a name) recur across many short phrases that are
each individually novel.

- [x] Added `_apply_repetition_penalty` (`inference.py`): CTRL-style
      per-token penalty (Keskar et al. 2019, arXiv:1909.05858) -- every
      token already seen in context gets its logit divided (if positive)
      or multiplied (if negative) by the penalty, discouraging reuse
      without forbidding it outright. `repetition_penalty=1.0` (default)
      is a no-op, threaded through `predict_next` and every generation
      wrapper (`generate_with_grounding`, `generate_with_resampling`,
      `mcts_generate`, `generate_with_repair`) with zero behavior change
      unless explicitly set.
- [x] **Measured, not guessed, before picking a default.** Tested
      directly at the `predict_next` level first: 1.0 reproduces the
      known repetitive loop; 1.2-2.0 all broke it, with real character
      variety (JULIET, CAPULET, NURSE, MERCUTIO) and stage directions
      (`[_Exeunt._]`, `SCENE II`) appearing instead of cycling. **Real,
      disclosed tradeoff**: the same mechanism that stops phrase-level
      repetition can also discourage an already-used subword piece
      mid-word, occasionally corrupting an unrelated rarer word
      ("tidy-sed", "pvogy") -- character names stay correct throughout
      (still single tokens after Phase AC's fix), only less-common
      multi-piece words are at risk. 1.3 kept slightly less of this
      collateral garbling than 1.5 while fixing the loop just as
      cleanly; set as the new `ask()` default.
- [x] Verified end-to-end across 4 prompts and all four generation modes
      (plain, resample, MCTS, repair) with the new default -- no crashes,
      the repetitive-phrasing pattern is gone, replaced by genuine
      multi-character dialogue with scene structure.

**Two real mistakes made and disclosed during this phase, not hidden**:
both the first bigger-shared-vocab retrain and the first rj-only-tokenizer
retrain saved to the exact same run-directory name as the existing
default checkpoint (no distinguishing suffix for `--vocab-size` alone or
for the new `--tokenizer-model-path` flag), silently overwriting it each
time -- the same overwrite-collision bug class this project has hit
several times before. Both caught immediately (config.json's own
recorded vocab_size didn't match what was just run), the original
vocab=1024 checkpoint was re-trained and confirmed reproducing its
recorded number (best_val 3.5943588 vs. 3.5944) each time, and
`train.py`'s run-naming logic was fixed to add a `_tok{stem}` suffix for
`--tokenizer-model-path` so this can't recur for that flag.
The RTX 5070 that had been pinned at 11.2/12.2GB by a live uchi training job
(the constraint stated at the top of `tasks/todo.md` since 2026-07-15) is
now free (`nvidia-smi`: 826MiB/12,227MiB, 5% util). This directly unblocks
Phase Q's own stated gap: "a firm verdict at real production scale would
need m/l points too" -- only xs/s (2 close-together points) had ever been
tested. Per `no-scaleup-without-proof` and `core_principle.md`'s own
"fit a curve across sizes" upgrade, the right next step is exactly this --
more size points on the existing cheap harness -- not a jump to a real
production run.

**GPU wiring did not exist.** Despite `tasks/todo.md`'s Phase A intent
("device auto-detect"), nothing in `train.py`/`model.py`/`rwkv_model.py`
ever called `.to(cuda)` -- every tensor implicitly lived on CPU regardless
of what hardware was free. Added `DEVICE` autodetect + `.to(DEVICE)` calls
to `run_scaling_sweep.py` only (scoped to this sweep, not a `train.py`
CLI change -- out of scope here).

**First smoke test looked catastrophic (14.3s/step) -- diagnosed before
trusting it.** A controlled CPU-vs-GPU steady-state comparison (warm-up
steps excluded) showed the opposite: GPU is 30-40x faster per step once
past a one-time `torch.compile` cost that varies wildly by shape (2.8s to
255s observed, since the WKV scan's `torch.compile` fusion -- Phase I --
recompiles per distinct model config). Worst-case compile tax is still
dwarfed by steady-state savings over a real step budget. This is why the
literal first-step number was misleading; always separate compile-time
from steady-state before judging GPU vs CPU on a compiled recurrent scan.

**Extended `run_scaling_sweep.py` to `SIZES_SWEPT = ["xs","s","m","l"]`**
(24 configs: 4 sizes x 3 domains x 2 archs) and ran the full sweep on GPU.
**Result looked decisive but was contaminated**: `rj/l` had selective
decay edge ahead (6.199 vs 6.2193, matching the small-margin outcome that
also survived the fix below), but `code_core`/`terminal` showed wild,
non-monotonic flip-flops at m/l -- e.g. `code_core/m`: rwkv 7.1279
(early-stopped at step 150) vs. selective 6.2156 (ran to step 950); then
`code_core/l` flipped the other way, rwkv 5.1537 (step 1650) vs. selective
7.1175 (stuck at step 150). Every large gap correlated with one arch
getting stuck at a suspiciously early step (150-250) while the other
trained on for hundreds more steps.

**Root cause, diagnosed not assumed**: `PATIENCE=2` at 50-step checks (100
steps of grace) combined with only 5 val batches per check -- tuned
against xs/s, where it happened not to bite -- was too aggressive and too
noisy at m/l. A config could hit a noisy dip, exhaust its 2-check grace
window, and halt permanently before it would have broken through, while
an equally-capable config with a slightly different loss trajectory
escaped the same dip and kept converging for another 800+ steps. This is a
stopping-rule artifact, not an architecture difference -- confirmed by
the same "stuck-at-150-vs-1000+" pattern recurring on all 3 domains
(`terminal/m`: rwkv stuck at 450 = 5.197 vs. selective at 1450 = 3.8075).

**Fix (`rerun_ml.py`, new file)**: loosened `PATIENCE` 2->6 and
`VAL_BATCHES` 5->10 in `run_scaling_sweep.py`, re-trained only the
contaminated m/l points (12 runs), reused the already-clean xs/s points
straight from their `config.json` files (never showed the stuck-early
pattern, no need to re-spend GPU time on them). Confirmed the fix worked:
matched-arch pairs now stop at the *same* step far more often (e.g.
`rj/m`: both stop at 750; `code_core/m`: both hit the full 2000-step
ceiling; `terminal/m`: both stop at 1950) instead of one at ~150 and the
other at ~1000+.

**Real, trustworthy 4-point verdict** (`fit_scaling_law.py`, extrapolated
to 10x past the largest tested `block_stack_params`):
- `rj`: near-exact tie -- rwkv alpha=0.0179, selective alpha=0.0190,
  extrapolated loss 5.932 vs. 5.928 (selective wins by 0.004 nats,
  noise-level). Both fitted exponents are tiny, consistent with rj being
  this project's most-established data-ceiling-limited domain (see Phase
  T/U) -- neither architecture has much curve left to show on this corpus
  regardless of capacity.
- `code_core`: rwkv wins, alpha=0.0709 vs. 0.0711 (near-identical
  exponents), extrapolated loss 4.322 vs. 4.345 -- a real, small,
  consistent margin (0.023 nats).
- `terminal`: rwkv wins clearly, alpha=0.1086 vs. 0.1004 (meaningfully
  steeper, ~8% relative), extrapolated loss 2.614 vs. 2.771 -- the
  largest, most decisive margin (0.157 nats) and the only domain where the
  fitted exponents themselves (not just one extrapolated number) visibly
  favor one architecture.
- **`promote selective decay = False` stands**, but now on a materially
  stronger footing than the original 2-point (xs/s) result it supersedes:
  4 real size points, no stopping-rule contamination, and a coherent
  story -- RWKV wins or ties everywhere, with the size of its win tracking
  how much real scaling headroom each domain has (near-zero on
  data-capped rj, real and growing on code_core/terminal).

**Honest caveat carried forward, not smoothed over**: several `l`-size
runs (`code_core/l` both archs, `terminal` less so) hit the 2000-step
ceiling without patience ever triggering -- they may not have fully
converged, so the exact extrapolated-loss numbers could still move with a
longer budget. The *ranking* is unlikely to flip given how it lines up
with each domain's already-established scaling headroom, but this is
flagged as a real, not-yet-closed loose end, same discipline as every
other "still improving when cut off" note in this file.

**Answers the standing "architecture across small and large text" question
concretely for the first time**: RWKV-hybrid is the architecture to carry
into any future production-scale commitment on this codebase -- it never
loses meaningfully across 4 real size points on 3 domains, and its margin
grows (not shrinks or reverses) exactly where a domain has real headroom
left to give. Per `no-scaleup-without-proof`, this is the kind of
cheap-but-real evidence that would justify a genuine scale-up decision,
now that it exists -- the decision itself is a separate, deliberate step,
not a default next action.

## Docstring drift found and fixed: DEFAULT_RUN's real training data vs. ducky.py's own claim

`ducky.py`'s top-of-file docstring claimed (2026-07-22) "Deliberately
single-domain... rj is the one corpus in scope" -- true of the checkpoint
`DEFAULT_RUN` pointed to *at the time that sentence was written*, but
stale after the real scale-up landed (same commit that introduced
`DEFAULT_RUN`'s current value): the actual default checkpoint
(`rj_base_chinchilla_scaleup_rwkv_rank80_tokspm_ducky_scale_32768_
scaleup_lrplateau_nanogpt_seed57`) has `scale_up: true` in its
`config.json`, trained on four weighted pools -- literary (rj+gutenberg,
~300M tokens), conversation (~9K tokens, hand-curated), code_core+
code_breadth (~44M tokens combined) -- 344.5M tokens total, not rj alone.
The docstring and the code it sits above had quietly diverged.

Verified with live generations through the real `Ducky()` SDK (not just
read from config), per the user's explicit request to "see some
generations":
- `"ROMEO:"` -> generic Gutenberg-era prose, not Shakespearean -- the
  ~300M-token literary pool dwarfs rj's ~26K words even after the
  tokenizer's 50x upsampling of rj+conversation.
- `"def parse_config("` -> structurally real Python (docstring, `if not
  path:`, a comment), semantically loopy -- unambiguous evidence of code
  training.
- `"User: how are you feeling today?\nDucky:"` -> coherent multi-turn
  dialogue that naturally invokes rj content ("Wherefore art thou
  Romeo", "parting is such sweet sorrow") -- real cross-domain blending,
  not just three separate skills bolted together.
This matches what the checkpoint's own training-time `samples.json`
already showed (step 15000/20000/25000 samples drifting from "ROMEO:"
into Python snippets and Gutenberg-metadata-style text), just not
previously cross-checked against the docstring's claim.

Fixed: the docstring now states the real four-pool mix and points at
`DEFAULT_RUN`'s own comment as the source of truth, and the `self.domain
= "rj"` line in `__init__` is now commented as a grounding-behavior
switch (skips code-only checks like syntax validity), not a claim about
training data -- the two were easy to conflate reading the code before
this pass. No behavior changed, only the documentation now matches what
the checkpoint actually is.

## Agent harness: public SDK + terminal UI (ducky_agent package)

Per the user's explicit request to build out the North Star's "growing
into something that needs to act as an agent" objective: a real,
installable Python SDK (`ducky-agent` distribution, `from ducky_agent
import DuckyAgent`) and a terminal UI (`ducky-agent` console script,
Textual-based), adapted from the harness principle of a reference
open-source repo (decodingai-magazine/building-a-coding-agent-from-
scratch-course) to what Ducky actually is: a raw next-token predictor
with zero instruction-tuning and zero native function-calling, unlike
the reference harness's reliance on a provider LLM's own structured
tool-calling API.

**Architecture**: a text-parsed Thought/Action grammar
(`ducky_agent/action_parser.py`) extracts tool calls from Ducky's free
generation via a fixed few-shot example in the prompt preamble, validated
with a restricted `ast.parse(mode="eval")` (single `Call`, `Name` func,
all-keyword `Constant` args -- the same restricted-AST idiom this repo
already uses in `grounding.py`, not a hand-rolled regex or `eval()`). A
permission gate (`ducky_agent/permissions/`) evaluates deny-rule >
allow-rule > mode-default > ask, shipping with `PermissionMode.DEFAULT`
asking a human before every `write_file`/`run_shell` call (only
`read_file`/`list_dir` auto-allow) -- given Ducky is measured
unreliable, gated-by-default is the safe posture, not an afterthought. A
4-tool minimal set (`read_file`, `list_dir`, `run_shell`, `write_file`)
covers what a toy-scale coding agent needs without diff/patch precision
(deliberately cut -- pays off only once the model reliably produces
correct patches, which it doesn't). `loop.py`'s `run_turn` is a generator
(`BuildPrompt -> Generate -> Parse -> Gate -> Execute/Ask -> Observation`)
that pauses at `PermissionAsked` and resumes via `generator.send()`, the
same yield-before-a-pause-point shape the reference harness's own turn
handler uses. `context/window.py` reuses `session_history.SessionHistory`
directly for transcript compaction (extractive, verbatim, bounded --
never an abstractive summary sitting in the model's own context) rather
than inventing a second compaction mechanism.

**Verified correct before ever touching real Ducky** (matching this
repo's own `bench_ducky.py`/`run_sandboxed` precedent: scripted-known-
cases before trusting a number from the real model): 56 pytest unit
tests (parser, tools, gate, loop, all fast and torch-free except the
TUI's own suite), `verify_agent_harness.py`'s 6 flat scripted
integration checks (canned-correct action executes and feeds its
observation forward; malformed action retries then succeeds; exhausted
retries fall back to a raw-text answer instead of crashing/hanging; a
model that never stops emitting Actions is halted cleanly by
`max_turns`; permission deny produces **zero real filesystem mutation**,
verified on disk; permission allow **actually writes the real file**,
verified on disk), and 4 headless Textual `Pilot`-driven TUI tests
(including driving the actual permission modal through real button
clicks and confirming deny still produces zero mutation through the full
UI path). All passing on the first real end-to-end run against
`ScriptedModel` before `DuckyModel` (the real-Ducky wrapper) was ever
built.

**Real, honest measurement** (`bench_ducky_agent.py`, 5 gradeable tasks
through the real `DuckyAgent` SDK + `DuckyModel` wrapping the same
default checkpoint as `Ducky()`, `max_turns=3`/`max_parse_retries=1`,
`temperature=0.5`/`top_p=0.5`/`repetition_penalty=1.3` -- Ducky's own
measured defaults): **0/5 tasks passed, and 0/5 produced a single
parseable Action across every attempt** (list a directory, read a file
and report its first line, write a file, trivial arithmetic, chain two
tool calls). Not just wrong syntax -- genuinely didn't engage with the
Thought/Action instruction format at all. Every completion drifted
straight into code-completion-flavored continuation regardless of the
task (`'""" if not hasattr(os, "join"): return None # --- torch/
_inductor/cmp.py ---...'`, `'py """ # This module is used to be an
object that will be used to use the # current directory...'`) -- the
model's own dominant training-pool pattern (code) overriding the
few-shot instruction in the prompt, not a parser bug: 0 `ParseErrorEvent`
alongside 0 `ActionParsed` means the text literally never contained an
`Action:` line to even attempt parsing.
**Sharper and more specific than `bench_ducky.py`'s own 0/10**: that
benchmark already established Ducky can't reliably produce a correct
function body given a docstring. This shows something more basic
upstream of that -- zero instruction-tuning means Ducky doesn't yet
reliably recognize a structured task format as something to follow at
all, even loosely, even incorrectly. Consistent with, not contradicted
by, every other 0/10-family result in this file (Phase L's
`mcts_lite.py`/`repair_loop.py`/`session_history.py`, Phase S/T/U's
flywheel and Chinchilla-min checkpoints) -- reasoning/agent scaffolding
amplifies existing capability, it cannot manufacture capability (here,
instruction-following) the base model was never trained toward.
**Does not move Track 1's `bench_ducky.py`-0/10 gate (`tasks/todo.md`
Phase V) in either direction** -- that gate is about code-completion
capability and stays exactly where it was; this measures the new
agent-harness scaffolding honestly, on its own terms, the same
Phase-L-consistent discipline as everything else built without base
capability yet catching up.
