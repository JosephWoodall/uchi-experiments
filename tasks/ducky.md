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
- Repeat-seed check done (3 seeds, both corpora): **hybrid beats dense
  6/6.** On code, a strong, reliable win (mean margin 0.152 nats, std
  0.035 — small relative to the mean). On rj, the win is real but the
  *size* varies a lot seed to seed (mean margin 0.035, std 0.023, nearly
  as large as the mean) — direction is trustworthy, a precise margin
  number on rj specifically is not. Also confirmed the code corpus's
  growth (51K->149K tokens) mattered independent of architecture — dense's
  own baseline moved more from that (4.815->~4.06) than the hybrid-vs-dense
  gap did.

**Non-negotiable scope discipline:** Ducky's job is next-token prediction
quality first. Every grounding/abstention addition earns its place by
being cheap and checkable against something real (parse validity, a real
symbol, a real n-gram, this checkpoint's own recalibrated confidence) —
never by adding a second model, a vote, or an unverified heuristic dressed
up as intelligence. If a future addition can't point at what it's checked
against, it doesn't belong in Ducky.
