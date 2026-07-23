# This Repo's North Star

**Core idea (revised, supersedes the original MoE-vs-dense framing below):
Ducky becomes the single, unified engine behind both Uchi (FLUX's role —
text/code reasoning) and Noosphere v2 (the ZOH-SSM stream encoder's role —
continuous biosignal processing for prosthetic control). One model, one set
of weights, two production systems currently running separate purpose-built
engines.** This is now the explicit, standing objective, not a hypothesis to
be talked out of — it is *how* it gets built, validated toy-scale-first at
every step exactly like every other architectural claim in this repo's
history, that stays non-negotiable.

**Why this feels correct:** Uchi's FLUX and Noosphere's stream encoder are
both, underneath very different production wrapping, sequence models over
tokens/vectors predicting what comes next (text token; continuous 6-DOF
intent). RWKV's linear-time-mixing recurrence doesn't care what a "token" is
mathematically — a text ID and a projected EEG sample are both just vectors
entering the same recurrence. If one shared core can serve both, the payoff
is real: a single training loop, a single set of weights, and a single place
that gets better instead of two divergent codebases each needing its own
tuning forever.

**The immediate prerequisite, not yet met:** neither replacement can happen
by fiat. Ducky is still 0/10 on `bench_ducky.py` (every measurement this
entire project, including post-scale-up) — it cannot yet replace FLUX at
even the text-only task. And no evidence yet exists that a ~17-20M-param
shared core retains competence on both text and continuous biosignal
prediction simultaneously — real successful shared-weight generalist models
(Gato, Reed et al. 2022, arXiv:2205.06175; PaLM-E, Driess et al. 2023,
arXiv:2303.03378) operate at 100M-1B+ params, not this repo's toy scale.
Both of these get closed by evidence, not by skipping the check — see
`tasks/todo.md` Phase V for the two-track plan (Uchi track gated on
`bench_ducky.py`; Noosphere track gated on a cheap toy shared-vs-separate
loss comparison, entirely off Noosphere's safety-critical production path
until that comparison produces a real number).

**Original core idea (Phase A-U's stated goal, kept for record — largely
answered, superseded by the above as the repo's live objective):** Prove, at
toy scale and near-zero GPU cost, whether a sparse (MoE) unified-tokenizer
next-token predictor buys more effective parameters per training FLOP than
uchi's current dense SSM+BitNet architecture — before spending real GPU
budget finding out the hard way.

**Why that felt correct:** The obvious move is to just scale FLUX up. But
dense compute scales linearly with parameter count — on a single consumer
GPU, that linearly caps how many parameters you can ever afford to train.
Mixture-of-Experts decouples total parameters from active-compute-per-token:
the same FLOP budget buys a larger model, *if* there are enough tokens per
expert for routing to specialize. That "if" is exactly what a toy-scale
experiment (one book, one small code corpus) can cheaply falsify before it
costs a real GPU-week to find out. **Verdict, from `tasks/todo.md`'s
Backlog section: MoE lost to dense at every tested point — this question is
answered, not open.**

**State-of-the-art grounding:**
- Scaling laws: Kaplan et al. 2020 (arXiv:2001.08361), Hoffmann et al. 2022 /
  Chinchilla (arXiv:2203.15556) — already used by uchi's own parameter budget.
- MoE done right at smaller scale: Shazeer et al. 2017 (arXiv:1701.06538),
  Fedus et al. 2022 Switch Transformer (arXiv:2101.03961), DeepSeekMoE 2024
  (arXiv:2401.06066, fine-grained experts + shared expert).
- Unified multimodal tokenization: Chameleon (Meta 2024, arXiv:2405.09818,
  early-fusion single token space), EnCodec for audio codes (Défossez et al.
  2022, arXiv:2210.13438), VQGAN for pixel tokens (Esser et al. 2021,
  arXiv:2012.09841). The backbone doesn't need to know the modality once
  tokenized — one architecture, one objective, three data sources.
- Hallucination is not a solvable-to-zero target: Kalai & Vempala 2024
  (arXiv:2311.14648) prove a calibrated model must hallucinate on facts seen
  once. The measurable goal is verified-accuracy + abstention-rate, matching
  uchi's own retrieval-grounded, "abstain when it can't verify" design —
  this repo tests whether training-time factuality objectives (e.g.
  DPO-style factuality tuning, Tian et al. 2023, arXiv:2311.08401) push that
  number further, not whether hallucination hits zero.
- Base architecture stays uchi's: Mamba/SSM (Gu & Dao 2023, arXiv:2312.00752),
  BitNet 1.58-bit (Ma et al. 2024, arXiv:2402.17764).

**Alternatives rejected:**
1. *Just scale dense FLUX up.* Rejected — dense FLOPs cap achievable params
   on a single GPU; this is the exact ceiling MoE exists to break.
2. *Bolt on separate per-modality models (uchi's current frozen
   VisionProjector pattern).* Rejected as the long-term target — it never
   unifies the learning objective. A shared discrete token space lets one
   backbone and one next-token loss cover text/audio/pixel.
3. *Solve hallucination via training alone.* Rejected as the stated goal —
   provably impossible to reach zero. uchi's retrieval+verification layer is
   the correct production answer and is out of scope here; this repo only
   tests whether factuality-aware training complements it.

**Non-negotiable scope discipline (revised):** Ducky-as-unified-engine
(replacing FLUX in Uchi, replacing the ZOH-SSM stream encoder in Noosphere)
is now the live objective, not backlog — but it is still gated exactly like
every other claim in this repo: no production swap and no joint multimodal
training run happens ahead of a cheap toy-scale result proving it should.
Concretely (`tasks/todo.md` Phase V): the Uchi swap is gated on
`bench_ducky.py` clearing 0/10; the Noosphere swap is gated on a toy
shared-vs-separate-baseline loss comparison, run entirely off Noosphere's
safety-critical production path. MoE and factuality-training-alone remain
answered-not-open (see verdicts above/in `tasks/todo.md`'s Backlog). Any
step taken toward the unified-engine goal that skips its gate, or that
touches Noosphere's actual safety-gated code before its gate clears, is
scope creep against this document, not progress. Drift from this triggers a
re-plan, not a bigger experiment.

**Standing methodology: small-scale-first, always.** Every architectural
idea proves itself at toy scale — `xs`/`s`/`m` preset, `rj` domain (or a
small slice of a bigger one) — *before* any `xl`/`xxl`-scale commitment is
made. Not a new rule: the pattern every real decision in this repo has
already followed. MoE/swarm were rejected using a 700-step toy run before
any GPU-week was spent chasing them. `tie_layers` and the confidence-gated
early-exit idea were both validated on `rj`/"m" (minutes, megabytes)
before touching `code`/xl (hours, gigabytes). The tokenizer-fairness
comparison trained three small (vocab=8192, a few MB) candidates before
committing to the one full-scale (vocab=32768, ~1.5GB) production build.
What's being written down here is the rule those decisions already
obeyed, so it stops being an instinct applied inconsistently and starts
being a checklist applied every time:

1. State the hypothesis and the cheapest possible test that could kill it
   — an inference-only probe against an existing checkpoint beats a new
   training run; a toy `rj`/"m" training run beats a `code`/xl one.
2. Run that test first. Report the real number, win or lose, before
   writing a single line of code that only matters at production scale.
3. Only after a toy-scale result is genuinely promising does spending
   xl/xxl-scale compute (hours, not minutes; gigabytes, not megabytes)
   become justified — and that jump itself gets called out explicitly as
   a separate, deliberate decision, not a natural next step taken by
   default.
4. A toy-scale negative result is a real result, not a reason to
   "try again bigger to be sure" — `tasks/ducky.md`'s own five-for-five
   BPTT-retention rounds and the MoE/swarm rejections both stand on toy-
   scale evidence, and scaling either of them up was never the fix for a
   mechanism-level negative finding.

This is the same discipline as the Chinchilla-ratio checks already used
for every dense/hybrid comparison in this file — just stated as a rule for
*how* to run any future experiment, not only *what* to measure once it's
running.

**Upgrade: fit a curve across sizes, not just one toy-scale point, when
the stakes justify it.** A single small-scale result can mislead — see
`tasks/ducky.md`'s selective-decay entry, where a single (vocab=1024,
"m"-size) point looked like a clean win and a 2-size, 3-domain sweep
reversed it. `src/fit_scaling_law.py` (Kaplan et al. 2020, arXiv:2001.08361;
Hoffmann et al./Chinchilla 2022, arXiv:2203.15556) fits `L(N) = a * N^
(-alpha)` from a log-log linear regression over several small sizes and
extrapolates — this is the literal, general-purpose answer to "simulate
large scale cheaply," and it's what `tasks/todo.md`'s original Phase A-D
plan set out to build before Ducky's own architecture questions took
over. Use it whenever a promotion decision (parked here vs. moved into the
production backbone) is being made, not just a single toy-scale
comparison — and, per the tie_layers and selective-decay lessons both,
across more than one domain, since averaging or single-domain checks are
exactly what hides the disagreement that matters.
