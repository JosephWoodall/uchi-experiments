# uchi-experiments — Compressed Plan

Constraint: RTX 5070 is at 11.2/12.2 GB, 66% util from the live uchi training
(confirmed via `nvidia-smi` on 2026-07-15). Everything below defaults to CPU;
GPU use here is opt-in and capped small, never assumed.

Goal of this plan: a params-vs-loss-vs-tokens scaling law, cheap enough to
run entirely on CPU, that tells us whether MoE / unified multimodal tokens /
factuality training are worth spending real uchi GPU budget on. Not: a
production model.

## Phase A — Harness (blocks everything else)
- [ ] Byte-level or small-BPE tokenizer, single shared vocab (reused later
      for the multimodal token-space test — build it generic from day one,
      but do not build audio/pixel tokenizers yet)
- [ ] Minimal dense Transformer or SSM train loop, CPU-first, device
      auto-detect with a hard VRAM ceiling (e.g. skip GPU entirely unless
      explicitly passed `--device cuda` and `--max-vram-mb` under ~1GB)
- [ ] Loss/perplexity logging per step, checkpointing, deterministic seed,
      hidden-state/activation dumping (needed later for Phase D probes —
      cheap to add now, expensive to retrofit)
- [ ] `--arm` flag with three values, all on the same base model/loop:
      - `base` — plain next-token CE (control)
      - `mtp` — + n extra linear heads predicting t+2..t+n (Gloeckle et al.
        2024, arXiv:2404.19737), heads dropped at inference
      - `jepa-aux` — + projection head and cosine/VICReg loss between two
        *paired* views of the same example (see Phase C — code only)
- [ ] Sanity run: train to near-zero loss on a 3-sentence toy string,
      confirm memorization — proves the loop is correct before spending
      any real compute

## Phase B — Text scaling sweep (Romeo & Juliet)
- [ ] Corpus: public-domain Romeo & Juliet text (~25K tokens)
- [ ] Arms in scope: `base`, `mtp` only — no natural paired second view
      exists in R&J alone, so `jepa-aux` is skipped here (see note above)
- [ ] Train 4 model sizes (e.g. ~50K / 200K / 1M / 5M params) × 2 arms,
      same tokens, same schedule, log final loss + wall-clock + CPU-seconds
- [ ] Fit Kaplan/Chinchilla-style power law L(N, D) per arm; compare
      `mtp` vs `base` at matched params/tokens
- [ ] Held-out lookahead proxy test (small synthetic completions requiring
      a token defined later in the same clause) to check whether `mtp`
      actually reduces teacher-forcing shortcut learning (Bachmann &
      Nagarajan 2024, arXiv:2403.06963), not just lowers loss

## Phase C — Code scaling sweep
- [ ] Small permissively-licensed code corpus (single language, comparable
      token count to Phase B), with natural docstring↔function-body pairs
      preserved — this pairing *is* the second "view" for `jepa-aux`
- [ ] All three arms in scope: `base`, `mtp`, `jepa-aux`
- [ ] Same 4-size sweep, same harness, no architecture changes
- [ ] `jepa-aux` metric: docstring→function nearest-neighbor retrieval
      accuracy in embedding space (did alignment learn shared semantics,
      not just shrink the loss number)
- [ ] Compare fitted exponents and per-arm deltas text vs. code

## Phase D — Combine & estimate
- [x] For each corpus, keep only arms that beat `base` outside noise at
      matched params/tokens — a dud arm does not get combined.
      **Result (500-step pass, val loss, xs/s/m): neither `mtp` nor
      `jepa-aux` beats `base` at any matched param count on either corpus.**
      `mtp`: consistently worse, shallower scaling exponent (0.050 vs 0.064
      on rj, 0.041 vs 0.054 on code) — more steps (150->500) did not close
      the gap. `jepa-aux`: apparent win at 150-300 steps was overfitting —
      only ~86 train pairs, val loss bottoms ~step 300 then rises. Nearly
      flat scaling exponent (0.019) — needs more paired examples before
      it's a fair test, not more steps.
- [ ] Nothing combined this pass — no arm cleared the bar. Before retrying:
      widen the code corpus (more stdlib modules -> more docstring/function
      pairs) and re-test `jepa-aux` alone; re-test `mtp` only if pursuing
      the longer training-scale regime the original paper used
- [ ] Latent steering probes (TSV/SAE-style, arXiv:2503.01917) on saved
      activations from the winning checkpoint(s) — measured as
      verified-accuracy + abstention-rate shift, never "hallucination-free"
- [ ] From the fitted laws, extrapolate params/compute needed at "real"
      scale (order-of-magnitude), stated against the current GPU's free
      capacity
- [ ] Go/no-go on the backlog below, based on measured numbers, not vibes

## Backlog — explicitly NOT started until Phase D says so
- [x] MoE routing layer — user overrode the gate (explicitly wanted it
      tested regardless of the data-ceiling finding). **Result (700-step
      base arm, matched active params via shared+top1 expert = dense MLP
      params, 4 routed experts total): dense beats MoE beats MoE+BitLinear,
      consistently, on both corpora.**
      rj: dense 4.361 / MoE 4.407 / MoE+BitLinear 4.424 (val loss, lower better)
      code: dense 4.815 / MoE 4.883 / MoE+BitLinear 4.934
      MoE also cost ~30-60% more wall-clock per run despite matched active
      FLOPs (naive per-expert masking loop, not a batched/grouped kernel —
      an implementation-overhead cost, not a FLOPs one). BitLinear
      (ported from uchi/uchi/flux/bitnet.py) added a further small, consistent
      regression on top, expected — ternary quantization is lossy and
      typically needs more training to recover, not less.
      Hallucination-gap probe (id_confidence - ood_confidence, see
      hallucination_probe.py) on the same 6 checkpoints was **inconsistent
      between domains** (rj: dense best at +0.037, MoE+BitLinear negative
      at -0.020; code: reversed, MoE+BitLinear best at +0.056, dense worst
      at +0.013) — small effect sizes, single seed, 3+5 prompts. Read as
      noise, not a finding, until re-run with multiple seeds/larger prompt
      sets. **Net verdict: no evidence MoE is worth its cost at this scale
      yet, on either the loss or hallucination axis** — confirms rather
      than overturns the original params-per-FLOP-ceiling gate.
- [x] Unified multimodal tokens — pixel + audio added. Synthetic single
      inputs (data/pixel/image.png, data/audio/clip.wav — same "one
      deliberate input" choice as R&J/stdlib, not scraped), small
      from-scratch VQ-VAE codecs (codec.py, van den Oord et al. 2017,
      not pretrained EnCodec/VQGAN — avoids a large download and keeps
      training CPU-fast), fed through the *identical* train.py/model.py
      pipeline as new --dataset options (pixel: 4096 tokens/64 codes,
      audio: 8000 tokens/64 codes). Confirms the mechanism: same
      architecture, same training loop, only the data source and vocab
      size change.
      **Update: joint unification done.** One shared vocab (text/code BPE
      0-1023, pixel codes 1024-1087, audio codes 1088-1151, +4 modality
      marker tokens 1152-1155), one model (`--dataset joint`), per-example
      modality sampling each batch (uniform 1/4) so the ~50K-token text/code
      corpora don't drown out the 4-8K-token pixel/audio ones. 700-step run:
      held-out loss dropped monotonically and simultaneously across all
      four modalities in the *same* model (rj 5.67->4.89, code 5.89->5.24,
      pixel 3.34->2.08, audio 2.59->0.99) — the mechanism works. Seeded with
      only a modality marker token (no other context), generation mostly
      stayed within that modality's token range (in-lane rate, single
      40-token sample per checkpoint, so noisy: code and audio mostly
      92-100%, pixel and rj more variable, one sample as low as 52%/62%).
      Real, occasional cross-modal leakage observed (e.g. rj generation
      drifting into audio-range tokens at step 700) — boundary-respecting
      is mostly learned, not perfect, at this toy scale/budget. Honest
      read: unification works as a mechanism; the in-lane metric needs
      multiple samples per checkpoint (not just one) before trusting its
      trend, unlike the val-loss numbers which are clean.
- [ ] Paired-view dataset for R&J (paraphrase or line↔scene-summary) if
      `jepa-aux` wins big on code and is worth extending to text
- [ ] Full JEPA-as-primary-objective (no token decoder) — rejected for
      this project, solves a different problem than "predicts the next
      token"; revisit only if the auxiliary-loss version clearly stalls

## Phase F — Swarm + Knowledge Graph (toy validation, per tasks/swarm.md)
Compressed scope vs. the full spec in swarm.md — see conversation record for
the full cut list. Reuses this session's existing harness/MoE/data, CPU-only.
- [x] Cut for the toy pass: RWKV backbone (use existing Transformer), 32k-50k
      vocab (use existing small shared vocab — contradicted by today's own
      finding that a 1024 vocab already hit a data ceiling at similar
      corpus size), IE-based text fact extraction/Neo4j/FAISS/sentence-
      transformers (plain Python graph instead), adaptive fast/slow
      inference paths, new data curation (reused cached rj + code)
- [x] Graph module (`graph.py`): AST-fact edges (code only) + co-occurrence
      edges (domain-agnostic), plain dict-based directed graph, no new
      dependency. **9307 total edges (212 AST facts, 9095 co-occurrence).**
      AST-fact precision problem found and diagnosed, not fully fixable:
      with a 1024-token vocab on a 5-module corpus, most identifiers are
      too rare to get their own BPE piece and fragment to near-character
      level, so "token at the AST boundary" is often a meaningless
      fragment (`'self'->'break'`) rather than a clean fact
      (`'import'->'Fraction'`, which works because "Fraction" is common
      enough in fractions.py to be one token). Filtered to reject
      single-character targets; genuinely fixable only with a bigger
      vocab/corpus or word-level (not BPE) fact tokens.
- [x] 3 query heuristics (not 6): local next-token, frequency-weighted,
      fact-grounded-only (`swarm.py`)
- [x] Swarm wrapper: neural logits (reused trained checkpoint) + graph-query
      suggestions per heuristic, confidence-weighted vote aggregation
- [x] Ran swarm.md's own 5 validation tests, adapted to rj + code:
      **Test 1 (routing collapse): PASS** — 4 experts on a joint rj+code
      MoE model, ~25% utilization each, no collapse.
      **Test 2 (code-vs-text specialization): FAIL** — JS divergence
      0.0000, routing distributions for rj and code are essentially
      identical (`moe_analysis.py`). Real negative result, not noise: two
      maximally different domains (Shakespeare dialogue vs. Python
      stdlib) produced zero learned routing specialization at 1.75M
      params / 700 steps, checked at the first MoE block.
      **Test 3 (graph extraction quality): partial** — edge count in
      range, but AST-fact precision issue above means the 90%+ precision
      bar isn't cleanly met at this vocab/corpus scale.
      **Test 4 (swarm vs. single expert): PASS** — 71% token diff (bar:
      >20%). **Test 5 (graph vs. no graph): PASS** — 79% token diff (bar:
      >10%). Caveat on both: sampled autoregressive generation cascades
      once any early token differs, so this magnitude likely overstates
      per-step graph/swarm influence — a tighter follow-up would compare
      single-step logit distributions at matched contexts, not full
      diverged sequences.
      **Net, per swarm.md's own decision rule: 1 clear fail (Test 2) +
      1 partial (Test 3) = "debug those specific components, re-test,"
      not "rethink the architecture."** Specialization (Test 2) is the
      one worth debugging first — try a deeper block, more steps, or an
      explicit domain-conditioning signal before concluding it can't work.
- [ ] Not yet done: comparing swarm+graph's held-out loss against today's
      dense/MoE baselines (Tests 1-5 check mechanism, not final quality,
      exactly as swarm.md itself says) — that comparison is the next step
      if Test 2 is fixed and specialization actually emerges

## Phase G — RWKV standalone test + salvaged concepts on TinyGPT
Per-concept follow-up after ditching swarm (Test 2 failed on all 4 layers,
confirmed, not just layer 0). RWKV and the graph tested separately, on
their own merits, not bundled with the failed swarm mechanism.

- [x] RWKV standalone (`rwkv_model.py`, `train_rwkv.py`): pure RWKV
      underperforms dense TinyGPT on val loss (rj: 4.74 vs 4.36, code: 5.22
      vs 4.81) — expected, a fixed-size recurrent state is more constrained
      than full attention, real RWKV needs more scale/tuning to close this.
      Not disqualifying on its own; see hybrid result below.
- [x] Unlimited-context claim (`test_unlimited_context.py`): **(a)/(b)
      confirmed cleanly** — state size exactly 6144 bytes at 512 through
      32,768 tokens (constant), wall-time scales ~4x per 4x length (linear,
      not quadratic). Processed the full 48,259-token rj corpus (377x
      TinyGPT's own block_size) in one continuous pass — structurally
      impossible for TinyGPT (`generate()` hard-crops to the last 128
      tokens). **(c) NOT demonstrated**: carried state showed zero (KL=0.0)
      measurable influence from 8,192-token-old context. Diagnosed cause,
      not a mechanism failure: the model was trained exclusively on
      128-token crops, so nothing ever rewarded slow-decay/long retention.
      Untested whether training with cross-chunk gradient flow would fix
      this — flagged as a real open question, not concluded either way.
- [x] Hybrid backbone (`model.py`: `use_rwkv_hybrid`, `attention_layers`) —
      mostly RWKV time-mixing blocks + periodic attention (1 attention
      layer of 4, matching uchi's own SSM+periodic-attention precedent).
      **Result: beats pure dense TinyGPT on val loss, on both corpora,
      reproducibly** — rj: 4.351 vs 4.361, code: 4.636 vs 4.815 (both
      matched at ~941K params, 700 steps). Also decisively beats pure RWKV
      (4.74/5.22). This is the new best backbone: same param budget, better
      loss than pure attention, and 3 of 4 blocks now carry the
      unlimited-context property. Single run each, not seed-averaged —
      real signal (consistent direction and magnitude across two very
      different domains), but not yet a claim to over-trust.
      **Bug found and fixed**: run naming didn't distinguish rwkv-hybrid
      runs, so the first hybrid run silently overwrote the original pure-
      dense checkpoints (`rj_base_m`, `code_base_m`). Loss numbers are safe
      (reproduced multiple times, recorded here before the overwrite); the
      checkpoint files themselves are gone. Fixed in `train.py` (`_rwkv`
      suffix) so it can't happen again.
- [x] Unified graph, third knowledge source (`graph.py`:
      `add_model_prediction_edges`) — single-model high-confidence
      predictions (>0.95, no expert-agreement gate since no swarm) added
      as edges, only when genuinely novel (skipped if the graph already
      has that edge — confirmation isn't new knowledge). Echo-chamber risk
      from the weaknesses list is real and not mitigated here beyond the
      confidence bar and the existing facts-override-statistics rule.
- [x] User corrections (`graph.py`: `add_user_correction`) — downweights
      (not deletes) a named wrong edge, adds the correct one at max
      confidence/provenance, takes effect on the next graph query with no
      retraining. Matches concept 4 (continuous updates without touching
      model weights).
- [x] Single-model abstention + fast/slow path (`inference.py`:
      `predict_next`) — fast path (confidence >0.85) skips the graph
      entirely; slow path blends in graph suggestions only when neural
      confidence is low, and abstains on low combined confidence *or*
      neural/graph disagreement. No voting, no multiple experts — concepts
      5 and 6 from the weaknesses list, working off one model's own
      calibration signal.
- [x] End-to-end demo: hybrid model + graph + model-prediction edges + user
      correction + abstention, all wired together, no crashes. Named the
      hybrid architecture **Ducky** (TinyGPT + RWKV/attention hybrid +
      unified confidence-gated graph + single-model abstention).
- [x] Threshold calibration (`inference.py`: `calibrate_thresholds`,
      `measure_confidence_distribution`) — first-pass thresholds (0.85
      fast / 0.3 abstain) were copied from swarm.md's production-scale
      numbers and were badly miscalibrated: Ducky's actual confidence
      distribution at 700 steps has median 0.125, p85 only 0.339, max
      0.964 — everything abstained, including in-domain prompts.
      Percentile-based recalibration (fast=p85≈0.40, abstain=p15≈0.056)
      grounds thresholds in this checkpoint's own behavior. Verified: fast
      path now fires at ~23% on real training-data contexts, matching its
      p85 target. Found a further real gap while checking, not smoothed
      over: the slow (graph-blended) path never successfully answers,
      only abstains, because it reuses the *raw-neural* threshold to gate
      a *combined* (neural+graph) confidence with a different scale —
      needs its own calibration pass, flagged as a follow-up, not fixed.
- [x] Re-verified unlimited-context specifically on the Ducky checkpoint
      (previously only checked on standalone pure RWKV) — state size
      exactly 4608 bytes (3 RWKV blocks' worth; the attention block
      correctly contributes none) from 512 through 32,768 tokens, time
      scaling ~4x per 4x length (linear). Closes the gap flagged earlier.
- [ ] Not yet done: separately calibrate the slow-path (graph-blended)
      threshold; repeat-seed check on Ducky's win margin over pure dense
      before fully trusting it; see `tasks/ducky.md` for the architecture
      write-up

See [`core_principle.md`](core_principle.md) for why this order and not the
obvious one.
