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
- [x] Slow-path threshold fix: calibrate_thresholds now measures the
      combined (neural+graph-blended) confidence distribution separately
      and conditionally (only over cases that actually reach the slow path
      with graph coverage), instead of reusing the raw-neural threshold on
      a differently-scaled distribution. Verified on 30 real training
      contexts: slow-answered went from 0/30 to 17/30, abstain from 23/30
      to 6/30 -- the mechanism now mostly answers and abstains only when
      it should, not by default.
- [x] Toward uchi's actual mission (verify against something real, abstain
      otherwise) while keeping next-token prediction as Ducky's core
      objective, not replacing it (`grounding.py`):
      - `verify_code_syntax` — ast.parse validity as a genuine grounding
        check, scoped down from uchi's full REPL/execute-and-check since
        generated snippets aren't runnable programs on their own.
      - `self_critique_score` — teacher-forced re-scoring of the model's
        own generated tokens; single-model analog of Devil's Advocate, no
        second expert needed.
      - `build_symbol_table` / `identifier_grounded` — checks *decoded*
        identifier strings against real function/attribute/import names
        from the corpus, sidestepping the BPE-fragment precision problem
        in graph.py's AST facts entirely rather than chasing better token
        alignment.
      - `build_ngram_index` / `ngram_grounded` — verbatim n-gram lookup
        against real source token sequences, a cheap stand-in for
        brain.uchi's retrieval index (no embeddings/vector DB).
      All four validated directly: syntax check correctly flags the
      model's still-garbled toy-scale output as invalid; self-critique
      score ~0.08 (consistent with the low confidence distribution already
      measured); identifier check correctly distinguishes a real vs.
      made-up symbol; n-gram check correctly distinguishes a real corpus
      slice from an arbitrary one.
- [x] Grew the code corpus 5 -> 10 stdlib modules (~150KB -> 425KB, 51K ->
      149K tokens; pairs.jsonl 106 -> 287), still hand-picked pure-Python
      modules, not a switch to scraped data. rj left untouched — a second
      book would violate the original single-input design from the start
      of the session. Deliberately did NOT retrain the tokenizer to avoid
      invalidating every checkpoint trained this session (same vocab_size,
      same token-id meanings) — only the `code` token cache needed
      regenerating. Code corpus is now ~2.8x bigger than rj; noted as an
      asymmetry for future joint-training runs, not fixed here.
- [x] Wired grounding.py's four signals into inference.py, split by the
      granularity they actually operate at rather than forced into one
      shape: `ngram_grounded` is per-token, folded into `predict_next`'s
      own disagreement branch (real evidence a sequence occurred in
      source can rescue an otherwise-disagreement-triggered abstention,
      but never overrides a pure low-confidence abstention). The other
      three (`verify_code_syntax`, `self_critique_score`,
      `identifier_grounded`) are inherently span-level, not single-token
      — added as a new `generate_with_grounding` wrapper that runs
      `predict_next` repeatedly (stopping immediately on ABSTAIN, not
      guessing past it), then verifies the completed span. Tested end-to-
      end on 5 real code prompts: honest, not flattering -- most
      generations stop within 1-4 tokens (self-critique 0.09-0.26, syntax
      invalid on the short fragments), which is the correct behavior for
      an honestly-calibrated 700-step toy model, not a bug to chase.
- [x] Repeat-seed check (3 seeds each) + fresh retrain on the grown code
      corpus. **Ducky (hybrid) wins 6/6 seeds across both corpora**, but
      the two corpora tell different-strength stories:
      rj (unchanged corpus): dense mean 4.3619 (std 0.0008, very stable),
      hybrid mean 4.3265 (std 0.0236), margin mean 0.0354 but margin std
      0.0234 — nearly as large as the mean. Real, consistent-direction win,
      but the *size* of the win is noisy; don't over-quote a precise number.
      code (grown corpus, 51K->149K tokens): dense mean 4.0625 (std
      0.0305), hybrid mean 3.9109 (std 0.0121), margin mean 0.1516, margin
      std 0.0346 — small relative to the mean. Strong, reliable win.
      Also confirms corpus growth mattered independent of architecture:
      code's dense baseline alone dropped from 4.815 (old 51K-token
      corpus) to ~4.06 (new 149K-token corpus, averaged across seeds) —
      a bigger jump than the architecture choice produced on its own.
      Naming bug fixed proactively before this run (same collision class
      as the earlier MoE/rwkv bugs): added a `_seed{N}` suffix so repeat
      seeds don't overwrite each other's checkpoints.

## Phase H — "Do all 6" (making Ducky better)
- [x] BitLinear wired into Ducky's own blocks (`model.py`, `--use-bitlinear`)
      -- attention/RWKV projections + dense MLP, not just the abandoned MoE
      experts. Moved BitLinear/quantizers to their own `bitnet.py` module to
      avoid a circular import between model.py and rwkv_model.py.
- [x] TensorRankEmbedding (`model.py`) -- uchi-style low-rank factored
      embedding, `--embedding-rank N`. Real savings even at our small 1024
      vocab: rank 32 cuts 940,800 -> 846,592 params (~10%), more than the
      initial "probably not worth it at this vocab size" assumption. Output
      head reuses the same factorization transposed (uchi's own symmetric
      projection) -- not uchi's separate syntax-prediction DualHead, which
      needs labels we don't have; noted as a scope reduction, not hidden.
- [x] Cross-chunk BPTT training (`train_bptt.py`) -- K=4 consecutive
      128-token chunks per step, RWKV state carried (not detached) across
      all K, so gradients from later chunks can shape how earlier chunks
      were processed. The only training regime that could actually reward
      long-range retention, since isolated-crop training provably couldn't
      (KL=0.0 recall test result). Ran on rj and code, 700 steps each.
- [x] Extended-step ceiling sweep (dense/hybrid/hybrid+BitLinear) on the
      grown code corpus, 2000 steps. **Found a new, later ceiling**: dense
      bottoms at step 1250 (val loss 3.921), notably better than the
      700-step number (~4.05 avg across seeds) -- confirms training had
      been cut short before, not that the model had converged.
- [x] jepa-aux re-tested with the grown pairs dataset (86 -> 287 pairs).
      **Overfitting problem genuinely fixed**: smooth curve, bottoms at
      step 600 (val_code_lm_loss 4.155), train/val gap real but no longer
      catastrophic. Still doesn't beat plain base on code_lm_loss alone
      (3.921) -- not a win yet, but no longer a diagnosed failure either.
- [x] Extended sweep complete. **Both dense and hybrid bottom out at the
      exact same step (1250)** on the grown code corpus -- strong evidence
      the ceiling is set by the data, not the architecture. Dense: 3.921.
      Hybrid: 3.826 (still winning, margin 0.094). Hybrid+BitLinear: still
      improving at step 2000 (3.931), hadn't found its ceiling within the
      tested budget -- quantization needs more steps to recover, as
      expected from the MoE+BitLinear finding earlier in the session; not
      yet comparable to the other two until it actually converges.
- [x] BPTT long-range recall re-test: **negative, and more decisive than
      "not yet tested."** KL divergence is 0.0000 at every horizon checked
      (128 through 640 tokens) -- including 512 tokens, exactly the span
      BPTT training (K=4 chunks) was designed to cover. Carrying state
      forward measurably changes nothing, even within the mechanism's own
      training horizon. Diagnosed, not just observed: (1) 700 BPTT steps
      is a small budget to shift learned per-channel decay rates away from
      their aggressive default, and the cross-chunk gradient signal is
      real but likely weak relative to the dominant within-chunk
      prediction signal; (2) more structurally, the hybrid's attention
      layer gives the model an escape hatch -- it can hit low loss on each
      chunk via full local attention alone, with no pressure to ever rely
      on the RWKV layers' carried state. Nothing forces the mechanism to
      be used just because it's available. Open question for a future
      pass: would a pure-RWKV (no attention escape hatch) BPTT run, or a
      much larger BPTT step budget, actually induce retention? Not tested.
- [x] Grounding/abstention net-positive check (`eval_grounding.py`) --
      selective-prediction evaluation: is accuracy on answered
      (non-abstained) tokens actually higher than the pure-neural
      unconditional baseline? **Yes, on both domains:**
      rj: baseline 18.0% accuracy, grounded-on-answered 23.3% (+5.3 points,
      71.2% coverage). code: baseline 22.4%, grounded-on-answered 25.7%
      (+3.3 points, 67.0% coverage). Real, consistent-direction signal on
      both corpora -- this is the first thing this session that actually
      closes the "difference vs. improvement" gap the original swarm.md
      postmortem flagged, and that ducky.md had left as an open question.

## Phase I — Training efficiency
- [x] CPU thread tuning: 8 threads measured optimal (0.209s/step) vs default
      10 (0.252s/step) vs 16-20 (0.91-1.31s/step, 4-6x slower from
      thread-sync overhead) -- now train.py's default.
- [x] `torch.compile` on the WKV scan: 2.65x steady-state speedup
      (0.5757s/step eager -> 0.2174s/step), ~169s one-time compile cost,
      numerically verified identical (diff ~1e-7). `UCHI_FUSE_SSM_SCAN=0`
      escape hatch, same name uchi itself uses.
- [x] `--compile-full-model`: compiles the whole forward pass, 3.95x
      speedup (0.146s/step, beats uchi's own reported 3x), ~480s compile
      cost, breakeven ~4300 steps -- opt-in, not default. Found + fixed a
      real bug while testing (overly-broad `replace_all` renamed a
      parameter but not its body reference).
- [x] bf16 tested and **rejected**: only ~13% speedup against real,
      compounding precision loss (up to 8.5% of a std dev, worst case) in
      a 128-step sequential recurrence.
- [x] Automated early stopping (`--patience`/`--min-delta`, default
      disabled): verified stopping at step 900 instead of running the full
      2000-step budget, correctly identifying the same best checkpoint
      every prior extended sweep had to discover by running long.
- [x] Ducky SDK setup caching (graph + calibration, keyed by checkpoint
      mtime): 10.71s cold -> 2.24s warm, identical output confirmed.

## Phase J — Scaling: vocab, depth, proportional growth
- [x] Chinchilla-ratio pushback: 941K params was already ~125x
      over-parameterized for a 149K-token corpus. User chose "grow data +
      params together, proportionally" over blind 100x param growth.
      Empirically confirmed: dense 'l' (5.03M params) on the grown corpus
      hit 3.027 val loss vs 'm' size's 3.921 on the same corpus -- large
      win from doing both together, not params alone.
- [x] Code corpus grown 10 -> 49 stdlib modules (2,026,710 chars,
      1,311 pairs) -- still hand-picked, not scraped.
- [x] Tokenizer regrown 1024 -> 8192 vocab: real compression confirmed
      (14 tokens vs 20 for the same test sentence; common words now single
      tokens). Fixed a real bug: `MODEL_PREFIX`/tokenize-cache filenames
      weren't versioned by vocab size, so the vocab-8192 run had already
      silently overwritten the shared code/rj token-id cache with new-vocab
      ids mid-session -- fixed by versioning both (`spm_{vocab}.model`,
      `{name}_{vocab}.pt`).
- [x] New 'xl' size preset (256 d_model, 12 layers, 8 heads) + rank-64
      factored embedding (10.05M params, controls the 8x-bigger embedding
      table's cost). New-generation hybrid xl beat dense xl again: best val
      5.1643 vs 5.2873 (not comparable to pre-8192-vocab numbers -- higher
      cross-entropy floor from the bigger softmax, same-vocab comparisons
      only). Same architecture win reproduced at the new scale.
- [x] `config.json` now records `vocab_size` explicitly (train.py) so no
      future checkpoint needs guessing which tokenizer generation it used.

## Phase K — Ducky SDK: ask()/learn(), reject-and-resample, session memory
- [x] `ducky.py` built: `Ducky(domain, backbone).ask(prompt)` /
      `.learn(text)`, mirrors uchi's own ask/learn contract. Both
      `backbone="hybrid"` (default) and `"dense"` supported side by side.
- [x] Reject-and-resample (`generate_with_resampling`, `ask(n_candidates=N)`):
      required first fixing that `predict_next` was fully deterministic
      (added a `temperature` param; abstention decision still always uses
      greedy confidence, unaffected). Verified: resampling 8 candidates
      selected a *lower*-self-critique but syntax-*valid* completion over
      the deterministic path's higher-self-critique but invalid one (1.024
      vs 0.149 scored) -- proof the selection logic does real work.
- [x] Session-scoped working memory (`session_memory.py`'s `SessionTrie`):
      catches self-contradiction within one generation (different token
      chosen for an identical trailing context), distinct from the
      corpus-level graph/n-gram signals. Hash-chained keys (not
      single-token hashes -- 8192-token vocab makes those trivially
      reversible), observability-only so far.
- [x] Old (vocab=1024) and new (vocab=8192) checkpoint generations now
      coexist correctly through the same SDK: `Ducky()` reads each
      checkpoint's own `vocab_size` from its config (falling back to 1024
      for pre-versioning checkpoints) instead of assuming the SDK default.
      Regression-verified: rj (old-gen) and code (new-gen) both load and
      answer correctly in the same process.

## Phase L — uchi-inspired reasoning mechanisms + honest benchmark
Stated goal: scale Ducky toward eventually scoring on MMLU/SWE-bench.
Reality check given first: at ~10M params / ~2M training characters,
Ducky is 6-7 orders of magnitude below the scale/data either benchmark
needs -- agreed as intentional, Ducky is the testbed, scale-up comes
later. Built and validated four mechanisms anyway, each checked with real
(non-cherry-picked) output before combining them:
- [x] `mcts_lite.py` -- PUCT/UCB search over chunk-level generation
      branches, `self_critique_score` as an honest (untrained, unlike
      uchi's own) value proxy. Verified: explores genuinely differently
      than best-of-N (different completion, different score).
- [x] `repair_loop.py` -- sequential, feedback-informed retry (real
      SyntaxError text spliced into the next attempt's prompt, not a blind
      resample), max_attempts=4, tri-state PASS/FAIL/ABSTAIN outcome.
- [x] `session_history.py` -- extractive (never paraphrased), bounded,
      RAM-only cross-call memory, opt-in via `track_history=True`.
- [x] `check_call_arity_consistency` (`grounding.py`) -- narrow,
      deterministic, additive-only veto (same discipline as uchi's
      relational_reasoning.py). Verified against synthetic
      consistent/inconsistent/unparseable cases.
- [x] `bench_ducky.py` -- 10 held-out docstring->function-body tasks,
      graded by executing generated code against real asserts in a
      restricted, timeout-guarded sandbox. Harness verified correct first
      (canned-correct=100%, wrong answer fails, infinite loop times out).
- [x] **Real benchmark result: 0/10 on all four mechanisms** (baseline,
      resample, MCTS-lite, repair loop). Every completion tops out at 1-2
      tokens before abstaining -- nothing for any mechanism to search/
      retry/remember over. Confirms the reality check: reasoning
      scaffolding amplifies existing capability, it can't manufacture
      capability the base model doesn't have yet. All four mechanisms are
      built and ready for when base-model scale catches up.

## Phase M — BPTT long-range retention: five rounds, all decisive, all negative
- [x] Round 1 (K=4 chunk BPTT, 700 steps): KL=0.0000 at every horizon
      (128-640 tokens). Diagnosed two candidate causes: attention "escape
      hatch," insufficient steps.
- [x] Round 2 (pure RWKV, escape hatch removed): identical null result --
      rules out the escape hatch. 5000-step budget also tested: made
      things *worse* (best val at step 500, then exploded), ruling out
      "just needs more steps" in the opposite direction.
- [x] Round 3 ("big data + big budget together," grown 387K-token corpus,
      vocab=8192, 3000 steps with early stopping): still KL=0.0000-0.0001
      at every horizon. Fixed a real bug found along the way:
      `train_bptt.py` saved the *final* checkpoint to a file misleadingly
      named `..._best.pt`.
- [x] Round 4 (`train_stream_bptt.py`, true sequential streaming with
      Transformer-XL-style truncated BPTT, gulp in {4,8,16,32} = 512-4096
      token gradient spans, extended retention eval to horizons up to
      8192 tokens): still KL=0.0000 at every horizon, every gulp size --
      a genuinely different regime (fixes random-restart windows' inherent
      "no example ever has real info before its own boundary" flaw), still
      negative.
- [x] Round 5 (`--final-chunk-only-loss`: score only the last chunk of
      each gulp, forcing earlier chunks to matter only via the carried
      state): still KL=0.0000 at every horizon, every gulp size -- the
      most targeted test possible, objective can't be satisfied without
      using the state, still no retention.
- [x] **Conclusion: five-for-five across corpus size, step budget,
      architecture, training regime, and the objective itself.** Very
      unlikely to be a training-recipe problem; most consistent
      explanation is vanishing gradients through 4000+ steps of recurrence
      at this model's scale (~1.86-10M params) -- the same well-documented
      reason LSTMs/GRUs replaced vanilla RNNs historically. Recommendation
      (given to and accepted by the user): stop active pursuit at toy
      scale. The unlimited-context property remains real and structurally
      verified (constant memory, linear time), just not exploitable by
      training at this size -- revisit only once base-model scale grows
      substantially.

See [`core_principle.md`](core_principle.md) for why this order and not the
obvious one. See [`ducky.md`](ducky.md) for Ducky's own architecture record
in full detail -- this file tracks the compressed plan; ducky.md tracks the
complete evidence trail.
