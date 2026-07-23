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

## Phase N — Next architectural lever: confidence-gated conditional compute
Architecture critique (see `ducky.md`'s new "Architecture critique" section)
ranked three genuinely untested levers, after confirming two axes are
exhausted at this scale (recurrence-block-type shopping and reopening
MoE/swarm, both dominated by data-scale effects or the same proven
data-diversity failure mode). Items 1 and 2 implemented and tested this
round (kept to a small memory/time budget -- `rj` domain, vocab=1024, "m"
size -- a concurrent process was using real RAM); item 3 stays backlog:
- [x] Confidence-gated conditional compute / Mixture-of-Depths-style probe
      (Raposo et al. 2024, arXiv:2404.02258; `eval_early_exit.py`, zero
      training, "logit lens" reuse of the model's own trained head at
      intermediate depth). Result: at threshold=0.5, 2.7% avg compute
      saved for a noise-level -0.4pp accuracy cost; at threshold=0.3,
      13.15% saved but a real -3.8pp accuracy cost. Honest conclusion: a
      zero-training probe doesn't cleanly separate "safe to exit early"
      from "needs full depth" at this checkpoint's scale -- doesn't kill
      the idea (CALM/MoD train the exit decision jointly, a materially
      different test not attempted this round), but rules out "free
      compute savings with no training" as the easy win. See `ducky.md`
      for full numbers.
- [x] Layer weight-sharing (Universal-Transformer/ALBERT-style tied
      blocks, `GPTConfig.tie_layers` in `model.py` + `--tie-layers` in
      `train.py`). Result: 940,800 -> 345,984 params (63.2% cut to the
      block stack), best_val 3.6623 (untied) vs 3.6845 (tied) -- a real
      but small 0.0222-nat gap. Real caveat: parameter-count win, not a
      compute win (tied model needed *more* steps/wall-clock to converge,
      same FLOPs/token).
      **Follow-ups completed:** width-reallocation (d_model=224,
      863,520 params) made things *worse* (3.7121), not better -- the
      naive "reinvest freed params into width" version of the idea
      doesn't work at this scale. Repeat-seed (3 seeds, `rj`): untied wins
      all 3, gap 0.0053-0.0330 (mean~0.02) -- direction confirmed.
      **Code-domain check: gap is +0.2165, ~10x bigger than any rj gap** --
      tie_layers costs substantially more on code, the exact domain it was
      proposed for. See `ducky.md` for the full table; this is now a
      corrected, not just extended, finding.
- [x] Input-selective decay (Mamba/S6-style), first-rung check (per the
      small-scale-first rule in `core_principle.md`): new `mamba_lite.py`'s
      `SelectiveTimeMixing`, sanity-verified (forward/backward correct)
      then trained matched at `rj`/m scale. **Initial result: selective
      decay (990,336 params) beat plain RWKV hybrid (941,184 params) --
      3.5635 vs. 3.5886 -- both beat dense (3.6623).** A promising first
      try, but single-point/single-domain -- **superseded below by the
      proper multi-size/multi-domain validation.**
      Rebuilding the cross-chunk BPTT + retention-eval harness (deleted in
      an earlier cleanup) to test the original retention question is still
      not attempted -- stays backlog regardless of the verdict below,
      since it answers a different question (retention, not base loss).
- [x] **Real bug found while testing the above -- fixed, not just flagged.**
      `runs/rj_base_m` (and `rj_base_m_seed1`, the two pre-versioning
      checkpoints `ducky.py`'s `DEFAULT_RUNS` actually serves) were trained
      under the original unversioned `data/tokenizer/spm.model`, not
      today's `spm_1024.model` -- same vocab size, different BPE merges/
      token-ID meanings. Fixed via `Tokenizer(model_path=...)` (bypasses
      the vocab_size/variant naming convention) + `ducky.py` routing
      no-`vocab_size` checkpoints there instead of guessing. Verifying
      this surfaced a **second, deeper bug**: `data.py`'s tokenize cache
      was keyed only by `vocab_size`, so the legacy and retrained
      `spm_1024.model` collided on the same cache file -- fixed via
      `Tokenizer.cache_key` (verified identical to `str(vocab_size)` for
      every existing plain caller, zero disruption to `code`/`text`
      production paths). Verified end-to-end: `rj_base_m` now reproduces
      val loss 4.3855 (matches recorded 4.3746) instead of 8.638;
      `Ducky(domain="rj").ask(...)` runs correctly.
- [x] Trained confidence-gated halting head (`GPTConfig.use_halting`,
      `model.py`'s `halting_loss`, `--use-halting`/`--tokenizer-variant`
      wired into `train.py`) -- the trained counterpart to the untrained
      logit-lens probe above. **Result: clearly better than the untrained
      probe** -- at threshold=0.7, 14.5% compute saved for -0.002 accuracy
      cost (untrained probe's best near-zero-cost point: only 0.75%
      saved); at threshold=0.5, 47.45% saved for -0.056 cost (untrained's
      most aggressive setting at *any* threshold only reached 13.15%).
      Real cost: the auxiliary objective added +0.10 nats to the primary
      LM loss at this toy scale (`HALT_AUX_WEIGHT` untuned). New
      `eval_halting.py` mirrors `eval_early_exit.py`'s exact methodology
      for direct comparability. See `ducky.md` for the full table.

## Phase P — small-scale-first, codified
`tasks/core_principle.md` now states explicitly (not just by precedent)
that every architectural idea proves itself at toy scale before any xl/xxl
commitment -- Phases N/O/P above are the pattern this makes into a rule
rather than an instinct. Nothing left to do here; this phase exists so the
rule has a place in the compressed plan, not just the North Star file.

## Phase O — Tokenizer fairness: does the shared BPE vocab dominate one domain?
Follow-up to Phase N's "token efficiency" thread. Diagnostic + small-scale
candidate comparison done this round (kept cheap -- SentencePiece only, no
LM training, another process was pinning CPU/GPU); production tokenizer
migration is a deliberate non-decision, not started:
- [x] Sourced NL2Bash (arXiv:1802.08979, `extract_terminal_corpus.py`) --
      12,607 real bash one-liners, 574,351 chars, exact expected count.
- [x] Built `eval_tokenizer_fairness.py` (fertility + Gini across domains,
      works against any `.model` file). Baseline result on the current
      production `spm_32768.model`: text 0.2467, code 0.2395, terminal
      0.4070 fertility, Gini=0.125 -- confirms terminal (never in
      training) pays a real ~65-70% token-count penalty vs. text/code.
- [x] Trained 3 small candidates (`build_candidate_tokenizers.py`,
      vocab=8192, ~209MB peak RSS combined, production tokenizer
      untouched): naive-concat BPE (+terminal data), resampled BPE,
      resampled Unigram. Results: just including terminal data in
      training is the single biggest lever (Gini 0.125->0.079);
      resampling adds a further, real but modest improvement (Gini
      0.079->0.070, small trade-off cost to text); Unigram does not help
      here -- worse compression on every domain than BPE on the identical
      resampled corpus, a real measured negative result. Recommendation:
      include terminal-command data, use stratified resampling, keep BPE
      -- not Unigram. See `ducky.md` for the full comparison table and
      confounds (vocab size 8192 vs. production 32768, small vs. full
      corpus -- directionally clear, not a production-scale-validated
      magnitude).
- [x] Side finding: code round-trip (encode/decode) already loses
      indentation/newlines on the *production* tokenizer too (SentencePiece
      default normalizer collapsing whitespace), not something the new
      candidates introduced -- flagged, not fixed this round.
- [x] **Real, full-scale production tokenizer built and validated
      (`spm_32768_balanced.model`, `build_production_tokenizer.py`) --
      tokenizer only, per the user's explicit choice, given the
      code/xl checkpoint's own measured ~3.8-hour retrain cost and the
      concurrent CPU/GPU load.** Versioning fixed first, additively
      (`tokenizer.py`'s new `variant` param -- default behavior
      unchanged, verified). Alpha picked empirically (swept 1.0/0.5/0.3
      on the small samples) then corrected for real domain-size ratios
      (~2300:300:1, not the small comparison's ~4:4:1): alpha=0.3 looked
      marginally better small-scale but implies repeating NL2Bash ~156x
      at full scale; alpha=0.5 (~39x) chosen to avoid that regime. Built
      via streaming line-by-line I/O (never holds a full domain in
      memory), 1,886MB peak RSS for the full ~1.5GB combined corpus.
      **Result: Gini 0.1250 -> 0.0930 (-25.6%), terminal fertility 0.4070
      -> 0.3561 (-12.5%), text/code fertility unchanged or slightly
      improved (0.2467->0.2474, 0.2395->0.2386)** -- the fairness gain
      held at real scale, better than the small comparison's own
      text-pays-a-cost pattern predicted. See `ducky.md` for the full
      table and reasoning.
- [ ] **Still not started, explicit non-decision, unchanged:** retraining
      any checkpoint on `spm_32768_balanced.model`. `code_base_xl_rwkv_
      rank64`, `text_base_xl_rwkv_rank64`, and `rj_base_m*` all still run
      on `spm_32768.model`, confirmed untouched. Only worth doing with an
      explicit go-ahead, given it requires retraining every downstream
      checkpoint -- the same costly migration already paid for twice
      (vocab 1024->8192->32768), now a third time if undertaken.

## Phase Q — Scaling-law simulation tool + real selective-decay verdict
User's top priority: a genuine way to "simulate large scale" cheaply.
Also incorporated the new tokenizer for real (training use, not a
production retrain, per the user's explicit choice not to retrain
anything this round):
- [x] `src/fit_scaling_law.py` -- log-log power-law fit (Kaplan/Chinchilla),
      verified against a synthetic known power law first (recovers
      `a=10, alpha=0.3` exactly) before trusting it on real data.
- [x] `src/run_scaling_sweep.py` -- 12 configs (`xs`/`s` x `rj`/`code_core`/
      `terminal` x rwkv/selective), trained under `spm_32768_balanced.model`
      for the first real use of the new tokenizer. Peak RSS 3,516.8MB.
- [x] **Verdict, reversing the single-point result above: promote
      selective decay = False.** Selective decay led at `xs` on 2/3
      domains, then lost on all 3 at `s` -- its relative edge shrinks with
      scale instead of growing. Fitted extrapolation picks RWKV on all
      three domains; selective decay's own fitted `alpha` is *negative* on
      `code_core` and `terminal` (predicts getting worse with more
      capacity, in this range). Exactly the failure mode a single-domain,
      single-size check can't catch -- see `ducky.md` for the full tables
      and honest caveats (2 close-together size points, vocab=32768's
      entropy floor likely still dominant at this scale, so a firm verdict
      at real production scale would need `m`/`l` points too).
- [x] **Explicit non-decision, unchanged: no checkpoint retrained.** The
      next production retrain should use the new tokenizer + RWKV
      (unchanged), not selective decay, based on this round's evidence.

## Phase R — Halting+tokenizer, confidence-gated width, adjacent-only selective decay
Three follow-ups, toy-scale, resource-conscious:
- [x] Halting under the new tokenizer (`rj`/m/dense, vocab=32768/balanced,
      no embedding-rank -- `n_params=5,004,548`). best_val=5.7857, not
      directly comparable to the vocab=1024 result (3.7639) given the
      much higher entropy floor. `eval_halting.py`'s accuracy-delta metric
      is inconclusive here (`full_depth_accuracy`=0.4%, 2/500 samples --
      too sparse to mean anything), but confidence and agreement-with-
      final-layer still rise sensibly with depth -- the mechanism's
      internal signal looks coherent, just needs a bigger/better-trained
      model to validate the accuracy trade-off cleanly at this vocab.
- [x] **New mechanism: confidence-gated width** (`WidthGatedMLP`,
      `GPTConfig.use_width_gating`, `TinyGPT.width_sparsity_loss`,
      `train.py`'s `WIDTH_SPARSITY_WEIGHT`) -- the width-axis analog of
      halting's depth-axis question, sanity-verified before training.
      Matched against `rj_base_m_seed57` exactly (`rj`/m/dense,
      vocab=1024): 941,316 params, best_val=3.7242 (+0.062 nats, a smaller
      cost than halting's +0.10). Sparsity regularizer worked (avg gate
      0.163, mostly closed) but **the gate does not correlate with actual
      correctness** (avg gate on correct predictions 0.1696 vs. incorrect
      0.1605, n=119/381 -- noise-level difference). Weaker result than
      halting: learns to be sparse, not to be *selectively* sparse.
      Also: the soft multiplicative gate never skips real compute (scales
      activations, doesn't index/prune them), so even a working version
      wouldn't yield real FLOP savings without a harder sparse
      implementation -- a further reason this is a weaker candidate than
      halting's literal early-exit.
- [x] Selective decay restricted to attention-adjacent layers
      (`GPTConfig.selective_decay_layers`, backward-compatible
      generalization of `use_selective_decay`). Reused the scaling
      sweep's own `rj`/`code_core` size-s data for 2 of 3 comparison
      points; 2 new runs for "adjacent-only". **RWKV still wins both
      domains** -- adjacent-only gives a real partial recovery on
      `code_core` (7.0988, vs. 7.1187 uniform-selective, 7.0639 RWKV) but
      is slightly worse than uniform selective decay on `rj` (6.7640 vs.
      6.7600). Verdict from the scaling sweep stands: no form of selective
      decay tested so far (uniform or adjacent-only) beats RWKV. A full
      Mamba-2/S6 implementation (selective B/C, not just decay) remains
      untested and is a materially bigger undertaking.
- [x] **No checkpoint retrained this round either** -- all three
      experiments stayed toy-scale, matching the resource-conscious
      discipline explicitly repeated at the start of this round.

## Phase S — Ensemble check + grounded self-training "flywheel", `s`-size only
- [x] `eval_ensemble.py`: probability-averaging 3 fresh, differently-
      seeded `s`-size models (223,808 params) beats the best single model
      on both domains tested -- `rj`: NLL improves (3.7357 vs. 3.817),
      accuracy flat; `code`: both accuracy (0.174 vs. 0.166) and NLL
      (4.4529 vs. 4.4661) improve. Distinct from the rejected swarm/MoE
      idea (that needed specialization that never appeared; a plain
      ensemble only needs decorrelation, already measured to exist via
      the repeat-seed check).
- [x] `run_flywheel.py`: round-0 baseline 0/10 (`bench_ducky.py`, ensemble
      and all 3 single members), matching the historical pattern. Real
      methodological finding along the way: the first verification gate
      (`verify_code_syntax` + `check_call_arity_consistency`) let 6/10
      comment-degenerated-gibberish completions through as "verified" --
      caught by reading the actual pooled text before retraining on it,
      not after. Fixed with a new, reusable grounding signal
      (`grounding.has_real_statement` -- does the function body have a
      real statement beyond its own docstring). **With the corrected
      gate: 0 Tier-1, 0 Tier-2, 0 verified fuel. No retrain performed --
      nothing to retrain on.** Honest verdict: the flywheel has nothing to
      spin on at `s` scale on code, consistent with the project's most
      repeated finding (scaffolding amplifies capability, doesn't
      manufacture it) -- now doubly confirmed since the gate that would
      have said otherwise was caught and fixed first.

## Phase T — Code's Chinchilla-matched "minimum viable" size, real data, real result
- [x] `safe_tokenize_breadth.py` -- chunked/streaming tokenization of
      `corpus_breadth.txt` (167MB) under `spm_32768_balanced.model` for
      the first time, specifically to avoid repeating this session's
      earlier ~19.5GB RSS spike from a single-call tokenization. **Peak
      RSS 2,917.9MB** -- verified an order of magnitude safer, not
      assumed. 43,263,940 real combined code tokens now known exactly
      (core 2,528,834 + breadth 40,735,106).
- [x] Computed (not guessed) the Chinchilla-optimal size for that real
      token count (2,163,197 params) and searched real `GPTConfig`s for a
      close match: `d_model=128, n_layer=6, n_head=4, embedding_rank=32`,
      RWKV hybrid -> 2,259,584 params (within 4.5%).
- [x] Trained one model (real weighted `code_core`/`code_breadth`
      sampling, `spm_32768_balanced.model`, 11,000-step ceiling,
      patience=6). Took ~2.3 hours (longer than estimated -- RWKV hybrid's
      known per-step overhead wasn't fully accounted for in the original
      25-50 min estimate), peak RSS stayed healthy/stable throughout
      (checked directly at multiple points during the run, no growth
      trend). **best_val=3.6779 at best_step=11000 (full ceiling,
      patience never triggered -- may still have room left in this
      budget).** Dramatically better than the toy vocab=32768 scaling-
      sweep points (5.6-7.1, 25K-158K block-stack params) -- direct,
      measured evidence for the Chinchilla-ratio thesis this project has
      repeated all along.
- [x] `eval_chinchilla_min.py` against `bench_ducky.py`: **still 0/10**,
      but a real qualitative shift -- completions are built from genuine
      Python idioms (type checks, error handling, real library calls),
      not gibberish, failing on malformed f-strings/brackets and
      generation drifting past the function's natural end. Same
      "coherent but not capable" pattern already documented for the much
      bigger (11.6M+ param) production checkpoint, reached here at ~5x
      fewer params. Honest caveat: `max_new_tokens=48` cutting generation
      short is a real, untested confound in some of these failures --
      not retested at a longer budget this round.

## Phase U — Dreaming test + the 3 actionable weaknesses
- [x] **Part A, self-distillation dreaming: real negative.** Frozen
      teacher (Phase T's Chinchilla-min checkpoint) generated 30 dreamed
      continuations; student distilled via KL on the teacher's softened
      targets over those dreams only. `held_out_loss` went **4.0813 ->
      4.2888** (worse), `bench_ducky` stayed 0/10 -> 0/10. Plausible cause:
      400 distill steps over 30 sequences (~107 revisits/seq) overfit to
      a narrow self-generated set -- a small-scale direct confirmation of
      the model-collapse risk (Shumailov et al. 2024) flagged before
      running it. This dreaming design doesn't work at this scale/config;
      more dream diversity relative to steps is the natural untested
      follow-up.
- [x] **Part B2, stopping criterion: implemented, and it falsifies last
      round's over-generation theory.** New `grounding.is_complete_statement`
      wired into `generate_with_grounding` (`stop_when_complete=True`
      default). Through `Ducky.ask()`, abstention already truncates first.
      Through plain greedy decode: `stopped_complete=False` on all 10
      `bench_ducky` tasks -- completions never reach a valid-parse state,
      confirming the 0/10 gap is genuinely about capability, not
      generation running past a near-miss.
- [x] **Part B3, grounding-signal audit: one clean, one real gap found and
      fixed.** `check_call_arity_consistency` already honestly scoped
      (own docstring discloses the limitation). `identifier_grounded` had
      a real gap -- single-letter identifiers (x, a, n) match almost any
      real symbol table, tested against 22,537 real identifiers from
      `corpus_core.txt`. Fixed with `min_length=3`; verified against both
      the buggy case and the still-correctly-failing "flatten_one_level"
      case.
- [x] **Part B4, reach validated wins through the real SDK.** Fixed
      `Ducky.__init__` to read `tokenizer_variant` from `cfg_dict` (was
      silently loading the wrong tokenizer for any variant-trained
      checkpoint -- same bug class fixed once already this session for
      the legacy pre-versioning case); registered `chinchilla_min`
      permanently in `train.py`'s `SIZES`. New `EnsembleModel` (averages
      softmax probabilities across N loaded checkpoints behind `TinyGPT`'s
      exact call contract) + `Ducky(ensemble_run_names=[...])` -- verified
      end-to-end with 3 real seed-varied checkpoints through the full
      `ask()` pipeline, zero changes needed to any existing caller.

## Phase V — Calculator-grounded arithmetic + a step-sequencer alternative
Investigated `/home/redleadr/workspace/uchi` (predictor.py, tool_calling.py,
numeric_plausibility.py) before writing any code, per the user's explicit
"nail the concept first." Scope confirmed: pure numeric expressions only,
`UniversalPredictor` tested as a step-sequencer only, never a value-producer.
- [x] **`evaluate_arithmetic`/`find_arithmetic_expression`/
      `arithmetic_grounded` (`grounding.py`)**: restricted AST evaluator
      (whitelist BinOp/UnaryOp/numeric Constant, never `eval()`) + detection/
      verification signals, same abstain-on-inapplicable convention as
      `check_call_arity_consistency`. Verified: correct on real precedence,
      refuses every disallowed construct tried (`__import__`, names, calls,
      attributes, `bool`-as-`int`).
- [x] **`generate_with_calculator` (`inference.py`)**: splices the real
      computed value in place of the model's own digit prediction whenever
      a literal expression completes right before `=`, reusing
      `predict_next`/`SessionTrie`/no-repeat-ngram unchanged. Real bug found
      and fixed while testing: the trigger could be skipped when a prompt
      already ends in "expr =" and BPE fuses `=` with the answer's first
      digit into one step -- fixed with an additional pre-loop check.
- [x] New synthetic, real-by-construction corpus
      (`generate_arithmetic_corpus.py`, cross-validated against Python's own
      `eval()`) + one toy checkpoint (`runs/arithmetic_base_s_rwkv`,
      223,936 params, best_val=0.6130, peak RSS 1,150MB) -- learned the
      step-trace format perfectly, got the arithmetic wrong on every
      multi-digit step (exactly the problem this phase targets).
- [x] **`bench_arithmetic.py`: plain (mimicry) 45% -> calculator-grounded
      100%** on 20 held-out single-expression tasks. Honest boundary found
      (not hidden): guarantees each *individual* detected expression is
      correct, does not by itself make the model carry a prior real result
      forward as the next step's own operand in a free-running chain --
      a distinct, harder capability, out of scope this round.
- [x] **`UniversalPredictor` step-sequencer test (`sequence_predictor.py`
      ported, `eval_step_sequencer.py`): inconclusive-to-negative.**
      Majority-class baseline 43.6%, `UniversalPredictor` 40.4% (*below*
      baseline), tiny Ducky 44.8% (+1.2pp over baseline). Neither predictor
      shows real sequential learning beyond trivial guessing on this task;
      doesn't make a case for `UniversalPredictor` over Ducky's own
      generation here -- a longer/deeper chain task is untested and might
      differ.

## Phase W — Three weaknesses, tested small-scale-first (new hard rule)
**New standing rule (saved to memory, `no_scaleup_without_proof`): never
commit to expensive/large training to test whether the architecture can
work -- always prove cheaply, at small scale with representative data,
that architecture (not lack of scale) is the ceiling first.** Skipped
weakness #1 (scale) by explicit instruction; did #2/#3/#4.
- [x] **#4 (self-improvement) re-evaluated, not re-attempted** -- the
      seemingly-new "verified-correct fuel" angle turned out to be
      already-present in the original (already 100%-correct-by-
      construction) arithmetic corpus, so it doesn't add information;
      folded its one real new angle into #2 below instead of a third
      blind self-training retry.
- [x] **Part 1, real execution-based grounding**: moved `bench_ducky.py`'s
      validated sandboxed-exec primitive into `grounding.run_sandboxed` +
      new `executes_without_error` signal, wired into
      `generate_with_grounding`. Verified on 4 known cases; **regression-
      confirmed the refactor changed nothing -- 10/10 canned-correct,
      0/10 on the real checkpoint, byte-for-byte matching history.**
- [x] **Part 2, operand-chain coherence**: new chained-operand corpus +
      toy checkpoint (223,936 params, best_val=0.5337). Found and fixed a
      real bug in the analysis itself (no stopping criterion -> spurious
      later-trace steps were being compared against the wrong answer).
      **Corrected result: original checkpoint already had 90.5%
      continuity / 89.5% final-answer self-consistency without any
      chain-aware training; the chained corpus only marginally improved
      continuity (93.75%, ~noise) and made self-consistency measurably
      worse (65.0%)** -- doesn't confirm the hypothesis, a real negative-
      leaning result, not chased further without a new hypothesis.
- [x] **Part 3, RWKV retention via synthetic associative recall: a sharp
      capacity cliff, precisely located.** All 4 conditions (pure RWKV/
      hybrid x short/long gap) landed at chance. Distrusted the
      suspiciously-uniform result and ran 3 real controls before trusting
      it (flat loss trajectory at 2x steps, width check 32->64, and the
      decisive one: **L=0 control reaches 100% accuracy, proving the task/
      setup is correctly implemented**). Cliff located precisely: **100%
      accuracy at L=0,1,2; collapse to chance at L=3,5,10** -- a hard wall
      at 2-3 tokens, not gradual, unaffected by architecture or width.
      Sharpens Phase M's "no training pressure" story into something more
      decisive: data engineered to *require* retention still fails almost
      immediately, real evidence toy-scale capacity itself (not just data)
      is the bottleneck -- the kind of finding that would justify testing
      at real scale next, earned cheaply first per the new rule, not
      assumed.

## Phase X — Training-recipe gap, not architecture: nanoGPT/uchi recipe vs. this project's untuned AdamW defaults
User's prompt: check nanoGPT and uchi for anything usable, on the premise
that a small model should still respond well -- i.e., is the whole
session's "0/10 bench_ducky, data/params is the ceiling" story confounded
by an unexamined training recipe, not just data/architecture. Real,
previously-uncontrolled variable found: every checkpoint logged this
session trained with raw `torch.optim.AdamW(model.parameters(), lr=lr)`
(betas=(0.9, 0.999), weight_decay=0.01 applied uniformly, including to
embeddings/LayerNorm/biases) and, except where explicitly opted in, a flat
undecayed LR -- never nanoGPT's or uchi's own (`flux/train_v2.py`) tuned
recipe (betas=(0.9, 0.95), weight_decay=0.1 on >=2D params only, GPT-2
scaled residual-projection init).
- [x] Added `--nanogpt-recipe` (`train.py`: param-group AdamW, betas=(0.9,
      0.95); `model.py`: `GPTConfig.scaled_residual_init`, GPT-2-paper
      std=0.02/sqrt(2*n_layer) init on attn_out/mlp-fc2 -- the residual-
      branch output projections, ported from nanoGPT, absent from both
      this file and uchi's own `flux/model.py` until now). Opt-in, zero
      change to any existing run's reproducibility.
- [x] Matched A/B on `rj/base/m` (vocab=1024, seed=57, dense attention,
      identical everything else). **First test (700 steps, cosine
      warmup=70) looked like a false negative** (3.898 vs baseline 3.746)
      -- diagnosed as a horizon-mismatch artifact: cosine decayed to its
      floor before the model (still improving at step 700 under flat LR)
      had converged, not a real recipe failure. Isolating just betas/
      weight-decay/init at flat LR already won narrowly even at 700 steps
      (3.7258 vs 3.7456).
- [x] **Re-tested at a matched, longer horizon (2000 steps, warmup=200)
      where each recipe's own best step could actually surface -- clean,
      decisive win.** Both peaked at step 1000: baseline 3.7694, full
      nanogpt-recipe+cosine **3.6572** (-0.112 nats). More importantly,
      **overfitting past the peak is dramatically gentler with the fixed
      recipe**: baseline blows up 3.7694->4.4537 (+0.68) by step 2000 as
      train loss collapses to 1.31; nanogpt-recipe only drifts
      3.6572->3.8371 (+0.18) as train loss reaches 2.30 -- weight decay
      excluding LayerNorm/bias/embedding params is visibly doing its job,
      not just changing optimization speed.
- [x] **Conclusion: real, reproducible win, orthogonal to every
      architecture/data finding in this file.** Doesn't retroactively
      invalidate prior comparisons (all were apples-to-apples under the
      same untuned recipe), but every past and future run was/is leaving
      real quality on the table until `--nanogpt-recipe --lr-schedule
      cosine` (with a step budget matched to the model's own convergence
      point, not a short fixed one) becomes the default recipe for
      comparisons going forward.
- [x] **Cheap-proxy re-test directly on the real target checkpoint's
      architecture/corpus** (`code`/`chinchilla_min`, real ~43M-token
      corpus, 1200/11000 = ~11% of the original run's horizon, same
      horizon-mismatch lesson controlled for by isolating cosine out).
      Isolated recipe (betas/weight-decay-scope/scaled-init, flat LR):
      **5.9205 vs. baseline 6.1973** -- a bigger proportional win (0.277
      nats, ~4.5% of the baseline) than the rj toy test showed, same
      direction, on the checkpoint that actually scored 0/10 on
      `bench_ducky`. Confirms the effect isn't rj-specific or an artifact
      of the smaller vocab=1024 toy setup.
- [x] **Live plateau-based LR adapter added** (`train.py`:
      `--lr-schedule plateau`, `--plateau-patience`/`--plateau-factor`/
      `--plateau-cooldown`, ReduceLROnPlateau-style, reuses the existing
      `best_val`/patience-counter tracking) -- directly answers the
      horizon-guessing problem that produced both false negatives above:
      no `max_steps` assumption anywhere in this path. Matched test
      (`rj`/m, 2000-step budget, seed=57): **best_val=3.6551 @ step
      1000**, fractionally beating the hand-tuned cosine run's 3.6572 at
      the same budget -- reaching the same quality as a correctly-guessed
      cosine schedule, without needing to guess. Auto-decayed LR exactly
      once (step 1500, 3e-4->1.5e-4), right where the flat/cosine runs'
      own overfitting inflection had already been observed.
- [x] `--weight-decay`/`--beta2` exposed as CLI args (previously
      hardcoded 0.1/0.95 inside `--nanogpt-recipe`) and `train.py`'s
      parser factored into `build_parser()` so a sweep script can build
      the same `argparse.Namespace` in-process instead of shelling out or
      duplicating the CLI surface.
- [x] `run_hpo_sweep.py` -- outer-loop random search (Bergstra & Bengio
      2012, arXiv-free JMLR 13, random > grid when few dimensions matter;
      no new dependency over Optuna, matching this project's own
      established no-unnecessary-dependency preference) over
      {lr, beta2, weight_decay, plateau_patience, plateau_factor} at toy
      scale (`rj`/m, ~800 steps/trial), `--lr-schedule plateau` fixed for
      every trial specifically because it needs no horizon guess -- a
      short toy trial and the eventual long real run don't need separate
      schedule tuning. 20 trials + the hand-copied default as a fixed
      reference point, single seed/single domain (`rj`) -- same
      single-point caveat as every other result in this file until
      repeated. **Results: best found (trial_18: lr=4.59e-4, beta2=0.955,
      weight_decay=0.144, plateau_patience=3, factor=0.57) val_loss=3.6797
      vs. the hand-copied default_recipe's 3.7258 -- a real but modest
      -0.046 nats.** Top 5 trials all cluster around lr~4.5-6e-4 (higher
      than nanoGPT's typical 3e-4), beta2 close to the hand-picked 0.95
      (not a strong lever here); weight_decay/plateau_patience show no
      clean trend in the top 5, need more trials to separate from noise.
      Not used for the default promotion below -- single-domain (`rj`
      only), and porting it untested to `code` would repeat the exact
      copy-without-checking mistake this phase exists to fix.
- [x] **User's explicit choice: promote the best validated EXPERIMENTAL
      SETUP (architecture + recipe) to Ducky's default now, not a
      production-scale retrain** ("I dont care about a real production
      run. Those are irrelevant until I have a promising architecture").
      Combined RWKV-hybrid (established best backbone) + plain
      `--nanogpt-recipe --lr-schedule plateau` defaults (not trial_18's
      numbers, for the reason above -- the recipe direction is validated
      on both domains, the fine-tuned hyperparameters are not) at `m`
      size, matched seed=57. **rj: best_val=3.5944, cleanly early-stopped
      at step 600/1200** -- beats every prior rj reference point,
      including hybrid-alone (4.351, no recipe fix) and today's
      dense+recipe-only result (3.66): the two levers stack. **code:
      best_val=5.4292 at the 2000-step budget cutoff, honestly NOT
      converged** (still improving when the run stopped, unlike rj) --
      best proven-recipe point so far, not a finished checkpoint.
      `ducky.py`'s `DEFAULT_RUNS` updated: `("rj","hybrid")` ->
      `rj_base_m_rwkv_lrplateau_nanogpt_seed57`, `("code","hybrid")` ->
      `code_base_m_rwkv_rank32_tokbalanced_lrplateau_nanogpt_seed57`.
      Both verified loading and answering through the real `Ducky()` SDK
      call path, not just checked as raw checkpoints. The older, bigger,
      fully-converged xl-scale checkpoints (code: val 3.4822 on the real
      corpus; text: val 5.1040) are better in absolute terms but were
      never retrained with the fixed recipe -- left on disk, not deleted,
      just no longer the default, since retraining them is exactly the
      production commitment this phase is deferring.
- [x] **Real mistake made and disclosed, not hidden: overwrote a
      pre-existing checkpoint.** `runs/rj_base_m_seed57` already existed
      (2026-07-18, part of an earlier repeat-seed validation round) --
      the first diagnostic command this phase ran reused `--seed 57`
      without checking for a name collision first, silently overwriting
      its `model_best.pt`/`model_final.pt`. Same overwrite-collision bug
      class this project has hit and fixed multiple times before (MoE/
      rwkv/tied-layer naming), self-inflicted this time. The original
      numbers are safe (already written into this file's/ducky.md's
      prose at the time), but that specific checkpoint file is gone --
      `runs/` has never been under git, so there is no history to restore
      from. `runs/rj_base_xs` (unrelated, 2026-07-15) had new files added
      alongside its original `model.pt` by a later sanity-check run;
      restored to its original single-file state, nothing destroyed
      there. Lesson: check for an existing run directory before reusing a
      seed/name combo for a throwaway test.
- [x] Cleanup: removed the disposable diagnostic run directories from
      this phase's A/B tests (`rj_base_m_lrcosine_seed57`,
      `rj_base_m_nanogpt_seed57`, `rj_base_m_lrcosine_nanogpt_seed57`,
      `rj_base_m_lrplateau_nanogpt_seed57`, three
      `code_base_chinchilla_min_..._seed57` variants) and a stray log
      file (`src/hpo_sweep_run.log`) -- all fully superseded by the
      numbers already recorded in this file's prose, none were reference
      points anything else depends on. `hpo_sweep_rj_m/results.json` (the
      sweep's real record) kept.
- [x] `code`'s new default extended from 2000 to 5000 steps (same config,
      same run directory): **best_val=4.5421 at step 4250**, up
      substantially from the mid-descent 5.4292 snapshot. Ran the full
      5000-step budget without plateau ever triggering a decay -- may
      still have some room left, but this is now a real, substantially-
      converged number, not a cutoff artifact. `ducky.py`'s comment
      updated to match.
- [ ] Not yet done: repeat-seed confirmation of the new defaults (single
      seed each, same caveat every other single-point result in this file
      carries), and checking whether `bench_ducky`'s 0/10 ceiling moves at
      all under the fixed recipe -- still open, deferred along with the
      production-scale question in general per the user's explicit choice
      this round.

## Phase Y — UniversalPredictor as a Ducky decision-maker: principle tested, real negative
User's recurring idea: embed `sequence_predictor.UniversalPredictor`
(ported from `uchi/uchi/predictor.py`) into Ducky so its own I/O drives a
decision, not just another value predictor bolted onto tokens. Investigated
uchi's own usage first, per the user's explicit "principle before
implementation": `tool_calling.py` has no UniversalPredictor involvement
(pure grammar parsing/dispatch); `numeric_plausibility.py` documents a
real prior empirical test -- UniversalPredictor was tried and rejected for
scattered/IID numeric-fact plausibility ("feeding it an IID pool of
disconnected numbers gives it nothing learnable... 55 and 50,000 scored
within 0.05 of each other"), with the explicit conclusion that it remains
the right tool for "genuinely sequential claims." uchi also built it for
an `/anomaly` skill -- a CTW-style sequential surprise detector over
*ordered* streams (sensor readings, prices over time), not a value
predictor. This reframed the question: Ducky's own Phase V step-sequencer
test (arithmetic operation order, ~50 sparse traces) never really tested
UniversalPredictor in its validated niche -- it needs a genuinely
sequential, decently-dense discrete stream, and Ducky already produces
several as a byproduct of running (not tokens themselves, which are
TinyGPT's job).
- [x] Identified and ranked three real candidate decision streams already
      inside Ducky's own execution: (1) `inference.py`'s per-token fast/
      slow/abstain decision -- densest, one symbol per generated token;
      (2) `repair_loop.py`'s per-attempt PASS/FAIL/ABSTAIN outcome across
      retries -- sparser, <=4 symbols/call; (3) `session_history.py`'s
      per-turn answered/abstained pattern across a session -- sparsest.
      Proposed role, matching `numeric_plausibility.py`'s own established
      pattern for a secondary signal ("additional veto layer, can only
      reject what the primary check already accepted") and `grounding.py`'s
      n-gram-rescue pattern already in `predict_next`: a second, independent
      vote toward abstention/repair, never something that overrides a
      clean answer alone.
- [x] **User chose to test all three, not just the recommended Path 1.**
      Built `eval_predictor_paths.py`, mirroring `eval_step_sequencer.py`'s
      exact methodology (majority-class baseline, held-out split,
      `UniversalPredictor.train()`/`predict()`/`feedback()` cycle) applied
      to real decision streams pulled from the new default checkpoints
      (walking real corpus text through `predict_next` for Path 1, running
      `generate_with_repair` over `bench_ducky`'s 10 real tasks for Path 2,
      simulating 12 multi-turn sessions with a deliberate in-domain/out-of-
      domain prompt mix for Path 3).
- [x] **Result: clean negative on all three, including the strongest
      candidate.** Path 1 (Confidence Watchdog, n=119 held-out decisions):
      baseline 55.5% vs. UniversalPredictor 48.7% -- **worse than the
      trivial majority-class guess**, not just a non-win. This is the
      one adequately-powered, clean test (most data, closest match to
      uchi's own validated anomaly-skill use case) and it still lost.
      Path 2 (Repair Advisor, n=4 held-out symbols): both 0% -- too
      sparse to mean anything either way, genuinely inconclusive, not a
      negative result. Path 3 (Session Watcher, n=12): exact tie at
      83.3% -- learned nothing beyond the baseline, though partly
      confounded by the test's own fixed in-domain/out-of-domain prompt
      cycling, which the majority baseline can trivially exploit -- not
      as clean a result as Path 1.
- [x] **Diagnosis, not just the number.** uchi's `/anomaly` skill works
      because the underlying VALUE has real physical continuity (a
      sensor reading is near its own recent value). Ducky's fast/slow/
      abstain label has no such property -- it's a downstream readout of
      *which specific token* is being predicted (driven by whether the
      model knows that word, not by momentum in the label sequence
      itself), and UniversalPredictor never sees the token content, only
      the label stream. Same failure shape as `numeric_plausibility.py`'s
      own documented case: no real temporal dependency in the sequence
      fed to it, nothing learnable. Not a bug in UniversalPredictor or in
      Ducky -- a genuine, now twice-confirmed (Phase V's step-sequencer,
      this phase's decision-stream test) mismatch between what this
      specific predictor needs and what Ducky's own discrete signals
      actually offer at this scale.
- [x] **Recommendation, accepted: do not pursue embedding
      UniversalPredictor into Ducky's decision-making further** without a
      genuinely different candidate signal -- the three most natural
      streams, tested honestly and cheaply (per `core_principle.md`'s own
      small-scale-first rule), don't support it. A real, kept negative
      result, same discipline as Phase M's BPTT findings and Phase R's
      width-gating result: informative, not a reason to scale up and
      retry.

## Phase Z — GPU freed up: real 4-point (xs/s/m/l) scaling verdict, replaces Phase Q's 2-point one
- [x] The RTX 5070 constraint stated at the top of this file (pinned at
      11.2/12.2GB since 2026-07-15) is gone -- GPU is free (826MiB/12.2GB,
      5% util). Directly unblocks Phase Q's own flagged gap: only xs/s (2
      close points) had ever been tested. Wired real GPU device placement
      into `run_scaling_sweep.py` (never existed before, despite Phase A's
      original "device auto-detect" intent -- everything ran on CPU
      implicitly). Verified GPU is 30-40x faster steady-state (once past a
      one-time, shape-dependent `torch.compile` cost) before trusting it.
- [x] Extended the sweep to `xs/s/m/l` (24 configs). First full run looked
      decisive but was contaminated by a stopping-rule bug: `PATIENCE=2`
      @ 50-step checks + 5 val batches (tuned against xs/s only) let one
      arch halt on a noisy early plateau (~step 150) while an equally-good
      arch escaped the same dip and trained on to 800-1650 steps --
      producing huge, non-monotonic fake "wins" on `code_core`/`terminal`.
      Diagnosed via the correlated stuck-early-vs-trains-on-forever pattern
      recurring identically on all 3 domains, not assumed.
- [x] Fixed (`rerun_ml.py`): `PATIENCE`->6, val batches 5->10, re-trained
      only the contaminated m/l points (12 runs), reused the clean,
      unaffected xs/s points as-is. Matched-arch pairs now stop at the same
      step far more often -- confirms the fix.
- [x] **Real, trustworthy 4-point verdict, promote selective decay = False
      stands (supersedes Phase Q's 2-point result)**: rj near-exact tie
      (both near-zero fitted alpha, consistent with rj's established
      data-ceiling status), code_core rwkv wins by a small real margin
      (4.322 vs 4.345 extrapolated), terminal rwkv wins decisively
      (2.614 vs 2.771, and the fitted alpha itself is meaningfully
      steeper: 0.109 vs 0.100) -- RWKV-hybrid's margin grows exactly where
      a domain has real scaling headroom, shrinks to a tie where it
      doesn't. See `ducky.md`'s Phase Z for full numbers and the honest
      caveat (some `l`-size runs hit the step ceiling without fully
      converging).
- [x] **This is the concrete answer to "architecture that holds up on
      small text and large text, regardless of size": RWKV-hybrid**,
      validated across 4 real size points x 3 domains, not a single toy
      point. Per `no-scaleup-without-proof`, this is exactly the cheap,
      real evidence that would justify a genuine production-scale
      commitment next -- that commitment itself is a separate, deliberate
      decision, not taken here.

## Phase AA — Cashing in the proof: real production checkpoints, code and text, on GPU
- [x] Wired real GPU device placement into `train.py` (never existed
      before -- same gap Phase Z found in `run_scaling_sweep.py`).
      Re-trained code's Chinchilla-matched `chinchilla_min` config with
      the recipe Phase X validated but never fully applied
      (`--nanogpt-recipe --lr-schedule plateau`). Found and fixed a real
      bug along the way: `plateau_stall` and the early-stop
      `patience_counter` shared a clock and never reset relative to each
      other, so early stopping fired ~1 checkpoint after every LR decay,
      before the new LR could help -- exactly why the first attempt
      (3.7559) was worse than the old checkpoint (3.6779). Fixed (reset
      patience on decay) and re-ran: **best_val 3.6423**, now a real win.
- [x] Diagnosed why no valid text checkpoint existed at all (`text_base_
      xxl_rwkv_rank96`'s train.log was 0 bytes): `_load_text_domain`
      concatenates the now ~1.3GB expanded Gutenberg corpus and tokenizes
      it in one uncapped call -- the same failure mode that spiked to
      ~19.5GB RSS on a 167MB corpus, extrapolated to something well past
      this machine's 39GB RAM. Fixed with `safe_tokenize_text.py`
      (chunked, per-chunk tensors concatenated at the end, not one giant
      Python list) -- **305,451,284 real tokens, peak RSS only 5.6GB.**
      Computed the real Chinchilla-optimal size (15,272,564 params) and
      registered it as `chinchilla_text` (320d/10L/rank80, 15,021,120
      params, within 2%) in `train.py`'s `SIZES`.
- [x] Discovered mid-launch that the GPU wasn't actually free: a separate,
      legitimate, currently-running `uchi.flux.react_warmup_train` job
      (2+ hours in) was using 8.1GB/12.2GB VRAM -- the earlier "GPU is
      free" reading had caught a lull between phases of that same job.
      Added `--grad-accum-steps` to `train.py` (default 1, zero behavior
      change) so `--batch-size 8 --grad-accum-steps 4` reproduces the
      same effective batch of 32 while fitting alongside the other job,
      per the user's explicit choice. **Result: best_val 5.0030** at step
      7,500 of 75,000 (~38 min total, converged well before the ceiling) --
      beats the old, differently-sized `text_base_xl_rwkv_rank64` (5.1040),
      the first complete Chinchilla-matched text checkpoint this project
      has produced. Honest caveat: `eval_step` reuses the reduced
      micro-batch size for validation too, so this number is measured
      somewhat noisier (5x8 samples) than the code run's (5x32) --
      flagged, not fixed this round.
- [x] Wired both into `ducky.py`'s `DEFAULT_RUNS` and verified end-to-end,
      not just by loss number. `Ducky(domain="text")` surfaced a second
      real, serious bug on first real use: `build_ngram_index`/
      `build_graph` were only ever safe at code's ~43M-token scale,
      never re-checked after text's corpus grew to 305M -- a live call
      spiked past 25GB RSS and had to be killed before it OOM'd the
      machine (and risked the concurrent uchi GPU job with it). Fixed by
      capping the token corpus fed to both structures to the same
      already-proven-safe order of magnitude (50M tokens). Re-verified
      safe and working (`Ducky(domain="text").ask("ROMEO:")` -> `'The'`,
      no crash).
- [x] **Re-ran `bench_ducky` against the new code default: still 0/10**,
      but genuinely informative -- completions are now built from real
      (if ultimately wrong) Python idioms instead of gibberish or
      1-2-token abstention, the same "coherent but not capable" ceiling
      this project has now confirmed at four separate scale/recipe
      points. The architecture and recipe work is real and measurably
      better in every way that isn't this specific capability -- it just
      doesn't touch this ceiling. See `ducky.md`'s Phase AA for full
      detail on every bug found and fixed along the way.

## Phase V — "One fluent model": Ducky as the shared engine for Uchi + Noosphere
**This is now the repo's confirmed main objective, not a proposal under
review** (see `core_principle.md`'s revised North Star): Ducky becomes the
single unified model replacing Uchi's FLUX proposer *and* Noosphere v2's EEG
stream encoder -- one set of weights, not two separate models that happen to
share a repo. User's explicit instruction: pursue this as the goal to
strive for; no code implementation yet, plan only.

A Destructor pass was run against the *sequencing* before accepting this (see
conversation record): the original framing bundled a same-modality swap
(Uchi, text->text) with a cross-modality, safety-critical swap (Noosphere,
EEG->6DOF prosthetic control) into one joint-training bet, with zero cheap
evidence a ~17-20M-param shared core transfers across modalities at all, and
`bench_ducky.py` has been 0/10 on every measurement this entire project. The
destination survives that pass unchanged (user confirmed it explicitly); the
sequencing below is how it gets built without betting real GPU time or, worse,
safety-critical code, on an unproven assumption. Two tracks, run
independently, converging on one model once each clears its own gate; nothing
touches Noosphere's actual safety-gated production code (ERN halt, watchdog,
ZOH-stable stream encoder) until Track 2 produces a real number.

- [ ] **Track 1 (Uchi, text-only, no new science).** Ducky replaces FLUX only
      once `bench_ducky.py` moves off 0/10 -- that's the real, already-known
      blocker (data/capability), not architecture or training recipe. No
      further action until that number moves; revisit data mix / model size
      per the still-open Phase AJ items (rj/gutenberg split completion,
      Chinchilla-optimal resize) before assuming more scale alone fixes it.
- [ ] **Track 2 (Noosphere, cross-modality, the real open question).** One
      toy, CPU/small-GPU experiment, off the safety-critical path entirely:
      Noosphere's own synthetic EEG generator (`v2_digital_self_replication/
      data/synthetic_eeg.py`, zero hardware/subject risk) + Ducky's existing
      small text/code pools. Train (a) a tiny shared RWKV core on both
      streams (CE loss on text, MSE on the continuous 6-DOF target) and (b)
      two size-matched single-modality baselines. Compare loss on each side
      independently -- shared must not lose to either baseline to call the
      hypothesis alive.
- [ ] Track 2's result decides the *next* step, not whether the objective
      stands (the objective is confirmed). If shared wins/ties both sides at
      toy scale -- design the real modality-router + multi-head architecture
      next, same small-scale-first ramp (`xs`/`s`/`m` before `xl`) as every
      other architecture decision in this repo. If shared loses either side
      at this size -- that is real evidence about *this* toy config, not a
      verdict on the goal: next moves are a bigger shared core (real
      generalist models that pull this off run 100M-1B+ params, per
      `core_principle.md`), a different fusion point (e.g. shared trunk with
      per-modality adapter layers instead of one undifferentiated core), or
      staged distillation (train each side well, then merge) -- tried in
      that order, each still proven cheaply before the next is attempted.
      Noosphere's production encoder stays in place and unmodified through
      all of this until whichever approach actually clears its gate.
- [ ] Explicit non-decision, unchanged: no Noosphere production file
      (`stream_encoder.py`, `safety_gate.py`, `kalman_filter.py`, or anything
      wired into `DigitalTwin`/`run_twin.py`) has been touched. Track 2 is a
      standalone script, not an integration.

See [`core_principle.md`](core_principle.md) for why this order and not the
obvious one. See [`ducky.md`](ducky.md) for Ducky's own architecture record
in full detail -- this file tracks the compressed plan; ducky.md tracks the
complete evidence trail.
