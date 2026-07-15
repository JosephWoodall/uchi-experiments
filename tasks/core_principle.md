# This Repo's North Star

**Core idea:** Prove, at toy scale and near-zero GPU cost, whether a sparse
(MoE) unified-tokenizer next-token predictor buys more effective parameters
per training FLOP than uchi's current dense SSM+BitNet architecture — before
spending real GPU budget finding out the hard way.

**Why this feels correct:** The obvious move is to just scale FLUX up. But
dense compute scales linearly with parameter count — on a single consumer
GPU, that linearly caps how many parameters you can ever afford to train.
Mixture-of-Experts decouples total parameters from active-compute-per-token:
the same FLOP budget buys a larger model, *if* there are enough tokens per
expert for routing to specialize. That "if" is exactly what a toy-scale
experiment (one book, one small code corpus) can cheaply falsify before it
costs a real GPU-week to find out.

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

**Non-negotiable scope discipline:** MoE, multimodal fusion, and factuality
training are backlog items, gated behind one thing — a working dense,
CPU-only, single-modality scaling-law harness. Any experiment here that
can't trace back to "does this move the params-per-FLOP or
verified-accuracy needle, measured on toy data" is scope creep. Drift from
this triggers a re-plan, not a bigger experiment.
