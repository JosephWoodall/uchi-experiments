# Falcon: Next-Generation Scalable Generative Model

This is the roadmap for **Falcon**, a scalable, generative algorithm designed to mirror the architectural foundations of state-of-the-art models like OpenAI's GPT-4 and Anthropic's Claude 3. 

While Ducky focuses on linear recurrence (RWKV) and hallucination-resistance, Falcon focuses on raw scalability and capability. It is designed to start small (training on `romeo_and_juliet.txt`) and scale infinitely without architectural bottlenecks.

## Core Architectural Principles
Falcon will be built on the **Modern Decoder-Only Transformer**. It abandons older 2017 Transformer mechanics in favor of the current industry standards:
- **Rotary Position Embeddings (RoPE):** Better length extrapolation than absolute embeddings.
- **SwiGLU Activations:** Higher performance feed-forward layers compared to standard ReLU/GELU.
- **RMSNorm:** Faster, more stable normalization.
- **Grouped Query Attention (GQA):** Massive reduction in memory overhead for scaling to long context windows.

## Phase 1: The Foundation (Architecture & SDK)
- [ ] **Create `src/falcon_model.py`**
  - Implement `RMSNorm` (simple, fast layer normalization).
  - Implement `RoPE` (Rotary Positional Embeddings) to replace Ducky's `TensorRankEmbedding` absolute positions.
  - Implement `SwiGLU` for the feed-forward network block.
  - Implement `GQA` (Grouped Query Attention) inside the attention block.
  - Assemble these into the `FalconTransformer` class with a forward pass that accepts token `ids` and returns next-token `logits`.
- [ ] **Create `src/falcon.py` (The Python SDK)**
  - Implement the `GenerativeAgent` class.
  - Include an `__init__` that loads `Tokenizer` (from `tokenizer.py`) and the `FalconTransformer` weights.
  - Implement `.ask(prompt: str, max_new_tokens: int, temperature: float) -> str` using an autoregressive loop.
  - Add Top-P (Nucleus) sampling into `.ask()` to prevent the degenerate repetition loops Ducky experienced.

## Phase 2: Proof of Concept Training (Romeo and Juliet)
- [ ] **Create `src/train_falcon.py`**
  - Adapt Ducky's training loop for Falcon.
  - Point the training data specifically to `data/text/romeo_and_juliet.txt`.
  - Use a small parameter footprint initially (e.g., ~10M-15M parameters) so it can train on a CPU/local GPU in minutes.
- [ ] **Train & Validate**
  - Verify that the loss curve smoothly descends.
  - Use the SDK to generate samples.
  - **Success Criteria:** Falcon should correctly spell character names ("ROMEO:", "JULIET:"), follow dialogue structure, and speak in somewhat coherent Shakespearean English.

## Phase 3: Scaling Up (Future Work)
- [ ] **Implement Mixture of Experts (MoE)**
  - Add a gating mechanism and multiple SwiGLU expert networks inside the Transformer blocks. This allows scaling to massive parameter counts (100B+) while keeping the active compute per token low.
- [ ] **Expand the Training Data**
  - Swap `romeo_and_juliet.txt` for the massive 135M+ token combined code/text corpus used by Ducky, adjusting the `vocab_size` and layers appropriately.
- [ ] **Post-Training Alignment**
  - Implement a DPO (Direct Preference Optimization) script to train Falcon to act as an assistant rather than just a raw next-word predictor.
