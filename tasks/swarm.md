# Swarm Intelligence Architecture with Unified Knowledge Graph

## Core Concept

**Swarm intelligence for hallucination-resistant, scalable multimodal prediction**: A Mixture-of-Experts (MoE) architecture where specialized expert "pods" collectively arrive at next-token predictions by querying a continuously-growing unified knowledge graph that combines facts, token relationships, and high-confidence model predictions.

**Key innovations**: 
1. **Three-layer knowledge integration**: Neural weights (learned patterns) + Unified Knowledge Graph (verifiable facts + distributional statistics + confident predictions)
2. **Swarm consensus from day 1**: Multiple experts with different graph traversal strategies arrive at collective predictions
3. **Confidence-gated growth**: Knowledge graph grows only from high-confidence sources, ensuring quality at any scale
4. **Unlimited sequence length**: Linear-time architecture with constant memory per token — no artificial context limits

---

## Quick Validation Strategy (10-15 Minutes)

**Before committing to full training, validate the core mechanisms work**:

### Toy-Scale Experiment Setup
- **Model**: 5M-10M params (4 layers × 256 hidden × 6 experts)
- **Data**: 100k-1M tokens (tiny subset: 10-20 files of code + text)
- **Hardware**: Single RTX 5070 (12GB VRAM)
- **Training time**: 10-15 minutes (3-5 epochs)
- **Goal**: Test mechanisms, not final quality

### What You're Validating (5 Critical Tests)

**Test 1: MoE Routing Works** (pass/fail)
```python
# After 5 epochs, check expert utilization
expert_counts = count_expert_activations(val_set)
# PASS: All experts activated >10% of time
# FAIL: One expert dominates >50% (expert collapse)
```

**Test 2: Experts Specialize** (pass/fail)
```python
# Measure routing patterns by token type
code_tokens = get_tokens(val_set, modality='code')
text_tokens = get_tokens(val_set, modality='text')

code_routing = get_expert_distribution(code_tokens)
text_routing = get_expert_distribution(text_tokens)

js_divergence = compute_js(code_routing, text_routing)
# PASS: JS divergence > 0.2 (experts route differently)
# FAIL: JS divergence < 0.1 (no specialization)
```

**Test 3: Knowledge Graph Extraction Works** (pass/fail)
```python
# Extract graph from 100k tokens
graph = extract_knowledge_graph(train_data)

# Check quality
ast_facts = [e for e in graph.edges if e.provenance == 'ast_extraction']
co_occurrence = [e for e in graph.edges if e.relation_type == 'co_occurrence']

# PASS: 500-2000 AST facts extracted, 90%+ precision
# PASS: 1000-5000 co-occurrence edges, 75%+ precision
# FAIL: <100 edges total, or random/incorrect facts
```

**Test 4: Swarm Differs from Single Expert** (pass/fail)
```python
# Generate 100 tokens with single expert vs swarm
single_expert_output = generate(prompt, use_expert=0)
swarm_output = generate(prompt, use_swarm=True)

# Measure difference
token_difference = hamming_distance(single_expert_output, swarm_output)
# PASS: >20% of tokens differ (swarm has effect)
# FAIL: <5% differ (swarm is redundant)
```

**Test 5: Graph Affects Predictions** (pass/fail)
```python
# Generate with and without graph queries
no_graph_output = generate(prompt, use_graph=False)
with_graph_output = generate(prompt, use_graph=True)

# Measure difference
token_difference = hamming_distance(no_graph_output, with_graph_output)
# PASS: >10% of tokens differ (graph has effect)
# FAIL: <2% differ (graph integration broken)
```

### Success Criteria (Toy Scale)
- ✅ **All 5 tests pass**: Core mechanisms work, proceed to full scale
- ❌ **1-2 tests fail**: Debug those specific components, re-test
- ❌ **3+ tests fail**: Architecture has fundamental issues, rethink approach

### What This DOESN'T Tell You
- Whether final model is good at coding/conversation (need full scale for that)
- Whether it beats standard LLMs (need benchmarks)
- Whether swarm improves accuracy by 15%+ (need larger data)

### What This DOES Tell You
- ✅ MoE works and experts specialize (not collapsing)
- ✅ Knowledge graph extracts meaningful edges
- ✅ Swarm produces different outputs than single expert
- ✅ Graph integration affects model behavior
- ✅ Architecture is fundamentally sound → safe to scale up

**Timeline**: 10-15 min experiment → 30 min analysis → decision to proceed or pivot

---

## Architecture Goals

1. **Scales well**: MoE decouples total parameters from active compute-per-token
2. **Hallucination-resistant**: Swarm consensus + knowledge graph validation + uncertainty-aware abstention
3. **Efficient to train**: Sparse activation during training, BitNet quantization, consumer-friendly
4. **Text + Code focus**: Drop multimodal complexity, focus on what matters for coding + conversation
5. **Knowledge-updatable**: Graph updates continuously without model retraining
6. **Works at any scale**: Components designed to function from toy scale (100k tokens) to production (1B tokens)
7. **Unlimited context**: No hard sequence length limit — process arbitrarily long sequences with constant memory
8. **Single GPU friendly**: Designed for RTX 5070 (12GB VRAM) — no multi-GPU required

---

## Production Scale Requirements (After Toy Validation)

**Target**: Great coder + conversationalist

**Data requirements**:
- **Text**: 600M tokens (books, articles, documentation, conversations)
- **Code**: 400M tokens (GitHub repos, Stack Overflow, API docs)
- **Total**: 1B tokens (text + code only, drop audio/video)

**Why 1B tokens**:
- Modern coding assistants (Copilot, Claude) train on 100B+ tokens
- 1B is 1% of that scale — minimum for basic competence
- Below 100M: model memorizes, doesn't generalize
- 100M-1B: model generalizes within domain
- 1B+: competitive with specialized models

**Model scale** (Chinchilla optimal):
- **Parameters**: 50M total (not 100M-200M)
- **Why smaller**: Chinchilla showed 70M @ 1B tokens >> 200M @ 100M tokens
- **Active params per token**: 8M-10M (top-1 routing)
- **Memory footprint**: ~4GB (50M params × BitNet + optimizer state)

---

## Core Components

### 1. Base Architecture: RWKV + BitNet

**RWKV (Receptance Weighted Key Value)**:
- Linear O(n) time complexity for sequence modeling
- Constant O(1) memory per token — no sequence length limit
- RNN-like recurrent processing with parallel training
- Proven at scale: RWKV-7B, RWKV-14B models handle 100k+ token contexts
- Reference: Peng et al. 2023 (arXiv:2305.13048)

**Why RWKV for unlimited sequences**:
- **No KV cache**: Processes one token at a time with fixed-size hidden state
- **No attention**: Replaces quadratic self-attention with time-mixing (linear recurrence)
- **Constant memory**: Hidden state size is fixed regardless of sequence length
- **Battle-tested**: RWKV-World models process entire books (100k+ tokens) without OOM

**BitNet 1.58-bit quantization**:
- Ultra-low-bit weights and activations
- 8-16× denser parameter packing than FP16
- Reference: Ma et al. 2024 (arXiv:2402.17764)

**Why this base**: 
- RWKV provides true unlimited sequence length (vs Transformer hard limits)
- Proven MoE integration pattern (RWKV-MoE implementations exist)
- BitNet provides aggressive quantization for consumer GPUs
- Linear scaling allows processing full repos, long videos, entire conversations

**RWKV architecture overview**:
```python
class RWKVBlock(nn.Module):
    """RWKV block: time-mixing (sequence) + channel-mixing (features)"""
    def __init__(self, d_model):
        super().__init__()
        self.time_mixing = TimeMixing(d_model)      # Replaces attention
        self.channel_mixing = ChannelMixing(d_model) # Replaces FFN
    
    def forward(self, x, state):
        # Time-mixing: sequence dependencies (linear, not quadratic)
        x, state = self.time_mixing(x, state)
        
        # Channel-mixing: feature transformation
        x, state = self.channel_mixing(x, state)
        
        return x, state  # State is constant size
```

**Integration with MoE**:
```python
class RWKVMoE(nn.Module):
    """RWKV with MoE: time-mixing for sequence, MoE for features"""
    def __init__(self, num_layers, num_experts):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'time_mixing': TimeMixing(d_model),        # Keep RWKV sequence model
                'moe': MoELayer(num_experts, d_model)     # Replace channel-mixing with MoE
            })
            for _ in range(num_layers)
        ])
```

**De-risking strategy**:
- Week 1: Toy experiment (5M params, 100k tokens, 10-15 min) → validate mechanisms
- Week 2-3: Small scale (20M params, 10M tokens, 1-2 days) → validate scaling
- Week 4-8: Production scale (50M params, 1B tokens, 3-4 weeks) → full training

### 2. Mixture of Experts (MoE) - Core Component

**Training mode - Sparse activation**:
- Top-k gating: route each token to 1 expert (k=1)
- Experts specialize via routing gradients
- Load balancing to prevent expert collapse
- References: Shazeer et al. 2017 (arXiv:1701.06538), DeepSeekMoE 2024 (arXiv:2401.06066)

**Inference mode - Dense aggregation (Swarm)**:
- All experts process each token
- Experts use **different graph traversal strategies**
- Outputs aggregated via weighted voting
- Weights based on: (expert confidence) × (graph consistency) × (inter-expert agreement)
- This is where swarm intelligence emerges

**MoE + RWKV integration**:
- RWKV time-mixing layers for sequence dependencies (temporal patterns)
- MoE layers replace channel-mixing (feature transformation)
- Interleaved architecture: [TimeMixing → MoE → TimeMixing → MoE → ...]
- Shared recurrent state across layers, expert-specific feature transformations

**Expert specialization strategies** (what makes swarm work):
- **Local traversal expert**: Focuses on immediate next-token (n=1 co-occurrence)
- **Long-range expert**: Explores paths through graph (n-gram chains)
- **Frequency expert**: Weights edges by frequency (common patterns)
- **Novelty expert**: Explores low-frequency edges (creative generation)
- **Fact-grounded expert**: Only traverses edges validated by semantic facts
- **Syntax expert**: Specializes in code syntax patterns (AST facts)

**Forced diversity mechanisms**:
```python
class DiverseExpert(nn.Module):
    """Expert with forced specialization from initialization"""
    
    def __init__(self, expert_id, strategy, dropout_mask):
        super().__init__()
        self.expert_id = expert_id
        self.strategy = strategy  # Assigned graph traversal strategy
        self.dropout_mask = dropout_mask  # Forces different feature usage
        
        # Different random initialization per expert
        self.init_weights(seed=expert_id * 42)
    
    def forward(self, x, graph, context):
        # Apply expert-specific dropout (forces different features)
        x = x * self.dropout_mask
        
        # Neural prediction
        neural_logits = self.ffn(x)
        
        # Graph query with assigned strategy
        graph_suggestions = self.strategy.query(graph, context)
        
        # Combine
        return combine(neural_logits, graph_suggestions)
```

### 3. Unified Tokenization (Text + Code Only)

**Goal**: Single discrete token vocabulary for text and code

**Text & Code**:
- BPE/SentencePiece tokenization (standard)
- Vocabulary size: ~32k-50k tokens
- Shared vocabulary enables cross-domain learning (code comments ↔ natural language)

**Why drop multimodal**:
- Audio/video encoding adds 30-40% overhead
- Not needed for coding + conversation
- Simplifies architecture and speeds training
- Can add later if needed

**Modality delimiters**: Special tokens marking text vs code
- `<|text|>`, `<|code|>`, `<|/text|>`, `<|/code|>`
- Model learns to predict next token regardless of modality

### 4. Unified Knowledge Graph (Facts + Token Relationships + Model Predictions)

**Core idea**: A single graph structure that unifies three knowledge sources:
1. **Semantic facts** (extracted from data): "Paris is capital of France", "len(str) → int"
2. **Distributional patterns** (token co-occurrence): "In code, '.shape' follows 'df.' 87% of the time"
3. **High-confidence predictions** (from the model itself): Novel facts the model discovered with >95% expert agreement

**Why unified**: All three sources answer the same question: "What's likely/valid to predict next?" Keeping them in one graph enables:
- Single query mechanism (experts traverse one graph, not three)
- Cross-validation (distributional pattern + semantic fact + model confidence)
- Graceful growth (graph starts with high-precision facts, adds statistics as data accumulates)

---

#### 4.1 Graph Structure

**Nodes**: Tokens in the vocabulary (32k-50k nodes)

**Edges**: Relationships between tokens with rich metadata

**Edge schema**:
```python
Edge:
  - source_token: token ID
  - target_token: token ID
  - relation_type: [semantic_fact, co_occurrence, model_prediction]
  - weight: strength of relationship (0-1)
  - confidence: confidence score (0-1)
  - context: context in which this edge appears
  - frequency: number of times observed (for co-occurrence)
  - provenance: [ast_extraction, ie_model, training_corpus, user_correction, model_inference]
  - modality: [text, code]
  - created_at: timestamp
  - updated_at: timestamp
```

**Example edges**:

```python
# Semantic fact (code)
Edge(
  source="DataFrame", target="shape",
  relation_type="semantic_fact",
  weight=1.0, confidence=0.99,
  context="df.<attribute>",
  provenance="ast_extraction",
  modality="code"
)

# Co-occurrence pattern (text)
Edge(
  source="Paris", target="France",
  relation_type="co_occurrence",
  weight=0.85, confidence=0.92,
  context="<city> is capital of <country>",
  frequency=47,
  provenance="training_corpus",
  modality="text"
)

# High-confidence model prediction
Edge(
  source="import", target="pandas",
  relation_type="model_prediction",
  weight=0.93, confidence=0.96,
  context="import <library> as pd",
  provenance="model_inference",
  modality="code"
)
```

**Graph pruning for quality** (ruthless filtering):
```python
def should_add_to_graph(edge):
    """Only add edges the neural model can't learn well"""
    
    # Always add facts and corrections (high value)
    if edge.provenance in ['ast_extraction', 'ie_model', 'user_correction']:
        return True
    
    # Co-occurrence: only rare patterns with high confidence
    if edge.relation_type == 'co_occurrence':
        if edge.frequency > 100:  # Too common, model learned it
            return False
        if edge.frequency < 5:  # Too rare, might be noise
            return False
        if edge.confidence < 0.85:  # Not confident enough
            return False
        return True  # Rare + confident = valuable
    
    # Model predictions: very high bar
    if edge.relation_type == 'model_prediction':
        if edge.confidence < 0.95:  # Must be very confident
            return False
        # Cross-validate with semantic facts
        if conflicts_with_facts(edge):
            return False
        return True
    
    return True
```

---

#### 4.2 Knowledge Extraction & Graph Building

**During training** (parallel extraction pipeline):

**1. High-precision fact extraction** (added immediately):

**Code facts - Rule-based (AST parsing)**:
- Parse code → Abstract Syntax Tree
- Extract: function signatures, type hints, imports, call graphs
- Add to graph as `semantic_fact` edges with confidence=0.99
- Deterministic, high precision
- Example: `len(str) → int` becomes edge with weight=1.0

**Text facts - Information Extraction models**:
- Pretrained IE model: OpenIE, Stanford CoreNLP, or Rebel
- Extract subject-relation-object triples
- Filter by confidence threshold (> 0.9)
- Add to graph as `semantic_fact` edges
- Example: `(Paris, capital_of, France)` → edge with confidence=0.95

**2. Token co-occurrence statistics** (accumulated, added when frequency > threshold):

```python
class TokenCooccurrenceBuilder:
  def __init__(self, min_frequency=5, min_confidence=0.85):
    self.co_occurrence_counts = defaultdict(int)
    self.min_frequency = min_frequency
    self.min_confidence = min_confidence
    
  def update_from_batch(self, token_sequences):
    """Track co-occurrence from training batch"""
    for sequence in token_sequences:
      for i in range(len(sequence) - 1):
        current_token = sequence[i]
        next_token = sequence[i + 1]
        context = tuple(sequence[max(0, i - 5):i])
        
        # Increment counter
        edge_key = (current_token, next_token, context)
        self.co_occurrence_counts[edge_key] += 1
        
        # Add to graph when threshold met AND passes quality filter
        if self.co_occurrence_counts[edge_key] == self.min_frequency:
          frequency = self.co_occurrence_counts[edge_key]
          confidence = compute_confidence(frequency, context)
          
          # Quality filter: rare but confident patterns only
          if confidence > self.min_confidence and frequency < 100:
            graph.add_edge(
              current_token, next_token,
              relation_type="co_occurrence",
              weight=frequency / max_frequency,
              confidence=confidence,
              context=context,
              frequency=frequency,
              provenance="training_corpus"
            )
```

**Key insight**: Co-occurrence edges are **NOT** added on first occurrence. They're accumulated and only added when they reach statistical significance (5 < frequency < 100, confidence > 0.85).

**3. High-confidence model predictions** (added during inference):

```python
def add_model_prediction_to_graph(context, expert_predictions, graph):
  """Add novel facts discovered by model consensus"""
  
  # Check expert agreement
  agreement_score = compute_agreement(expert_predictions)
  
  if agreement_score > 0.95:  # Very high agreement
    predicted_token = most_common_prediction(expert_predictions)
    
    # Check if this is a novel pattern (not already in graph)
    if not graph.has_edge(context[-1], predicted_token):
      # Cross-validate with semantic facts
      if not conflicts_with_facts(context, predicted_token, graph):
        # Add to graph
        graph.add_edge(
          context[-1], predicted_token,
          relation_type="model_prediction",
          weight=agreement_score,
          confidence=agreement_score,
          context=tuple(context[-5:]),
          provenance="model_inference"
        )
```

**4. User corrections** (highest confidence, added immediately):

```python
def add_user_correction(incorrect_token, correct_token, context, graph):
  """User explicitly corrects a mistake"""
  
  # Remove or downweight incorrect edge if it exists
  if graph.has_edge(context[-1], incorrect_token):
    graph[context[-1]][incorrect_token]['weight'] *= 0.5
    graph[context[-1]][incorrect_token]['confidence'] *= 0.5
  
  # Add correct edge with maximum confidence
  graph.add_edge(
    context[-1], correct_token,
    relation_type="semantic_fact",
    weight=1.0,
    confidence=1.0,
    context=tuple(context),
    provenance="user_correction"
  )
```

---

#### 4.3 Graph Growth Over Time

**Toy scale** (100k tokens, 10-15 min training):
- **Semantic facts**: 500-2k edges (code AST from 10-20 files)
- **Co-occurrence**: 1k-5k edges (rare patterns only)
- **Model predictions**: 0 (not yet added)
- **Total**: 1.5k-7k edges
- **Purpose**: Validate extraction pipeline works

**Small scale** (10M tokens, 1-2 days training):
- **Semantic facts**: 5k-10k high-precision edges
- **Co-occurrence**: 10k-20k edges (5 < freq < 100, conf > 0.85)
- **Model predictions**: 0-500 edges (early swarm consensus)
- **Total**: 15k-30k edges
- **Coverage**: 15-25% of predictions have relevant graph context

**Production scale** (1B tokens, 3-4 weeks training):
- **Semantic facts**: 50k-100k edges
- **Co-occurrence**: 150k-200k edges (ruthlessly pruned)
- **Model predictions**: 20k-50k edges (high-agreement patterns)
- **User corrections**: 1k-5k edges (accumulated over time)
- **Total**: 220k-350k edges
- **Coverage**: 50-65%

**Key property**: Graph is **useful at every scale**, just with different coverage. At toy scale, 1.5k edges is enough to test if graph queries affect predictions. At production scale, 300k edges provide broad coverage.

---

#### 4.4 Querying the Knowledge Graph (Expert Traversal Strategies)

**At inference**, each expert queries the graph differently:

**1. Local Traversal Expert** (immediate next-token):
```python
def query_local(graph, context):
  """Get direct next-token edges from last token"""
  last_token = context[-1]
  
  # Get all outgoing edges
  candidates = []
  for next_token in graph.successors(last_token):
    edge = graph[last_token][next_token]
    score = edge['weight'] * edge['confidence']
    candidates.append((next_token, score))
  
  return sorted(candidates, key=lambda x: x[1], reverse=True)[:20]
```

**2. Long-Range Expert** (multi-hop paths):
```python
def query_long_range(graph, context):
  """Explore paths through graph (2-3 hops)"""
  last_tokens = context[-3:]
  
  # Find paths: token[-3] → token[-2] → token[-1] → ?
  paths = []
  for path in graph.paths_from(last_tokens, max_length=3):
    path_weight = product([edge['weight'] for edge in path])
    paths.append((path[-1].target, path_weight))
  
  return sorted(paths, key=lambda x: x[1], reverse=True)[:20]
```

**3. Frequency Expert** (common patterns):
```python
def query_frequency(graph, context):
  """Prefer high-frequency edges (within pruned range)"""
  last_token = context[-1]
  
  candidates = []
  for next_token in graph.successors(last_token):
    edge = graph[last_token][next_token]
    if edge['relation_type'] == 'co_occurrence':
      score = edge['frequency'] * edge['confidence']
    else:
      score = edge['weight'] * edge['confidence']
    candidates.append((next_token, score))
  
  return sorted(candidates, key=lambda x: x[1], reverse=True)[:20]
```

**4. Fact-Grounded Expert** (only semantic facts):
```python
def query_facts_only(graph, context):
  """Only traverse semantic_fact edges"""
  last_token = context[-1]
  
  candidates = []
  for next_token in graph.successors(last_token):
    edge = graph[last_token][next_token]
    if edge['relation_type'] == 'semantic_fact':
      score = edge['confidence']
      candidates.append((next_token, score))
  
  return sorted(candidates, key=lambda x: x[1], reverse=True)[:20]
```

**5. Syntax Expert** (code-specific, AST patterns):
```python
def query_syntax(graph, context, current_modality):
  """Specialize in code syntax via AST facts"""
  if current_modality != 'code':
    return []  # Only activate in code context
  
  last_token = context[-1]
  candidates = []
  for next_token in graph.successors(last_token):
    edge = graph[last_token][next_token]
    if edge['provenance'] == 'ast_extraction':
      score = edge['confidence']
      candidates.append((next_token, score))
  
  return sorted(candidates, key=lambda x: x[1], reverse=True)[:20]
```

**6. Novelty Expert** (explores rare patterns):
```python
def query_novelty(graph, context):
  """Prefer low-frequency but high-confidence edges"""
  last_token = context[-1]
  
  candidates = []
  for next_token in graph.successors(last_token):
    edge = graph[last_token][next_token]
    # Novelty = high confidence but low frequency
    frequency = edge.get('frequency', 1)
    novelty_score = edge['confidence'] / log(1 + frequency)
    candidates.append((next_token, novelty_score))
  
  return sorted(candidates, key=lambda x: x[1], reverse=True)[:20]
```

**Why this creates swarm intelligence**:
- Each expert gets different graph suggestions based on traversal strategy
- Experts combine neural predictions + graph suggestions differently
- Aggregation produces consensus that's more robust than any single strategy
- When experts disagree strongly, it signals uncertainty → abstain

**Graph query optimization** (target <10ms per expert):
```python
from functools import lru_cache
import faiss

class FastGraphIndex:
    def __init__(self, graph):
        self.graph = graph
        # Pre-compute: context hash → candidate edges
        self.context_index = self.build_context_index()
        
        # FAISS for vector similarity search
        self.faiss_index = faiss.IndexFlatL2(384)
        self.embed_edges(graph)
    
    @lru_cache(maxsize=10000)
    def query(self, context_tuple, strategy):
        """O(1) lookup with caching"""
        context_hash = hash(context_tuple[-5:])
        return self.context_index.get(context_hash, [])
```

---

#### 4.5 Graph Storage & Implementation

**Storage backend**:
- **NetworkX** (development, toy/small scale): In-memory Python graph library
- **Neo4j** (production): Distributed graph database with Cypher queries
- **Vector index overlay**: Embed edge contexts for fast similarity search

**Hybrid architecture**:
```
┌─────────────────────────────────────┐
│   Vector DB (ChromaDB / FAISS)      │
│   For fast context→edge retrieval   │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│   Graph DB (Neo4j / NetworkX)       │
│   For path queries, traversal       │
└─────────────────────────────────────┘
```

**Query flow**:
1. Embed current context (last 5 tokens)
2. Vector search: find relevant edge contexts (top-20)
3. Graph traversal: follow edges from last token, filter by retrieved contexts
4. Return scored candidates per expert strategy

**Performance targets**:
- Graph query latency: <10ms per expert per token (with caching + indexing)
- Graph size: 1.5k-7k edges (toy) → 15k-30k edges (small) → 220k-350k edges (production)
- Insert throughput: >1000 edges/second
- Memory: <10MB (toy), <50MB (small), <200MB (production) for NetworkX in-memory

---

## Training Strategy

### Sparse MoE Training with Graph Building

**Objective**: Train experts to specialize while building knowledge graph

**Training loop**:
```python
# Initialize graph builder
graph_builder = UnifiedKnowledgeGraphBuilder()

for batch in dataloader:
  # 1. Forward pass with sparse routing
  logits, router_probs, expert_loss = model(batch, mode='sparse')
  
  # 2. Standard next-token prediction loss
  prediction_loss = cross_entropy(logits, targets)
  
  # 3. MoE auxiliary losses
  load_balance_loss = compute_load_balance(router_probs)
  router_z_loss = compute_router_z_loss(router_probs)
  
  # 4. Total loss
  loss = prediction_loss + α * load_balance_loss + β * router_z_loss
  
  # 5. Backward pass
  loss.backward()
  optimizer.step()
  
  # 6. Build knowledge graph (parallel)
  graph_builder.extract_facts(batch)           # Immediate: high-precision facts
  graph_builder.update_co_occurrence(batch)    # Accumulated: add when freq > threshold
```

**Parallel graph extraction**:
```python
class UnifiedKnowledgeGraphBuilder:
  def __init__(self):
    self.graph = nx.DiGraph()
    self.co_occurrence_tracker = TokenCooccurrenceBuilder()
    self.fact_extractors = {
      'code': ASTFactExtractor(),
      'text': IEFactExtractor()
    }
  
  def extract_facts(self, batch):
    """Extract high-precision facts, add immediately"""
    modality = detect_modality(batch)
    facts = self.fact_extractors[modality].extract(batch)
    
    for fact in facts:
      if fact.confidence > 0.9:
        self.graph.add_edge(
          fact.source, fact.target,
          relation_type='semantic_fact',
          weight=1.0,
          confidence=fact.confidence,
          provenance=f'{modality}_extraction'
        )
  
  def update_co_occurrence(self, batch):
    """Track co-occurrence, add when threshold met"""
    self.co_occurrence_tracker.update_from_batch(batch)
    
    # Add edges that just crossed threshold
    new_edges = self.co_occurrence_tracker.get_new_edges()
    for edge in new_edges:
      self.graph.add_edge(edge)
```

**Hyperparameters**:
- Top-k routing: k=1 expert per token (training)
- Load balance loss weight α: 0.01 - 0.1
- Router z-loss weight β: 0.001
- Expert count: 6 experts (toy/small scale), 8-12 (production)
- Co-occurrence threshold: 5 < frequency < 100, confidence > 0.85
- Fact extraction threshold: confidence > 0.9

**Data mixing** (text + code):
- Interleave text and code within training sequences
- Example sequence: text → code → text → code
- Use modality delimiter tokens
- Target distribution: 60% text, 40% code

**Measuring expert diversity** (critical for swarm):
```python
def measure_expert_diversity(expert_predictions):
    """Track diversity to ensure swarm isn't converging"""
    
    # 1. Prediction disagreement
    top_tokens = [pred.argmax() for pred in expert_predictions]
    unique_predictions = len(set(top_tokens))
    disagreement_rate = unique_predictions / len(expert_predictions)
    
    # 2. Distribution divergence (JS divergence)
    avg_divergence = mean([
        js_divergence(pred_i, pred_j) 
        for pred_i, pred_j in combinations(expert_predictions, 2)
    ])
    
    # Alert if diversity is collapsing
    if disagreement_rate < 0.3 or avg_divergence < 0.2:
        logger.warning("Expert diversity collapsing! Re-initialize or add noise.")
    
    return disagreement_rate, avg_divergence
```

---

## Inference Strategy: Adaptive Swarm Consensus

**Adaptive routing** (3-4× average slowdown, not 8×):

```python
def adaptive_swarm_inference(context, model, graph, experts):
    """Switch between sparse and dense based on uncertainty signals"""
    
    # Step 1: Fast sparse pass (top-1 expert)
    sparse_prediction, sparse_confidence = sparse_forward(context, top_k=1)
    
    # Step 2: Quick uncertainty check
    graph_edges = graph.get_edges(context[-1])
    
    # High confidence + graph support → trust sparse prediction
    if sparse_confidence > 0.85 and len(graph_edges) > 5:
        return sparse_prediction  # Fast path (1-2× slower than baseline)
    
    # Low confidence OR no graph → invoke full swarm
    else:
        return full_swarm_consensus(context, model, graph, experts)  # Slow path (6-8× slower)
```

**Full swarm inference** (per token, when triggered):
```python
def generate_next_token_swarm(context, model, knowledge_graph, experts):
  """Swarm consensus with unified knowledge graph"""
  
  # 1. Each expert: neural prediction + graph traversal
  expert_predictions = []
  for expert in experts:
    # Neural forward pass
    neural_logits = expert.forward(context)
    
    # Query knowledge graph with expert's traversal strategy
    graph_suggestions = expert.query_strategy(knowledge_graph, context)
    
    # Combine: neural + graph
    combined_logits = combine(
      neural_logits, 
      graph_suggestions,
      α=0.6,  # Neural weight
      β=0.4   # Graph weight
    )
    
    expert_predictions.append({
      'logits': combined_logits,
      'neural_confidence': softmax(neural_logits).max(),
      'graph_confidence': max([s[1] for s in graph_suggestions]) if graph_suggestions else 0,
      'combined_confidence': softmax(combined_logits).max()
    })
  
  # 2. Swarm aggregation
  # Weight each expert by: neural confidence × graph confidence
  weighted_predictions = []
  for pred in expert_predictions:
    weight = pred['neural_confidence'] * (0.5 + 0.5 * pred['graph_confidence'])
    weighted_predictions.append((pred['logits'], weight))
  
  # Aggregate
  final_logits = weighted_sum(weighted_predictions)
  
  # 3. Expert agreement (uncertainty signal)
  expert_agreement = compute_agreement([p['logits'] for p in expert_predictions])
  
  # 4. Graph consistency check
  predicted_token = final_logits.argmax()
  graph_consistent = check_graph_consistency(
    context[-1], predicted_token, knowledge_graph
  )
  
  # 5. Abstention check
  if expert_agreement < 0.3 or graph_consistent < 0.5:
    return ABSTAIN_TOKEN
  
  # 6. Return prediction
  return predicted_token
```

**Key mechanisms**:

**1. Swarm emerges from diverse graph traversal**:
- Each expert uses different query strategy
- Same graph, different paths → different suggestions
- Aggregation finds consensus across strategies

**2. Graph acts as shared ground truth**:
- All experts query same graph
- High-confidence edges (facts, frequent patterns) guide all experts
- Low-confidence edges only influence some experts

**3. Abstention from disagreement**:
- If experts disagree strongly → uncertain context → abstain
- If graph has no relevant edges → out-of-distribution → abstain
- If predicted token violates graph facts → likely hallucination → abstain

**Abstention strategy with partial coverage**:
```python
def inference_with_partial_coverage(context, model, graph):
    """Graph coverage isn't 100%, and that's OK"""
    
    # Check if graph has relevant edges
    graph_edges = graph.get_edges(context[-1])
    
    if len(graph_edges) == 0:
        # No graph coverage → high uncertainty
        confidence_penalty = 0.5  # Reduce confidence
        abstain_threshold = 0.6  # Lower bar for abstention
    else:
        # Graph coverage → high confidence
        confidence_penalty = 1.0
        abstain_threshold = 0.3
    
    # Generate with adjusted thresholds
    prediction = swarm_predict(context, model, graph)
    final_confidence = prediction.confidence * confidence_penalty
    
    if final_confidence < abstain_threshold:
        return ABSTAIN_TOKEN
    return prediction
```

**Coverage insight**: Even 20-30% graph coverage is useful. With graph (20-30% of tokens): high precision. Without graph (70-80% of tokens): higher abstention, signals uncertainty. **This is a feature**: system knows what it doesn't know.

---

## Implementation Phases

### Phase 0: Toy Experiment (10-15 minutes)

**Goal**: Validate core mechanisms work before investing in full training

**Setup**:
- **Model**: 5M-10M params (4 layers, 256 hidden, 6 experts)
- **Data**: 100k-1M tokens (10-20 code files + text documents)
- **Hardware**: RTX 5070 (12GB VRAM)
- **Training**: 3-5 epochs, 10-15 minutes
- **Graph**: Extract 1.5k-7k edges (facts + co-occurrence)

**5 Critical Validation Tests** (see Quick Validation Strategy section above)

**Success criteria**:
- All 5 tests pass → proceed to small scale
- 1-2 tests fail → debug specific components
- 3+ tests fail → architecture needs rethink

**Timeline**: 10-15 min training + 30 min analysis = **45 minutes total**

### Phase 1: Small Scale (1-2 days)

**Goal**: Validate scaling from toy to small scale

**Setup**:
- **Model**: 20M params (8 layers, 512 hidden, 6 experts)
- **Data**: 10M tokens (3M code, 7M text)
- **Hardware**: RTX 5070 (12GB VRAM)
- **Training**: 1-2 days
- **Graph**: 15k-30k edges

**Validation**:
- Measure loss curves (should converge smoothly)
- Expert specialization (code vs text routing)
- Graph coverage (15-25%)
- Swarm improves over single expert by >10%

**Success criteria**:
- Model converges (loss < 3.5)
- Experts specialize (JS divergence > 0.2)
- Graph has 15k+ edges, 85%+ precision
- Swarm beats best single expert

**Timeline**: 1-2 days training + 1 day evaluation

### Phase 2: Production Scale (3-4 weeks)

**Goal**: Full training on 1B tokens

**Setup**:
- **Model**: 50M params (12 layers, 768 hidden, 8 experts)
- **Data**: 1B tokens (400M code, 600M text)
- **Hardware**: RTX 5070 (12GB VRAM)
- **Training**: 3-4 weeks
- **Graph**: 220k-350k edges

**Milestones**:
- Week 1: First 100M tokens, validate loss curves
- Week 2: 300M tokens, measure expert specialization
- Week 3: 600M tokens, implement swarm inference
- Week 4: Full 1B tokens, final evaluation

**Success criteria**:
- Model converges (loss < 2.0)
- Passes HumanEval (code generation): >25% pass@1
- Passes MMLU (general knowledge): >40% accuracy
- Swarm improves accuracy >15% over single expert
- Graph coverage 50-65%
- Abstention rate 10-20%, accuracy >90% on non-abstained

**Timeline**: 3-4 weeks training + 1 week evaluation

### Phase 3: Continuous Improvement (Ongoing)

**Goal**: Continuous graph updates and model refinement

**Tasks**:
- User correction interface
- High-confidence model predictions → graph edges
- External knowledge ingestion (docs, Wikipedia)
- Edge versioning and conflict resolution

**Timeline**: Ongoing after Phase 2

---

## Technical Specifications

### Toy Scale (Phase 0: 10-15 min validation)
- **Parameters**: 5M-10M total
- **Expert count**: 6 experts
- **Expert size**: ~1M params each
- **Layers**: 4 RWKV layers
- **Hidden dimension**: 256
- **Vocabulary**: 32k tokens
- **Training data**: 100k-1M tokens
- **Batch size**: 16
- **Sequence length**: 512 tokens
- **Memory usage**: ~2-3GB VRAM
- **Graph edges**: 1.5k-7k

### Small Scale (Phase 1: 1-2 days)
- **Parameters**: 20M total
- **Expert count**: 6 experts
- **Expert size**: ~3M params each
- **Layers**: 8 RWKV layers
- **Hidden dimension**: 512
- **Vocabulary**: 32k tokens
- **Training data**: 10M tokens
- **Batch size**: 24
- **Sequence length**: 1024 tokens
- **Memory usage**: ~5-6GB VRAM
- **Graph edges**: 15k-30k

### Production Scale (Phase 2: 3-4 weeks)
- **Parameters**: 50M total
- **Expert count**: 8 experts
- **Expert size**: ~6M params each
- **Layers**: 12 RWKV layers
- **Hidden dimension**: 768
- **Vocabulary**: 32k-50k tokens
- **Training data**: 1B tokens (400M code, 600M text)
- **Batch size**: 32
- **Sequence length**: 2048 tokens (unlimited at inference)
- **Memory usage**: ~10-11GB VRAM (fits RTX 5070)
- **Graph edges**: 220k-350k

### Training Configuration (Production)

**Compute**:
- **Hardware**: Single RTX 5070 (12GB VRAM)
- **Training time**: 3-4 weeks on single GPU
- **BitNet quantization**: Enables 50M params in 12GB VRAM

**Data**:
- **Total**: 1B tokens
- **Text**: 600M tokens (books, articles, docs, conversations)
- **Code**: 400M tokens (GitHub repos, Stack Overflow, API docs)
- **Sources**: The Stack (code), Wikipedia, books3, web text

**Optimization**:
- Optimizer: AdamW
- Learning rate: 3e-4 with cosine decay
- Warmup steps: 2000
- Batch size: 32 sequences
- Gradient accumulation: 4 (effective batch = 128)
- Mixed precision: FP16 + BitNet

### Knowledge Graph Specifications

**Storage**:
- **Toy/Small scale**: NetworkX (Python, in-memory)
- **Production**: NetworkX or Neo4j (if graph > 500k edges)
- **Vector overlay**: ChromaDB or FAISS for context embedding search
- **Embedding model**: sentence-transformers (all-MiniLM-L6-v2)
- **Embedding dimension**: 384

**Scale progression**:
- **Toy** (100k tokens): 1.5k-7k edges, <10MB memory
- **Small** (10M tokens): 15k-30k edges, <50MB memory
- **Production** (1B tokens): 220k-350k edges, <200MB memory

**Performance targets**:
- Query latency: <10ms per expert per token (with caching)
- Insert throughput: >1000 edges/second
- Graph builds in parallel with training (no blocking)

---

## Success Metrics

### Phase 0: Toy Scale (Mechanism Validation)
- ✅ **MoE routing works**: No expert collapse, >10% utilization each
- ✅ **Experts specialize**: JS divergence > 0.2 between code/text routing
- ✅ **Graph extraction works**: 1.5k-7k edges, >85% precision
- ✅ **Swarm differs from single**: >20% token difference
- ✅ **Graph affects predictions**: >10% token difference with/without graph

### Phase 1: Small Scale (Scaling Validation)
- **Convergence**: Loss < 3.5 on 10M tokens
- **Expert specialization**: JS divergence > 0.2, all experts >5% usage
- **Graph quality**: 15k-30k edges, >85% precision
- **Graph coverage**: 15-25% of predictions have graph context
- **Swarm improvement**: >10% accuracy gain over best single expert

### Phase 2: Production Scale (Quality Targets)
- **Perplexity**: < 2.0 on held-out test set
- **Code generation**: HumanEval >25% pass@1 (vs 15-20% for 50M baseline)
- **General knowledge**: MMLU >40% accuracy
- **Hallucination resistance**: >15% accuracy gain from swarm vs single expert
- **Graph coverage**: 50-65%
- **Abstention**: 10-20% rate, >90% accuracy on non-abstained
- **Inference speed**: 3-4× average slowdown (70-80% fast path usage)

### Continuous Improvement (Phase 3)
- **Graph growth**: >500 edges/day from user corrections + model predictions
- **Update effectiveness**: >20% accuracy improvement on related queries after corrections
- **Edge quality**: >90% precision for semantic facts, >85% for co-occurrence

---

## Data Requirements & Sources

### Toy Scale (Phase 0: 100k-1M tokens)
**Code** (30-40%):
- 10-20 Python repos (your own projects or small popular repos)
- ~30k-400k tokens

**Text** (60-70%):
- 5-10 books or 50-100 articles
- ~70k-600k tokens

**Total**: 100k-1M tokens
**Curation time**: 2-3 hours (just download/filter)

### Small Scale (Phase 1: 10M tokens)
**Code** (30%):
- 30-50 Python repos
- ~3M tokens

**Text** (70%):
- 20-30 books
- Wikipedia sample (500-1000 articles)
- ~7M tokens

**Total**: 10M tokens
**Curation time**: 1-2 days

### Production Scale (Phase 2: 1B tokens)
**Code** (40%): 400M tokens
- **The Stack** (filtered for quality): 300M tokens
  - Python, JavaScript, TypeScript repos
  - Filter: remove generated code, tests without docs
- **Stack Overflow** (code snippets + explanations): 50M tokens
- **API documentation** (Python, JS, web APIs): 50M tokens

**Text** (60%): 600M tokens
- **Books**: 200M tokens (books3 dataset, public domain)
- **Wikipedia**: 150M tokens (English Wikipedia dump)
- **Web text**: 150M tokens (Common Crawl filtered for quality)
- **Conversations/Q&A**: 100M tokens (Reddit, forum posts, filtered)

**Total**: 1B tokens
**Curation time**: 1-2 weeks (download + filter + deduplicate)

**Data sources**:
- The Stack: [huggingface.co/datasets/bigcode/the-stack](https://huggingface.co/datasets/bigcode/the-stack)
- Wikipedia: [dumps.wikimedia.org](https://dumps.wikimedia.org/)
- Books: Project Gutenberg, books3 (if available)
- Web text: Common Crawl (filtered via C4 pipeline)

---

## Open Questions & Research Risks

### Architecture Risks

1. **RWKV + MoE compatibility at tiny scale**: Will 6 experts on 100k tokens specialize meaningfully?
   - **Mitigation**: Toy experiment (Phase 0) tests this in 10-15 minutes
   - **Fallback**: If experts don't specialize, increase to 1M tokens or 8 experts

2. **Swarm specialization at small scale**: Will 6 experts on 10M tokens develop diverse strategies?
   - **Mitigation**: Force diversity via initialization, dropout, traversal strategies
   - **Validation**: Measure disagreement rate and JS divergence every epoch

3. **Adaptive inference effectiveness**: Will 70-80% fast path usage achieve 3-4× average slowdown?
   - **Mitigation**: Tune confidence thresholds empirically in Phase 1
   - **Acceptable**: Even 4-5× average is OK if quality improves 15%+

4. **BitNet stability**: BitNet is experimental (2024 paper), may have training instability
   - **Mitigation**: Start without BitNet, add after Phase 1 if training is stable
   - **Fallback**: Use FP16 if BitNet causes issues

### Knowledge Graph Risks

5. **Initial coverage at toy scale**: 1.5k-7k edges may not affect predictions meaningfully
   - **Mitigation**: Test 5 specifically validates this (graph affects >10% of tokens)
   - **If fails**: Extract more facts or lower confidence thresholds

6. **Graph noise from co-occurrence**: Statistical edges at low frequency (5-10 occurrences) may be noisy
   - **Mitigation**: Ruthless pruning (5 < freq < 100, conf > 0.85)
   - **Validation**: Manual spot-check 100 random edges for correctness

7. **Edge conflicts**: Co-occurrence patterns may contradict semantic facts
   - **Mitigation**: Semantic facts always override co-occurrence in conflicts
   - **Monitor**: Track conflict rate, investigate if >5% of edges conflict

8. **Model prediction injection risk**: High-confidence model predictions may be hallucinations
   - **Mitigation**: Require >0.95 expert agreement + cross-validate with facts
   - **Validation**: Manual review of model-added edges every week

### Training Risks

9. **Single GPU training time**: 3-4 weeks on RTX 5070 is long, hard to iterate
   - **Mitigation**: Validate at toy/small scale first (Phase 0-1)
   - **Option**: Rent cloud GPUs for Phase 2 if toy/small succeed

10. **Overfitting risk at small scale**: 20M params on 10M tokens is borderline
    - **Mitigation**: Early stopping, monitor train/val loss gap
    - **If overfit**: Reduce model to 15M params or add regularization

### Quality Risks

11. **Will 1B tokens be enough?**: Modern models train on 100B+ tokens
    - **Reality**: You're building a specialized tool, not GPT-4
    - **Target**: Competent on common coding/conversation patterns, abstains on rare cases
    - **Mitigation**: Focus on high-quality data curation (quality > quantity)

12. **Swarm may not improve quality**: Voting might flatten outputs, hurt creativity
    - **Mitigation**: Measure improvement at each phase (10% → 15% → 20%)
    - **Fallback**: Use swarm only for factual/code tasks, single expert for creative tasks

---

## References

### Core Architecture
- **RWKV**: Peng et al. 2023, arXiv:2305.13048
- **BitNet**: Ma et al. 2024, arXiv:2402.17764
- **MoE foundations**: Shazeer et al. 2017, arXiv:1701.06538
- **Switch Transformer**: Fedus et al. 2022, arXiv:2101.03961
- **DeepSeekMoE**: 2024, arXiv:2401.06066

### Scaling Laws
- **Chinchilla**: Hoffmann et al. 2022, arXiv:2203.15556 (optimal compute allocation)
- **Kaplan et al.**: 2020, arXiv:2001.08361 (neural scaling laws)

### Knowledge & Hallucination
- **Hallucination bounds**: Kalai & Vempala 2024, arXiv:2311.14648
- **Factuality tuning**: Tian et al. 2023, arXiv:2311.08401
- **Knowledge graphs**: Bordes et al. 2013 (TransE)

---

## Implementation Checklist

### Phase 0: Toy Experiment (10-15 minutes)
- [ ] **Setup** (1-2 hours):
  - [ ] Install dependencies (PyTorch, NetworkX, transformers)
  - [ ] Download RWKV reference implementation (BlinkDL/RWKV-LM)
  - [ ] Curate 100k-1M tokens (10-20 code files + text docs)
  - [ ] Implement tokenizer (BPE, 32k vocab)
- [ ] **Model** (2-3 hours):
  - [ ] RWKV baseline (4 layers, 256 hidden)
  - [ ] Add MoE layers (6 experts, top-1 routing)
  - [ ] Force expert diversity (seeds, dropout, strategies)
- [ ] **Graph** (2-3 hours):
  - [ ] AST fact extractor (Python parser)
  - [ ] Co-occurrence tracker
  - [ ] NetworkX graph storage
- [ ] **Training** (10-15 minutes):
  - [ ] Train 3-5 epochs
  - [ ] Monitor loss, expert utilization
- [ ] **Validation** (30 minutes):
  - [ ] Run 5 critical tests (see Quick Validation Strategy)
  - [ ] **Decision point**: All pass → proceed to Phase 1

### Phase 1: Small Scale (1-2 days training + 1 day eval)
- [ ] **Scale up**:
  - [ ] 20M param model (8 layers, 512 hidden)
  - [ ] 10M tokens (3M code, 7M text)
- [ ] **Train**:
  - [ ] 1-2 days on RTX 5070
  - [ ] Monitor loss curves, expert specialization
- [ ] **Validate**:
  - [ ] Loss < 3.5
  - [ ] Expert diversity metrics
  - [ ] Graph coverage 15-25%
  - [ ] Swarm improves >10% over single expert
  - [ ] **Decision point**: Success → proceed to Phase 2

### Phase 2: Production Scale (3-4 weeks training + 1 week eval)
- [ ] **Data curation** (1-2 weeks):
  - [ ] Download The Stack (code): 400M tokens
  - [ ] Download Wikipedia + books: 600M tokens
  - [ ] Filter, deduplicate, tokenize
- [ ] **Model**:
  - [ ] 50M params (12 layers, 768 hidden, 8 experts)
  - [ ] BitNet quantization integration
- [ ] **Training** (3-4 weeks):
  - [ ] Week 1: 100M tokens, validate curves
  - [ ] Week 2: 300M tokens, measure specialization
  - [ ] Week 3: 600M tokens, implement swarm inference
  - [ ] Week 4: 1B tokens, final training
- [ ] **Evaluation** (1 week):
  - [ ] HumanEval (code generation)
  - [ ] MMLU (general knowledge)
  - [ ] Swarm improvement vs single expert
  - [ ] Graph coverage, abstention rate
  - [ ] Inference speed profiling

### Phase 3: Continuous Improvement (Ongoing)
- [ ] User correction interface
- [ ] High-confidence model predictions → graph
- [ ] External knowledge ingestion
- [ ] Edge versioning, conflict resolution
- [ ] Monitor graph growth, edge quality

---

## Next Steps

### **Immediate** (Today):

1. **Set up environment**:
   ```bash
   # Create conda environment
   conda create -n swarm python=3.10
   conda activate swarm
   
   # Install dependencies
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install transformers networkx matplotlib tqdm
   
   # Clone RWKV reference
   git clone https://github.com/BlinkDL/RWKV-LM.git
   ```

2. **Curate toy dataset**:
   - Pick 10-20 of your own Python files (or small repos)
   - Download 5-10 books from Project Gutenberg
   - Total: 100k-1M tokens
   - Time: 1-2 hours

3. **Implement toy model** (skeleton):
   ```python
   # /src/model/rwkv_moe.py
   class TinyRWKVMoE:
       def __init__(self):
           self.layers = 4
           self.hidden = 256
           self.num_experts = 6
           # ... implement based on RWKV-LM reference
   ```

4. **Run toy experiment**:
   ```bash
   python train_toy.py --data ./data/toy_100k.jsonl --epochs 5
   # Should finish in 10-15 minutes
   ```

5. **Validate mechanisms**:
   ```bash
   python validate_toy.py --checkpoint ./checkpoints/toy_epoch5.pt
   # Run 5 critical tests, get pass/fail
   ```

### **This Week** (if toy succeeds):

6. **Scale to small** (10M tokens, 1-2 days training)
7. **Validate scaling** (loss curves, specialization, swarm)
8. **Decision point**: Proceed to production or pivot

### **Next Month** (if small succeeds):

9. **Curate 1B tokens** (1-2 weeks)
10. **Train production model** (3-4 weeks)
11. **Evaluate quality** (benchmarks, real usage)

**First milestone**: Toy experiment passes all 5 tests (validates architecture) — **~1 day of work, 10-15 min compute**
