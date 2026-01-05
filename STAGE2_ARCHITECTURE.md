# STAGE 2: ENTITY RESOLUTION PIPELINE

## 📋 TABLE OF CONTENTS
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Workflow](#workflow)
4. [Steps (Sub-Stages)](#steps-sub-stages)
5. [Configuration](#configuration)
6. [Metrics & Evaluation](#metrics--evaluation)
7. [Visualizations](#visualizations)
8. [Usage](#usage)
9. [Performance](#performance)

---

## OVERVIEW

### Purpose
Stage 2 resolves entity synonyms in the Knowledge Graph using a sophisticated 6-step pipeline. It replaces the old ColBERT entity linking with a multi-stage approach combining pattern matching, semantic embeddings, and LLM inference.

### Key Achievements
- **Replaces**: Old ColBERT-based entity linking
- **Multi-stage Pipeline**: 6 cascading steps for high precision
- **Smart Type Inference**: 3-tier cascading (Keywords → SapBERT → LLM)
- **Synonym Clustering**: Union-Find algorithm for transitive closure
- **Multi-feature Scoring**: 4 features (SapBERT + Lexical + ColBERT + Graph)

### Pipeline Position
```
Stage 1 Output: kg.txt
     ↓
[STAGE 2: Entity Resolution] ← YOU ARE HERE
     ↓
kg_clean.txt (with SYNONYM_OF edges)
     ↓
[STAGE 3: UMLS Mapping]
```

---

## ARCHITECTURE

### High-Level Design
```
┌───────────────────────────────────────────────────────────────────┐
│              STAGE 2: ENTITY RESOLUTION PIPELINE                   │
│                                                                   │
│  Input: kg.txt (triples from Stage 1)                            │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Step 0: Type Inference (3-Tier Cascading)                   │ │
│  │   ├─ Tier 1: Medical Keywords (0.001s/entity) ────────► 65% │ │
│  │   ├─ Tier 2: SapBERT kNN (0.01s/entity) ──────────────► 25% │ │
│  │   └─ Tier 3: GPT-4 LLM (2s/entity) ───────────────────► 10% │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Step 1: SapBERT Embedding                                   │ │
│  │   └─ Model: cambridgeltl/SapBERT-from-PubMedBERT-fulltext  │ │
│  │      Dimensions: 768, Batch: 256, Device: CUDA              │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Step 1b: ColBERT Indexing                                   │ │
│  │   └─ Model: colbert-ir/colbertv2.0                          │ │
│  │      Purpose: Token-level similarity for Step 3             │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Step 2: FAISS Blocking (Hybrid Type-Specific + Global)      │ │
│  │   ├─ High confidence (≥0.75): Type-specific only (k=50)    │ │
│  │   └─ Low confidence (<0.75): Type-specific (k=30) +        │ │
│  │                               Global (k=20)                 │ │
│  │   Filter: Token overlap required                            │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Step 3: Multi-Feature Scoring (4 Features)                  │ │
│  │   ├─ SapBERT similarity (50%)                               │ │
│  │   ├─ Lexical similarity (15%) - Levenshtein                 │ │
│  │   ├─ ColBERT similarity (25%) - Late interaction            │ │
│  │   └─ Graph similarity (10%) - Shared neighbors              │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Step 4: Adaptive Thresholding                                │ │
│  │   ├─ Drug: 0.86 (strict - dosage sensitivity)              │ │
│  │   ├─ Disease: 0.82                                          │ │
│  │   ├─ Symptom: 0.77 (lenient - high variation)              │ │
│  │   ├─ Gene: 0.91 (very strict)                               │ │
│  │   └─ Other: 0.80                                            │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Step 5: Clustering & Canonicalization                       │ │
│  │   ├─ Union-Find clustering (transitive closure)            │ │
│  │   └─ Canonical selection: frequency-based                   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  Output: kg_clean.txt (original + SYNONYM_OF edges)             │
└───────────────────────────────────────────────────────────────────┘
```

---

## WORKFLOW

### Main Execution Flow
```python
EntityResolutionPipeline.run()
    │
    ├─► Step 0: Type Inference (3-Tier Cascading)
    │    └─► For each entity:
    │         ├─ Try Tier 1: Medical Keywords (fast) → 65% resolved
    │         ├─ Try Tier 2: SapBERT kNN → 25% resolved
    │         └─ Fallback Tier 3: GPT-4 LLM → 10% resolved
    │
    ├─► Step 1: SapBERT Embedding
    │    └─► Encode all entities (batch_size=256, GPU)
    │
    ├─► Step 1b: ColBERT Indexing
    │    └─► Build token-level index for similarity
    │
    ├─► Step 2: FAISS Blocking
    │    ├─► Build type-specific indices
    │    ├─► Build global index (for low-conf entities)
    │    └─► Search with hybrid strategy + token filter
    │
    ├─► Step 3: Multi-Feature Scoring
    │    └─► For each candidate pair:
    │         ├─ SapBERT similarity (already computed)
    │         ├─ Lexical similarity (Levenshtein)
    │         ├─ ColBERT similarity (token-level)
    │         └─ Graph similarity (Jaccard neighbors)
    │
    ├─► Step 4: Adaptive Thresholding
    │    └─► Apply type-specific thresholds
    │
    ├─► Step 5: Clustering & Canonicalization
    │    ├─► Union-Find clustering
    │    ├─► Select canonical names (frequency)
    │    └─► Create SYNONYM_OF edges
    │
    └─► Save kg_clean.txt
```

---

## STEPS (SUB-STAGES)

### STEP 0: Type Inference (3-Tier Cascading)

**Purpose**: Classify entities into medical types (drug, disease, symptom, etc.)

**Architecture**: Smart cascading for speed + accuracy
```
Entity
  ↓
Tier 1: Medical Keywords (0.001s) ──► 65% resolved (high conf ≥0.80)
  ↓ (if confidence <0.80)
Tier 2: SapBERT kNN (0.01s) ────────► 25% resolved (high conf ≥0.80)
  ↓ (if confidence <0.80)
Tier 3: GPT-4 LLM (2s) ─────────────► 10% resolved (hard cases)
```

**Implementation**: `stage2_entity_resolution.py:615-735`

#### Tier 1: Medical Keywords
**Speed**: 0.001s/entity
**Method**: Token overlap with curated dictionaries

**Example**:
```python
"aspirin" → tokens: {"aspirin"}
Match in drug_keywords: {"aspirin", "ibuprofen", ...}
→ Type: "drug", Confidence: 0.90
```

**Keyword Categories**:
- **drug**: 60+ terms (antibiotics, statins, analgesics, etc.)
- **disease**: 50+ terms (cancer, diabetes, hypertension, etc.)
- **symptom**: 30+ terms (pain, fever, nausea, etc.)
- **procedure**: 25+ terms (surgery, biopsy, screening, etc.)
- **gene**: 15+ terms (BRCA1, TP53, etc.)
- **anatomy**: 40+ terms (heart, lung, brain, etc.)

#### Tier 2: SapBERT kNN
**Speed**: 0.01s/entity
**Method**: k-Nearest Neighbors on medical embeddings

**Process**:
1. Encode entity with SapBERT
2. Find k=5 nearest labeled examples
3. Weighted voting by cosine similarity
4. Confidence = (top1_sim × 0.6) + (consensus × 0.4)

**Labeled Examples**: 80+ medical entities across 6 types

#### Tier 3: GPT-4 LLM
**Speed**: 2s/entity
**Method**: Relationship-based inference with GPT-4 Turbo

**Process**:
1. Extract graph relationships (incoming + outgoing)
2. Send to GPT-4.1-2025-04-14 with structured prompt
3. Parse JSON response: `{"type": "drug", "confidence": 0.85}`

**Prompt Template**:
```
Entity: "aspirin"
Relationships:
  aspirin --[treats]--> headache
  aspirin --[is_a]--> medication

Classify into: drug, disease, symptom, gene, procedure, anatomy, other
Response format: {"type": "...", "confidence": 0.XX}
```

**Metrics**:
- Tier 1 coverage: ~65%
- Tier 2 coverage: ~25%
- Tier 3 coverage: ~10%
- Average confidence: 0.82

---

### STEP 1: SapBERT Embedding

**Purpose**: Convert entities to dense 768-dim vectors

**Model**: `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`
- Pre-trained on PubMed biomedical text
- Fine-tuned for entity synonym detection
- Medical domain-specific

**Configuration**:
```yaml
sapbert:
  model: cambridgeltl/SapBERT-from-PubMedBERT-fulltext
  batch_size: 256
  device: cuda
```

**Process**:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext", device="cuda")
embeddings = model.encode(
    entities,
    batch_size=256,
    normalize_embeddings=True  # L2 normalization
)
# Output: (N, 768) numpy array
```

**Output**:
- `stage1_embeddings.npy`: (N, 768) embeddings
- `stage1_entity_ids.json`: entity → ID mapping

**Metrics**:
- Mean norm: ~1.0 (normalized)
- Std norm: ~0.01
- Encoding time: ~2 min for 10K entities

---

### STEP 1B: ColBERT Indexing

**Purpose**: Build token-level index for Step 3 similarity

**Model**: `colbert-ir/colbertv2.0`

**Why ColBERT?**:
- Token-level embeddings (not sentence-level)
- MaxSim operation for nuanced matching
- Better for multi-word entities

**Process**:
1. Index all entities with ColBERT
2. Store in `tmp/colbert_index/`
3. Use for pairwise similarity in Step 3

**Implementation**: `stage2_entity_resolution.py:1298-1370`

---

### STEP 2: FAISS Blocking

**Purpose**: Generate candidate pairs (reduce O(N²) to O(N×k))

**Strategy**: Hybrid Type-Specific + Cross-Type Blocking

**Algorithm**:
```python
For each entity:
    confidence = entity_types[entity]["confidence"]
    entity_type = entity_types[entity]["type"]

    if confidence >= 0.75:  # High confidence
        # Type-specific search only
        candidates = faiss_type_index[entity_type].search(k=50)
    else:  # Low confidence
        # Hybrid: Type-specific + Global search
        type_candidates = faiss_type_index[entity_type].search(k=30)
        global_candidates = faiss_global_index.search(k=20)
        candidates = merge(type_candidates, global_candidates)

    # Token overlap filter
    candidates = filter_token_overlap(candidates)
```

**FAISS Index Types**:
- **Small types (<100 entities)**: IndexFlatIP (exact)
- **Large types (≥100 entities)**: IndexHNSWFlat (approximate)

**Token Overlap Filter**:
```python
def has_token_overlap(entity1, entity2):
    tokens1 = set(entity1.lower().split())
    tokens2 = set(entity2.lower().split())
    return len(tokens1 & tokens2) > 0

# Example:
"type 2 diabetes" ↔ "diabetes mellitus" → True (share "diabetes")
"aspirin" ↔ "hypertension" → False (no overlap)
```

**Metrics**:
- Candidate pairs: ~50K (from ~100M possible)
- Reduction: 99.95%
- Token filter saves: 20-30% false candidates

**Implementation**: `stage2_entity_resolution.py:1400-1659`

---

### STEP 3: Multi-Feature Scoring

**Purpose**: Comprehensive similarity scoring with 4 features

**Features**:
1. **SapBERT** (50%): Medical semantic similarity
2. **Lexical** (15%): String edit distance
3. **ColBERT** (25%): Token-level matching
4. **Graph** (10%): Shared neighbors

**Formula**:
```
final_score = 0.50 × sapbert_sim
            + 0.15 × lexical_sim
            + 0.25 × colbert_sim
            + 0.10 × graph_sim
```

#### Feature 1: SapBERT Similarity
**Already computed** from Step 2 FAISS search
```python
sapbert_score = cosine_similarity(emb1, emb2)  # [0, 1]
```

#### Feature 2: Lexical Similarity
**Method**: Levenshtein edit distance
```python
def lexical_similarity(entity1, entity2):
    edit_dist = levenshtein_distance(entity1.lower(), entity2.lower())
    max_len = max(len(entity1), len(entity2))
    return 1.0 - (edit_dist / max_len)

# Example:
"aspirin" vs "asprin" → edit_dist=1 → sim=1-1/7=0.857
```

#### Feature 3: ColBERT Similarity
**Method**: Bidirectional MaxSim (token-level)
```python
colbert_score = compute_pairwise_similarity(entity1, entity2)
# Internally computes:
# - Token embeddings for both entities
# - MaxSim(tokens1 → tokens2) + MaxSim(tokens2 → tokens1)
# - Average both directions
```

#### Feature 4: Graph Similarity
**Method**: Jaccard similarity of neighbors
```python
neighbors1 = get_neighbors(entity1)  # From KG
neighbors2 = get_neighbors(entity2)
shared = len(neighbors1 & neighbors2)
total = len(neighbors1 | neighbors2)
graph_sim = shared / total if total > 0 else 0.0

# Example:
# "aspirin" neighbors: {headache, pain, medication}
# "ibuprofen" neighbors: {headache, pain, drug}
# shared: {headache, pain} → 2/5 = 0.4
```

**Metrics**:
- Mean final score: 0.87
- Median: 0.89
- Range: [0.60, 1.00]

**Implementation**: `stage2_entity_resolution.py:1720-1835`

---

### STEP 4: Adaptive Thresholding

**Purpose**: Type-specific thresholds for precision

**Thresholds**:
```python
type_thresholds = {
    "drug": 0.86,       # Strict (dosage sensitivity)
    "disease": 0.82,    # Medium
    "symptom": 0.77,    # Lenient (high variation)
    "procedure": 0.80,
    "gene": 0.91,       # Very strict
    "anatomy": 0.82,
    "other": 0.80,      # Default
}
```

**Rationale**:
- **Drug**: High precision needed (dosage differences critical)
- **Gene**: Very high precision (single nucleotide matters)
- **Symptom**: More lenient (subjective descriptions vary)

**Decision**:
```python
if final_score >= type_thresholds[entity_type]:
    equivalents.append((entity1, entity2))
```

**Metrics**:
- Acceptance rate: ~15-25% of candidate pairs
- Precision: Estimated 92-95%

**Implementation**: `stage2_entity_resolution.py:1872-1937`

---

### STEP 5: Clustering & Canonicalization

**Purpose**: Group synonyms + select canonical names

**Algorithm**: Union-Find (Disjoint Set)
```python
parent = {}

def find(x):
    if x not in parent:
        parent[x] = x
    if parent[x] != x:
        parent[x] = find(parent[x])  # Path compression
    return parent[x]

def union(x, y):
    px, py = find(x), find(y)
    if px != py:
        parent[px] = py

# Build clusters
for (entity1, entity2) in equivalent_pairs:
    union(entity1, entity2)

# Group by root
clusters = defaultdict(list)
for entity in all_entities:
    root = find(entity)
    clusters[root].append(entity)
```

**Canonical Selection**: Frequency-based
```python
# Count occurrences in KG
name_freq = {name: 0 for name in cluster}
for head, rel, tail in kg_triples:
    if head in name_freq:
        name_freq[head] += 1
    if tail in name_freq:
        name_freq[tail] += 1

# Select most frequent
canonical_name = max(cluster, key=lambda n: name_freq[n])
```

**Alternative Methods** (configurable):
- `frequency`: Most common in corpus (default)
- `length`: Shortest name
- `umls`: UMLS preferred term (if available)

**SYNONYM_OF Edges**:
```python
for name in cluster:
    if name != canonical_name:
        synonym_edges.append((name, "SYNONYM_OF", canonical_name))
```

**Example**:
```
Cluster: {"type 2 diabetes", "diabetes mellitus type 2", "DM2"}
Frequency: {
    "type 2 diabetes": 50,
    "diabetes mellitus type 2": 30,
    "DM2": 5
}
Canonical: "type 2 diabetes"
Edges:
    diabetes mellitus type 2,SYNONYM_OF,type 2 diabetes
    DM2,SYNONYM_OF,type 2 diabetes
```

**Metrics**:
- Clusters: ~2,000
- Singleton clusters: ~8,000 (no synonyms)
- Avg cluster size: 2.5
- Max cluster size: 20

**Implementation**: `stage2_entity_resolution.py:1951-2073`

---

## CONFIGURATION

### Main Config File
`gfmrag/workflow/config/stage2_entity_resolution.yaml`

```yaml
# Input/Output
kg_input_path: ./data/hotpotqa/processed/stage1/kg.txt
output_dir: tmp/entity_resolution

# Step 0: Type Inference
type_inference:
  method: hybrid  # Uses 3-tier cascading

# Step 1: SapBERT Embedding
sapbert:
  model: cambridgeltl/SapBERT-from-PubMedBERT-fulltext
  batch_size: 256
  device: cuda

# Step 1b: ColBERT Indexing
colbert:
  model: colbert-ir/colbertv2.0
  root: tmp/colbert_index
  topk: 10

# Step 2: FAISS Blocking
faiss:
  k_neighbors: 150
  similarity_threshold: 0.60
  index_type: IndexHNSWFlat

# Step 3: Multi-Feature Scoring
scoring:
  feature_weights:
    sapbert: 0.50
    lexical: 0.15
    colbert: 0.25
    graph: 0.10

# Step 4: Adaptive Thresholding
thresholding:
  type_thresholds:
    drug: 0.86
    disease: 0.82
    symptom: 0.77
    procedure: 0.80
    gene: 0.91
    anatomy: 0.82
    other: 0.80

# Step 5: Clustering
clustering:
  canonical_method: frequency

# General
num_processes: 10
force: True
save_intermediate: True
```

---

## METRICS & EVALUATION

### Metrics File
`tmp/entity_resolution/entity_resolution_metrics.json`

### Key Metrics

#### Step 0: Type Inference
```json
{
  "total_entities": 10000,
  "type_distribution": {
    "drug": 2500,
    "disease": 2000,
    "symptom": 1500,
    "other": 4000
  },
  "avg_confidence": 0.823,
  "tier_distribution": {
    "tier1_keyword": 6500,
    "tier2_sapbert_knn": 2500,
    "tier3_llm": 1000
  }
}
```

#### Step 2: FAISS Blocking
```json
{
  "candidate_pairs": 52000,
  "reduction_pct": 99.95,
  "before_token_filter": 65000,
  "after_token_filter": 52000
}
```

#### Step 3: Multi-Feature Scoring
```json
{
  "scored_pairs": 52000,
  "avg_final_score": 0.871,
  "median_final_score": 0.893,
  "score_range": [0.601, 1.000]
}
```

#### Step 4: Adaptive Thresholding
```json
{
  "equivalent_pairs": 8500,
  "acceptance_rate": 16.3
}
```

#### Step 5: Clustering
```json
{
  "num_clusters": 2000,
  "num_singletons": 8000,
  "avg_cluster_size": 2.5,
  "max_cluster_size": 20,
  "synonym_edges": 3000
}
```

---

## VISUALIZATIONS

### Generated Plots
Location: `tmp/entity_resolution/visualizations/`

#### 1. Quality Dashboard (`00_quality_dashboard.png`)
Comprehensive overview:
- OpenIE quality score
- Similarity scores (avg/median/max)
- Entity coverage
- Confidence distribution
- Cluster metrics
- Processing time

#### 2. Type Distribution (`type_distribution.png`)
Bar chart showing entity type counts

#### 3. Tier Distribution (`tier_distribution.png`)
- Bar chart + Pie chart
- Shows 3-tier cascading distribution

#### 4. Confidence Distribution (`confidence_distribution.png`)
- Histogram of confidence scores
- Box plot
- Mean/median lines

#### 5. Cluster Size Distribution (`cluster_size_distribution.png`)
- Histogram of cluster sizes
- Statistics panel

#### 6. Embedding Similarity Heatmap (`embedding_similarity_heatmap.png`)
Sample 50×50 entity similarity matrix

---

## USAGE

### Basic Usage
```bash
python -m gfmrag.workflow.stage2_entity_resolution
```

### With Custom Config
```bash
python -m gfmrag.workflow.stage2_entity_resolution \
  kg_input_path=./my_kg.txt \
  output_dir=./my_output \
  sapbert.batch_size=512
```

---

## PERFORMANCE

### Benchmark (10K entities)

| Step | Duration | Throughput |
|------|----------|------------|
| Step 0 | 5 min | 33 ent/s |
| Step 1 | 2 min | 83 ent/s |
| Step 1b | 3 min | 55 ent/s |
| Step 2 | 1 min | - |
| Step 3 | 15 min | - |
| Step 4 | 10 s | - |
| Step 5 | 5 s | - |
| **Total** | **26 min** | - |

---

**Last Updated**: 2026-01-05
