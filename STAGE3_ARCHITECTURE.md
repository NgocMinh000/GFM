# STAGE 3: UMLS MAPPING PIPELINE

## 📋 TABLE OF CONTENTS
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Workflow](#workflow)
4. [Steps (Sub-Stages)](#steps-sub-stages)
5. [Configuration](#configuration)
6. [Metrics & Evaluation](#metrics--evaluation)
7. [Visualizations](#visualizations)
8. [Quality Analysis & Improvements](#quality-analysis--improvements)
9. [Usage](#usage)
10. [Performance](#performance)

---

## OVERVIEW

### Purpose
Stage 3 maps biomedical entities from the Knowledge Graph to standardized UMLS (Unified Medical Language System) CUIs (Concept Unique Identifiers) through a sophisticated 7-stage pipeline.

### Key Achievements
- **7-Stage Pipeline**: Progressive refinement from 128 → 1 candidate
- **Dual Retrieval**: SapBERT (semantic) + TF-IDF (lexical)
- **Ensemble Fusion**: Reciprocal Rank Fusion (RRF)
- **Hard Negative Filtering**: Removes confusingly similar concepts
- **Cross-Encoder Reranking**: PubMedBERT for final scoring
- **Confidence Propagation**: Graph-based confidence through synonym clusters

### Pipeline Position
```
Stage 2 Output: kg_clean.txt
     ↓
[STAGE 3: UMLS Mapping] ← YOU ARE HERE
     ↓
umls_mapping_triples.txt (entity | mapped_to_cui | CUI)
     ↓
[Integration into Knowledge Graph]
```

---

## ARCHITECTURE

### High-Level Design
```
┌────────────────────────────────────────────────────────────────────────┐
│                 STAGE 3: UMLS MAPPING PIPELINE                         │
│                          (7 Steps)                                     │
│                                                                        │
│  Input: kg_clean.txt (entities from Stage 2)                          │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Step 0: UMLS Data Loading & Indexing                            │ │
│  │   ├─ Load 7.9M UMLS names from MRCONSO.RRF                      │ │
│  │   ├─ Load semantic types from MRSTY.RRF                         │ │
│  │   ├─ Build indices: name→CUI, CUI→concepts                      │ │
│  │   └─ Precompute: SapBERT embeddings (3.5h) + TF-IDF (1h)       │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Step 1: Preprocessing & Entity Extraction                        │ │
│  │   ├─ Load kg_clean.txt (comma-delimited)                        │ │
│  │   ├─ Extract unique entities                                     │ │
│  │   ├─ Build synonym clusters (from SYNONYM_OF edges)             │ │
│  │   └─ Normalize: lowercase, punctuation, abbreviations           │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Step 2: Candidate Generation (Dual Retrieval + Ensemble)        │ │
│  │   ├─ Method A: SapBERT FAISS Search                             │ │
│  │   │   └─ Top-64 semantically similar UMLS names                 │ │
│  │   ├─ Method B: TF-IDF Character N-grams                         │ │
│  │   │   └─ Top-64 lexically similar UMLS names                    │ │
│  │   └─ Ensemble: Reciprocal Rank Fusion → Final 128 candidates   │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Step 3: Cluster Aggregation                                      │ │
│  │   ├─ Group candidates across synonym cluster members            │ │
│  │   ├─ Aggregate scores with weighted voting                      │ │
│  │   ├─ Detect outliers (disagreement > threshold)                 │ │
│  │   └─ Output: 64 refined candidates per cluster                  │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Step 4: Hard Negative Filtering                                  │ │
│  │   ├─ Detect hard negatives (high string sim, different CUIs)    │ │
│  │   ├─ Apply penalty to confusing candidates                      │ │
│  │   ├─ Semantic type matching (from KG relations)                 │ │
│  │   └─ Output: 32 filtered candidates                             │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Step 5: Cross-Encoder Reranking                                  │ │
│  │   ├─ Model: microsoft/BiomedNLP-PubMedBERT-base                 │ │
│  │   ├─ Encode: [CLS] entity [SEP] UMLS_name [SEP]                │ │
│  │   ├─ Score: Binary classification (synonym or not)              │ │
│  │   └─ Weighted: 70% cross-encoder + 30% previous score           │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Step 6: Confidence Scoring & Propagation                        │ │
│  │   ├─ Compute confidence (4 factors):                            │ │
│  │   │   • Score margin (top-1 vs top-2)                           │ │
│  │   │   • Absolute score (top-1)                                  │ │
│  │   │   • Cluster consensus (agreement)                           │ │
│  │   │   • Method agreement (SapBERT + TF-IDF)                     │ │
│  │   ├─ Assign tiers: High (≥0.75) / Medium / Low                 │ │
│  │   └─ Propagate high-confidence mappings through clusters        │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  Output: final_umls_mappings.json + triples + statistics              │
└────────────────────────────────────────────────────────────────────────┘
```

### Component Hierarchy
```
Stage3UMLSMapping (Main Pipeline)
    ├─ UMLSLoader (Step 0)
    ├─ Preprocessor (Step 1)
    ├─ CandidateGenerator (Step 2)
    │   ├─ SapBERT encoder
    │   ├─ FAISS index
    │   └─ TF-IDF vectorizer
    ├─ ClusterAggregator (Step 3)
    ├─ HardNegativeFilter (Step 4)
    ├─ CrossEncoderReranker (Step 5)
    ├─ ConfidencePropagator (Step 6)
    └─ MetricsTracker
```

---

## WORKFLOW

### Main Execution Flow
```python
Stage3UMLSMapping.run()
    │
    ├─► Step 3.0: UMLS Data Loading
    │    ├─ Load MRCONSO.RRF (7.9M concept names)
    │    ├─ Load MRSTY.RRF (semantic types)
    │    ├─ Build lookup indices
    │    └─ Precompute embeddings + TF-IDF (one-time)
    │
    ├─► Step 3.1: Preprocessing
    │    ├─ Load kg_clean.txt
    │    ├─ Extract entities (heads + tails)
    │    ├─ Build synonym clusters from SYNONYM_OF edges
    │    └─ Normalize text
    │
    ├─► Step 3.2: Candidate Generation
    │    ├─ Batch encode entities with SapBERT
    │    ├─ FAISS search: Top-64 semantic matches
    │    ├─ TF-IDF search: Top-64 lexical matches
    │    └─ RRF ensemble: Final 128 candidates
    │
    ├─► Step 3.3: Cluster Aggregation
    │    ├─ For each synonym cluster:
    │    │    ├─ Collect all member candidates
    │    │    ├─ Aggregate scores (weighted voting)
    │    │    ├─ Mark outliers (disagreement)
    │    │    └─ Select top-64
    │    └─ Ensure cluster consensus
    │
    ├─► Step 3.4: Hard Negative Filtering
    │    ├─ Detect hard negatives (string sim >0.7, diff CUI)
    │    ├─ Infer semantic type from KG relations
    │    ├─ Match with UMLS semantic types
    │    ├─ Apply type mismatch penalty
    │    └─ Output top-32
    │
    ├─► Step 3.5: Cross-Encoder Reranking
    │    ├─ Encode pairs: (entity, UMLS_name)
    │    ├─ Binary classification score
    │    ├─ Weighted combination:
    │    │    final = 0.7 × cross_score + 0.3 × prev_score
    │    └─ Rerank candidates
    │
    ├─► Step 3.6: Confidence & Propagation
    │    ├─ Compute confidence (4 factors)
    │    ├─ Assign tier (high/medium/low)
    │    ├─ Propagate high-conf through clusters
    │    └─ Generate alternatives for review
    │
    └─► Save Outputs
         ├─ final_umls_mappings.json (full details)
         ├─ umls_mapping_triples.txt (for KG)
         ├─ mapping_statistics.json
         ├─ manual_review_queue.json
         └─ visualizations/
```

---

## STEPS (SUB-STAGES)

### STEP 3.0: UMLS Data Loading & Indexing

**Purpose**: Load UMLS Metathesaurus and build indices

**Input Files**:
- `MRCONSO.RRF`: Concept names (7.9M rows)
- `MRSTY.RRF`: Semantic types (~4.5M rows)
- `MRDEF.RRF`: Definitions (optional)

**MRCONSO.RRF Format**:
```
C0000005|ENG|P|L0000005|PF|S0007492|aspirin|...
│        │   │ │       │  │       │
CUI      Lang TTY ...  Preferred ...  Name
```

**Process**:
1. Parse MRCONSO.RRF (filter ENG language)
2. Build indices:
   - `concepts`: {CUI → Concept object}
   - `umls_aliases`: {name → [CUI1, CUI2, ...]}
   - `name_to_cui`: {name → preferred CUI}
3. Parse MRSTY.RRF → semantic types
4. Precompute (one-time):
   - SapBERT embeddings (7.9M × 768) ~3.5 hours
   - TF-IDF matrix (7.9M names) ~1 hour
   - FAISS index (IndexFlatIP) ~30 mins

**Output**:
- `umls_loader.concepts`: Dict[CUI, Concept]
- `umls_loader.umls_aliases`: Dict[str, List[str]]
- Cached: `sapbert_embeddings.npy`, `tfidf_matrix.npz`, `faiss.index`

**Metrics**:
- Total concepts: 4.5M CUIs
- Total unique names: 7.9M
- Avg names per concept: 1.76
- Concepts with definitions: 2.1M
- Semantic types: 127 unique

**Implementation**: `gfmrag/umls_mapping/umls_loader.py`

---

### STEP 3.1: Preprocessing & Entity Extraction

**Purpose**: Extract entities from kg_clean.txt and build synonym clusters

**Input**: `tmp/entity_resolution/kg_clean.txt`
```
Format: head,relation,tail
aspirin,treats,headache
aspirin,SYNONYM_OF,acetylsalicylic acid
diabetes,SYNONYM_OF,diabetes mellitus
```

**Process**:
1. Parse kg_clean.txt (comma-delimited)
2. Extract unique entities (heads + tails)
3. Build synonym clusters from SYNONYM_OF edges:
   ```python
   # Union-Find clustering
   for (entity1, "SYNONYM_OF", entity2) in triples:
       union(entity1, entity2)

   # Result: synonym_clusters[canonical] = [member1, member2, ...]
   ```
4. Normalize each entity:
   - Lowercase
   - Remove punctuation
   - Expand abbreviations (DM → diabetes mellitus)
   - Normalize whitespace

**Normalization Examples**:
```
"Type-2 Diabetes" → "type 2 diabetes"
"DM2" → "diabetes mellitus type 2"
"Aspirin (ASA)" → "aspirin asa"
```

**Output**:
- `entities`: Dict[str, EntityInfo]
  - EntityInfo: {original, normalized, synonym_group}
- `synonym_clusters`: Dict[str, List[str]]

**Metrics**:
- Total entities: 571
- Total clusters: 571
- Singleton clusters: 569
- Avg cluster size: 1.0
- Max cluster size: 3

**Implementation**: `gfmrag/umls_mapping/preprocessor.py`

---

### STEP 3.2: Candidate Generation

**Purpose**: Generate 128 UMLS candidates per entity using dual retrieval + ensemble

**Strategy**: Combine semantic (SapBERT) and lexical (TF-IDF) for robustness

#### Method A: SapBERT FAISS Search
**Model**: `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`

**Process**:
```python
# 1. Encode entity
entity_emb = sapbert_model.encode([entity])  # (1, 768)

# 2. FAISS search
scores, indices = faiss_index.search(entity_emb, k=64)

# 3. Lookup CUIs
candidates = []
for idx, score in zip(indices[0], scores[0]):
    umls_name = umls_names[idx]
    cui = name_to_cui[umls_name]
    candidates.append(Candidate(cui, umls_name, score, method="sapbert"))
```

**Metrics**:
- Top-1 score: 0.85 (avg)
- Top-64 coverage: ~95%

#### Method B: TF-IDF Character N-grams
**Config**: 3-character trigrams

**Process**:
```python
# 1. Transform entity
entity_vec = tfidf_vectorizer.transform([entity.lower()])  # Sparse vector

# 2. Compute similarities
similarities = (entity_vec * tfidf_matrix.T).toarray()  # (1, 7.9M)

# 3. Top-64
top_indices = np.argsort(-similarities[0])[:64]
candidates = [...]
```

**Metrics**:
- Top-1 score: 0.72 (avg)
- Handles typos and abbreviations well

#### Ensemble: Reciprocal Rank Fusion (RRF)
**Formula**:
```
RRF(candidate) = Σ (1 / (k + rank_in_method))
                 methods

where k = 60 (constant)
```

**Process**:
```python
sapbert_rank = {cui: rank for rank, cui in enumerate(sapbert_candidates)}
tfidf_rank = {cui: rank for rank, cui in enumerate(tfidf_candidates)}

rrf_scores = {}
for cui in (sapbert_rank.keys() | tfidf_rank.keys()):
    rrf_scores[cui] = (
        1/(60 + sapbert_rank.get(cui, 10000)) +
        1/(60 + tfidf_rank.get(cui, 10000))
    )

# Diversity bonus: +5% if in both methods
if cui in sapbert_rank and cui in tfidf_rank:
    rrf_scores[cui] *= 1.05

# Final top-128
final_candidates = sorted(rrf_scores.items(), key=lambda x: -x[1])[:128]
```

**Batch Processing**:
```python
# Process 32 entities at once for 5-10x speedup
batch_size = 32
batch_entities = entities[i:i+batch_size]

# Batch encode
batch_embs = sapbert_model.encode(batch_entities)  # (32, 768)

# Batch FAISS search
batch_scores, batch_indices = faiss_index.search(batch_embs, 64)

# Batch TF-IDF
batch_vecs = tfidf_vectorizer.transform(batch_entities)
batch_sims = (batch_vecs * tfidf_matrix.T).toarray()  # (32, 7.9M)
```

**Performance**:
- Batch processing: 35s/batch (32 entities)
- Total for 600 entities: ~11 minutes
- Speedup: 7.5x vs sequential

**Metrics**:
- Avg candidates per entity: 128
- Avg top-1 score: 0.78
- Candidates with no matches: 0

**Implementation**: `gfmrag/umls_mapping/candidate_generator.py`

---

### STEP 3.3: Cluster Aggregation

**Purpose**: Aggregate candidates across synonym cluster members

**Process**:
```python
for cluster_canonical, cluster_members in synonym_clusters.items():
    # Collect all candidates from all members
    all_candidates = {}
    for member in cluster_members:
        for candidate in entity_candidates[member]:
            cui = candidate.cui
            if cui not in all_candidates:
                all_candidates[cui] = []
            all_candidates[cui].append(candidate)

    # Aggregate scores (weighted voting)
    for cui, cand_list in all_candidates.items():
        avg_score = sum(c.score for c in cand_list) / len(cand_list)
        cluster_support = len(cand_list) / len(cluster_members)
        method_diversity = len(set(c.method for c in cand_list)) / 2

        aggregated_score = (
            0.6 * avg_score +
            0.3 * cluster_support +
            0.1 * method_diversity
        )

        # Outlier detection
        is_outlier = (cluster_support < 0.5)  # Disagreement > 50%

        aggregated_candidates.append(
            Candidate(cui, name, aggregated_score, cluster_support, is_outlier)
        )

    # Select top-64
    final_candidates = sorted(aggregated_candidates, key=lambda c: -c.score)[:64]
```

**Metrics**:
- Avg top-1 score after aggregation: 0.82
- Avg outliers per cluster: 15
- Avg cluster support: 0.85

**Implementation**: `gfmrag/umls_mapping/cluster_aggregator.py`

---

### STEP 3.4: Hard Negative Filtering

**Purpose**: Remove confusingly similar but incorrect concepts

**Hard Negative Definition**:
- String similarity > 0.7 (Levenshtein)
- Different CUIs
- Often share words but have different meanings

**Examples**:
```
Query: "type 2 diabetes"
Hard Negatives:
- "type 1 diabetes" (sim=0.93, WRONG)
- "diabetes insipidus" (sim=0.70, WRONG)
- "gestational diabetes" (sim=0.75, WRONG)

Correct:
- "diabetes mellitus type 2" (sim=0.88, CORRECT)
```

**Detection**:
```python
def detect_hard_negatives(entity, candidates):
    hard_negatives = []
    for cand in candidates:
        # String similarity
        sim = lexical_similarity(entity, cand.name)

        if sim >= 0.7:
            # Check if different CUI than top-1
            if cand.cui != candidates[0].cui:
                hard_negatives.append(cand)

    return hard_negatives
```

**Semantic Type Matching**:
```python
# Infer entity type from KG relations
entity_type = infer_from_relations(entity, kg_context)
# e.g., "aspirin --[treats]--> ..." → infer "drug"

# Get UMLS semantic types for candidate
umls_semantic_types = umls_loader.get_semantic_types(candidate.cui)
# e.g., ["Pharmacologic Substance", "Antibiotic"]

# Check match
type_match = (entity_type in semantic_type_groups[umls_semantic_types])
# If mismatch, apply penalty
```

**Scoring**:
```
filtered_score = 0.7 × base_score
               + 0.2 × type_match_bonus
               - 0.1 × hard_negative_penalty
```

**Output**: Top-32 filtered candidates

**Metrics**:
- Avg top-1 score after filtering: 0.84
- Type match rate: 65%
- Candidates with penalties: 8

**Implementation**: `gfmrag/umls_mapping/hard_negative_filter.py`

---

### STEP 3.5: Cross-Encoder Reranking

**Purpose**: Final reranking with deep bidirectional attention

**Model**: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`

**Architecture**: BERT-based binary classifier
```
[CLS] entity [SEP] UMLS_name [SEP] → score [0, 1]
```

**Process**:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Encode pairs
inputs = tokenizer(
    [entity] * len(candidates),
    [c.name for c in candidates],
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="pt"
)

# Get scores
with torch.no_grad():
    logits = model(**inputs).logits
    cross_scores = torch.softmax(logits, dim=-1)[:, 1].numpy()

# Weighted combination
for candidate, cross_score in zip(candidates, cross_scores):
    final_score = 0.7 * cross_score + 0.3 * candidate.previous_score
    candidate.score = final_score
```

**Training Status**: Zero-shot (NOT fine-tuned)
⚠️ **Warning**: Model weights not initialized for this task
→ Should TRAIN this model for better performance

**Metrics**:
- Avg cross-encoder score: 0.58
- Avg final score: 0.72
- Score improvement vs previous: +0.05

**Implementation**: `gfmrag/umls_mapping/cross_encoder_reranker.py`

---

### STEP 3.6: Confidence Scoring & Propagation

**Purpose**: Assign confidence tiers and propagate through clusters

**Confidence Computation** (4 factors):
```python
def compute_confidence(entity, candidates, cluster_members):
    # Factor 1: Score margin (top-1 vs top-2)
    if len(candidates) >= 2:
        score_margin = candidates[0].score - candidates[1].score
    else:
        score_margin = candidates[0].score  # Only 1 candidate

    # Factor 2: Absolute score
    absolute_score = candidates[0].score

    # Factor 3: Cluster consensus
    # How many cluster members agree on this CUI?
    top_cui = candidates[0].cui
    agreement_count = 0
    for member in cluster_members:
        member_top_cui = get_top_candidate(member).cui
        if member_top_cui == top_cui:
            agreement_count += 1
    cluster_consensus = agreement_count / len(cluster_members)

    # Factor 4: Method agreement
    # Did both SapBERT and TF-IDF rank this highly?
    sapbert_rank = get_rank_in_method(top_cui, "sapbert")
    tfidf_rank = get_rank_in_method(top_cui, "tfidf")
    method_agreement = 1.0 if (sapbert_rank <= 10 and tfidf_rank <= 10) else 0.5

    # Weighted combination
    confidence = (
        0.35 * score_margin +
        0.25 * absolute_score +
        0.25 * cluster_consensus +
        0.15 * method_agreement
    )

    return confidence, {
        "score_margin": score_margin,
        "absolute_score": absolute_score,
        "cluster_consensus": cluster_consensus,
        "method_agreement": method_agreement
    }
```

**Tier Assignment**:
```python
if confidence >= 0.75:
    tier = "high"
elif confidence >= 0.50:
    tier = "medium"
else:
    tier = "low"
```

**Propagation**:
```python
# For each cluster:
if tier == "high" and cluster_consensus >= 0.8:
    # Propagate to all members
    for member in cluster_members:
        if not member.has_mapping:
            member.mapping = top_mapping
            member.confidence = top_confidence * 0.9  # Penalty
            member.is_propagated = True
```

**Alternatives** (for review):
```python
if tier in ["medium", "low"]:
    alternatives = candidates[1:4]  # Top 2-4
    mapping.alternatives = alternatives
```

**Metrics**:
- Total mappings: 571
- High confidence: 2 (0.35%)
- Medium confidence: 569 (99.65%)
- Low confidence: 0 (0%)
- Propagated: 0 (0%)
- Avg confidence: 0.618
- Avg score margin: 0.055
- Avg cluster consensus: 0.92

**Implementation**: `gfmrag/umls_mapping/confidence_propagator.py`

---

## CONFIGURATION

### Main Config File
`gfmrag/workflow/config/stage3_umls_mapping.yaml`

```yaml
# Input/Output
input:
  kg_clean_path: tmp/entity_resolution/kg_clean.txt
  umls_data_dir: data/umls/META

output:
  root_dir: tmp/umls_mapping

# Step 3.0: UMLS Data
umls:
  files:
    mrconso: ${input.umls_data_dir}/MRCONSO.RRF
    mrsty: ${input.umls_data_dir}/MRSTY.RRF
  language: ENG
  cache_dir: ${input.umls_data_dir}/processed
  precompute:
    embeddings: true  # One-time: 3.5h
    tfidf: true       # One-time: 1h
    faiss: true       # One-time: 30min

# Step 3.1: Preprocessing
preprocessing:
  normalize:
    lowercase: true
    remove_punctuation: true
    expand_abbreviations: true

# Step 3.2: Candidate Generation
candidate_generation:
  sapbert:
    model: cambridgeltl/SapBERT-from-PubMedBERT-fulltext
    batch_size: 256
    device: cuda
    top_k: 64

  tfidf:
    analyzer: char
    ngram_range: [3, 3]
    top_k: 64

  ensemble:
    method: reciprocal_rank_fusion
    k_constant: 60
    diversity_bonus: 0.05
    final_k: 128

# Step 3.3: Cluster Aggregation
cluster_aggregation:
  score_weights:
    avg_score: 0.6
    cluster_consensus: 0.3
    method_diversity: 0.1
  outlier_detection:
    enabled: true
    threshold: 0.5
  output_k: 64

# Step 3.4: Hard Negative Filtering
hard_negative_filtering:
  hard_negative:
    similarity_threshold: 0.7
    penalty_weight: 0.1
  score_weights:
    base_score: 0.7
    type_match: 0.2
    hard_negative_penalty: 0.1
  output_k: 32

# Step 3.5: Cross-Encoder
cross_encoder:
  model: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
  device: cuda
  inference:
    batch_size: 32
    max_length: 256
  score_weights:
    cross_encoder: 0.7
    previous_stage: 0.3

# Step 3.6: Confidence
confidence:
  factors:
    score_margin: 0.35
    absolute_score: 0.25
    cluster_consensus: 0.25
    method_agreement: 0.15
  tiers:
    high: 0.75
    medium: 0.50
  propagation:
    enabled: true
    min_cluster_agreement: 0.8
    confidence_penalty: 0.9

# General
general:
  device: cuda
  force_recompute: false
  save_intermediate: true
```

---

## METRICS & EVALUATION

### Metrics File
`tmp/umls_mapping/pipeline_metrics.json`

### Step Metrics

#### Step 3.0: UMLS Loader
```json
{
  "total_concepts": 4500000,
  "total_unique_names": 7900000,
  "avg_names_per_concept": 1.76,
  "concepts_with_definitions": 2100000
}
```

#### Step 3.2: Candidate Generation
```json
{
  "entities_with_candidates": 571,
  "avg_candidates_per_entity": 128,
  "entities_with_no_candidates": 0,
  "avg_top1_score": 0.78
}
```

#### Step 3.6: Final Confidence
```json
{
  "total_mappings": 571,
  "high_confidence": 2,
  "medium_confidence": 569,
  "low_confidence": 0,
  "high_confidence_pct": "0.35%",
  "medium_confidence_pct": "99.65%",
  "avg_confidence": 0.618,
  "avg_score_margin": 0.055,
  "avg_cluster_consensus": 0.92
}
```

### Performance Metrics
```json
{
  "total_duration_seconds": 747.25,
  "stage_durations": {
    "Stage 3.0": "5.23s",
    "Stage 3.1": "0.12s",
    "Stage 3.2": "671.98s",  // Bottleneck: 90% of time
    "Stage 3.3": "24.15s",
    "Stage 3.4": "21.32s",
    "Stage 3.5": "18.45s",
    "Stage 3.6": "6.00s"
  }
}
```

---

## VISUALIZATIONS

### Generated Plots
Location: `tmp/umls_mapping/visualizations/`

#### 1. Candidate Reduction Funnel (`candidate_reduction_funnel.png`)
Shows progressive refinement: 128 → 64 → 32 → 1

#### 2. Confidence Distribution (`confidence_distribution.png`)
- Pie chart + Bar chart
- Shows High/Medium/Low distribution
- Target line at 60% for high confidence

#### 3. Score Progression (`score_progression.png`)
Line plot showing avg score across steps

#### 4. Stage Timing (`stage_timing.png`)
Horizontal bar chart of execution time per step

#### 5. Cluster Statistics (`cluster_statistics.png`)
Bar chart of cluster size distribution

#### 6. Metric Heatmap (`metric_heatmap.png`)
Normalized heatmap of all metrics across stages

#### 7. Quality Metrics (`quality_metrics.png`)
Comprehensive 2×2 grid:
- Actual vs Target confidence
- Confidence factors
- Propagation pie chart
- Quality summary table

---

## QUALITY ANALYSIS & IMPROVEMENTS

### ⚠️ Current Quality Issues

**Critical Gaps** (from visualization analysis):

| Metric | Actual | Target | Gap | Status |
|--------|--------|--------|-----|--------|
| High Confidence % | 0.35% | 60% | -99.7% | 🔴 CRITICAL |
| Score Margin | 0.055 | >0.20 | -73% | 🔴 CRITICAL |
| Overall Confidence | 0.618 | >0.65 | -5% | 🟡 Minor |
| Low Confidence % | 0% | <20% | ✓ | ✅ Good |

**Root Causes**:
1. **Cross-Encoder NOT trained** → Poor reranking (zero-shot)
2. **Weak candidate quality** → Initial candidates too similar
3. **Insufficient hard negative filtering** → Confusing candidates remain
4. **Suboptimal ensemble weights** → RRF may need tuning

---

### 🎯 Improvement Recommendations

#### PHASE 1: Quick Wins (1-2 hours)

**1. Increase Candidate Quality Thresholds**
```yaml
# candidate_generation.yaml
candidate_generation:
  sapbert:
    top_k: 32           # Reduce from 64 (more selective)
    min_score: 0.70     # Add minimum threshold

  tfidf:
    top_k: 32
    min_score: 0.60

  ensemble:
    final_k: 64         # Reduce from 128
```

**Expected Impact**:
- High confidence: 0.35% → 5-10%
- Score margin: 0.055 → 0.10

**2. Aggressive Hard Negative Filtering**
```yaml
# hard_negative_filtering.yaml
hard_negative_filtering:
  hard_negative:
    similarity_threshold: 0.60  # More strict (was 0.70)
    penalty_weight: 0.25        # Stronger penalty (was 0.10)
    top_k_check: 5              # Check top-5 for hard negs

  score_weights:
    base_score: 0.6             # Reduce (was 0.7)
    type_match: 0.25            # Increase (was 0.2)
    hard_negative_penalty: 0.15 # Increase (was 0.1)
```

**Expected Impact**:
- Score margin: +0.05
- Type match accuracy: +10%

**3. Tune Ensemble Weighting**
```yaml
# candidate_generation.yaml
ensemble:
  sapbert_weight: 0.65   # Increase SapBERT (was 0.50 implicit)
  tfidf_weight: 0.35     # Decrease TF-IDF (was 0.50)
  k_constant: 50         # Lower k (more weight to top ranks)
```

**Expected Impact**:
- Top-1 accuracy: +3-5%

---

#### PHASE 2: Major Improvements (1-2 days)

**1. CRITICAL: Fine-tune Cross-Encoder**

**Why**: Model currently uses zero-shot → very poor performance

**Approach**:
```python
# Training dataset: MedMentions or BC5CDR
# - Positive pairs: (entity, correct_UMLS_name, label=1)
# - Negative pairs: (entity, incorrect_UMLS_name, label=0)
# - Hard negatives: (entity, similar_but_wrong, label=0)

from transformers import AutoModelForSequenceClassification, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/BiomedNLP-PubMedBERT-base",
    num_labels=2
)

# Fine-tune with hard negative mining
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)
trainer.train()
```

**Expected Impact**:
- High confidence: 0.35% → 40-60%
- Score margin: 0.055 → 0.25+
- Overall confidence: 0.618 → 0.75+

**2. Implement Confidence Recalibration**
```python
# After cross-encoder reranking, recalibrate confidence
def recalibrate_confidence(raw_score, calibration_params):
    # Platt scaling or isotonic regression
    calibrated_score = platt_scaling(raw_score, calibration_params)
    return calibrated_score
```

---

### 📊 Expected Results After Improvements

| Metric | Current | Phase 1 | Phase 2 | Target |
|--------|---------|---------|---------|--------|
| High Conf % | 0.35% | 5-10% | 40-60% | 60% |
| Score Margin | 0.055 | 0.10 | 0.25+ | >0.20 |
| Overall Conf | 0.618 | 0.64 | 0.75+ | >0.65 |

---

## USAGE

### Basic Usage
```bash
python -m gfmrag.workflow.stage3_umls_mapping
```

### First-Time Setup (Precompute)
```bash
# This will take ~5 hours for first run
# - SapBERT embeddings: 3.5h
# - TF-IDF matrix: 1h
# - FAISS index: 30min

python -m gfmrag.workflow.stage3_umls_mapping \
  umls.precompute.embeddings=true \
  umls.precompute.tfidf=true \
  umls.precompute.faiss=true
```

### Subsequent Runs (Use Cache)
```bash
# Much faster (~12 minutes)
python -m gfmrag.workflow.stage3_umls_mapping \
  general.force_recompute=false
```

---

## PERFORMANCE

### Benchmark (571 entities)

| Step | Duration | % of Total |
|------|----------|------------|
| 3.0: UMLS Loading | 5s | 0.7% |
| 3.1: Preprocessing | 0.1s | 0.01% |
| 3.2: Candidate Gen | 672s | 89.9% |
| 3.3: Aggregation | 24s | 3.2% |
| 3.4: Filtering | 21s | 2.8% |
| 3.5: Reranking | 18s | 2.4% |
| 3.6: Confidence | 6s | 0.8% |
| **Total** | **747s** | **100%** |
| **= 12.5 min** | | |

**Bottleneck**: Step 3.2 Candidate Generation (90%)

**Optimization**: Batch processing (32 entities/batch)
- Speedup: 7.5x vs sequential
- Total improvement: 84 min → 11 min

---

## TROUBLESHOOTING

### Issue 1: Low High-Confidence Mappings
**Solution**: Implement Phase 1 + 2 improvements above

### Issue 2: Slow Candidate Generation
**Solution**: Already optimized with batching

### Issue 3: GPU Out of Memory
**Solution**:
```yaml
candidate_generation:
  sapbert:
    batch_size: 128  # Reduce from 256
```

---

**Last Updated**: 2026-01-05
**Version**: 1.0
**Author**: GFM-RAG Team

**⚠️ NEXT STEPS**: Implement Phase 1 improvements (see Quality Analysis section)
