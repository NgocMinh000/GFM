# Quick Start Guide - Stage 1 & 2 Workflow

## Prerequisites

1. **Conda environment `gfm-rag` must be activated**
   ```bash
   conda activate gfm-rag
   ```

2. **Configure API credentials in `.env` file**
   ```bash
   # Edit .env and add your API keys
   nano .env
   ```

   Required variables:
   ```bash
   YESCALE_API_BASE_URL=https://api.yescale.io/v1/chat/completions
   YESCALE_API_KEY=your_actual_yescale_key
   ```

## Running the Workflow

### Option 1: Using Shell Scripts (Recommended)

```bash
# Step 1: Run KG Construction
bash run_stage1.sh

# Step 2: Run Entity Resolution
bash run_stage2.sh
```

### Option 2: Direct Python Commands

```bash
# Activate environment first
conda activate gfm-rag

# Stage 1: KG Construction
python -m gfmrag.workflow.stage1_index_dataset

# Stage 2: Entity Resolution
python -m gfmrag.workflow.stage2_entity_resolution
```

## What Each Stage Does

### Stage 1: Knowledge Graph Construction
- **Input**: `./data/hotpotqa/raw/dataset_corpus.json`
- **Output**: `./data/hotpotqa/processed/stage1/kg.txt`
- **Process**:
  1. Extract entities using LLM NER
  2. Extract relationships using LLM OpenIE
  3. Create knowledge graph triples (head, relation, tail)
- **Note**: QA constructor is disabled as per your requirement

### Stage 2: Advanced Entity Resolution
- **Input**: `./data/hotpotqa/processed/stage1/kg.txt`
- **Output**: `./tmp/entity_resolution/kg_clean.txt`
- **Process** (6 sub-stages):
  1. **Stage 0**: Type Inference - Classify entities (drug/disease/symptom/etc.)
  2. **Stage 1**: SapBERT Embedding - Generate 768-dim medical embeddings
  3. **Stage 2**: FAISS Blocking - Find ~150 candidate matches per entity
  4. **Stage 3**: Multi-Feature Scoring - Calculate similarity scores (5 features)
  5. **Stage 4**: Adaptive Thresholding - Apply type-specific thresholds
  6. **Stage 5**: Clustering & Canonicalization - Group synonyms + select canonical names

## Viewing Results

### Stage 1 Output
```bash
# View KG triples
head -20 ./data/hotpotqa/processed/stage1/kg.txt

# Count triples
wc -l ./data/hotpotqa/processed/stage1/kg.txt

# View entity-document mapping
cat ./data/hotpotqa/processed/stage1/document2entities.json | jq '.' | head -50
```

### Stage 2 Output
```bash
# View cleaned KG with SYNONYM_OF edges
head -50 ./tmp/entity_resolution/kg_clean.txt

# View entity type distribution
cat ./tmp/entity_resolution/stage0_types.json | jq '.entity_types | length'

# View candidate pairs
cat ./tmp/entity_resolution/stage2_candidates.json | jq '. | length'

# View final clusters
cat ./tmp/entity_resolution/stage5_clusters.json | jq 'to_entries | length'
```

## Stage 2 Evaluation Metrics

Each stage produces evaluation metrics in the logs:

- **Stage 0**: Type distribution, confidence scores
- **Stage 1**: Embedding statistics (mean/std norm)
- **Stage 2**: Candidates per entity, blocking reduction
- **Stage 3**: Score distribution, feature contributions
- **Stage 4**: Precision/recall estimates, threshold effectiveness
- **Stage 5**: Cluster sizes, canonical name quality

## Logs Location

- **Stage 1**: `./outputs/kg_construction/YYYY-MM-DD/HH-MM-SS/`
- **Stage 2**: Logs printed to console (you can redirect: `bash run_stage2.sh > stage2.log 2>&1`)

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'torch'"
```bash
# Make sure conda environment is activated
conda activate gfm-rag
```

### Error: "OpenAIError: The api_key client option must be set"
```bash
# Check .env file exists and has correct values
cat .env
# Should show YESCALE_API_KEY=sk-... (not placeholder)
```

### Error: "File kg.txt already exists - no processing"
```bash
# Stage 1 skips if output exists. To force recompute:
rm ./data/hotpotqa/processed/stage1/kg.txt
rm ./data/hotpotqa/processed/stage1/document2entities.json
bash run_stage1.sh
```

### Error: "Stage 1 output not found"
```bash
# Run Stage 1 before Stage 2
bash run_stage1.sh
```

## Configuration

### Stage 1 Config
Edit: `gfmrag/workflow/config/stage1_index_dataset.yaml`
- NER/OpenIE models
- Number of processes
- Force recompute

### Stage 2 Config
Edit: `gfmrag/workflow/config/stage2_entity_resolution.yaml`
- SapBERT model
- FAISS parameters (k_neighbors, threshold)
- Feature weights
- Type-specific thresholds
- Clustering method

## Performance Tips

1. **Use GPU for SapBERT** (faster):
   ```yaml
   # In stage2_entity_resolution.yaml
   sapbert:
     device: cuda
   ```

2. **Adjust batch size** based on GPU memory:
   ```yaml
   sapbert:
     batch_size: 256  # Reduce if OOM
   ```

3. **Cache intermediate results**:
   - Set `force: false` in config to reuse cached stages
   - Only set `force: true` when debugging

4. **Parallel processing**:
   ```yaml
   num_processes: 10  # Adjust based on CPU cores
   ```

## Next Steps

After Stage 2 completes, you can:
1. Analyze the cleaned KG
2. Visualize entity clusters
3. Export to graph database (Neo4j, etc.)
4. Use for downstream RAG tasks
