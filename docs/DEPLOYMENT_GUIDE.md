# UMLS Mapping Pipeline - Complete Deployment Guide

**Hướng dẫn triển khai và chạy hoàn chỉnh Stage 3 UMLS Mapping Pipeline**

---

## 📋 Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Installation](#3-installation)
4. [Data Preparation](#4-data-preparation)
5. [Configuration](#5-configuration)
6. [Pipeline Execution](#6-pipeline-execution)
7. [Validation](#7-validation)
8. [Troubleshooting](#8-troubleshooting)
9. [Performance Tuning](#9-performance-tuning)
10. [Production Deployment](#10-production-deployment)

---

## 1. OVERVIEW

### 1.1. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    UMLS MAPPING PIPELINE                        │
│                   (Complete 6-Stage Process)                    │
└─────────────────────────────────────────────────────────────────┘

INPUT: Knowledge Graph (kg_clean.txt)
  │
  ├─> STAGE 0: UMLS Database Loading (ONE-TIME)
  │   ├── Parse MRCONSO.RRF (13M+ lines)
  │   ├── Parse MRSTY.RRF (7M+ lines)
  │   ├── Parse MRDEF.RRF (200K+ lines)
  │   └── Output: umls_concepts.pkl, umls_aliases.pkl
  │
  ├─> STAGE 1: Preprocessing
  │   ├── Extract entities from KG
  │   ├── Build synonym clusters
  │   ├── Normalize text
  │   └── Output: entities.txt, clusters.json, normalized.json
  │
  ├─> STAGE 2 SETUP: SapBERT + TF-IDF (ONE-TIME)
  │   ├── Encode 4M+ UMLS concepts with SapBERT
  │   ├── Build FAISS index
  │   ├── Build TF-IDF vectorizer
  │   └── Output: embeddings.pkl, faiss.index, tfidf.pkl
  │
  ├─> STAGE 2: Candidate Generation
  │   ├── Encode entities with SapBERT
  │   ├── Search FAISS + TF-IDF
  │   ├── Ensemble fusion (RRF)
  │   └── Output: 128 candidates per entity
  │
  ├─> STAGE 3: Cluster Aggregation
  │   ├── Aggregate across synonym clusters
  │   ├── Boost by cluster consensus
  │   └── Output: 64 candidates per entity
  │
  ├─> STAGE 4: Hard Negative Filtering
  │   ├── Identify confusable candidates
  │   ├── Apply penalties
  │   └── Output: 32 candidates per entity
  │
  ├─> STAGE 5: Cross-Encoder Reranking
  │   ├── Re-score with cross-encoder
  │   ├── Final ranking
  │   └── Output: Ranked candidates
  │
  └─> STAGE 6: Final Output
      ├── Compute confidence scores
      ├── Classify tiers (high/medium/low)
      └── Output: final_umls_mappings.json

OUTPUT: UMLS Mappings with Confidence Scores
```

### 1.2. Pipeline Stages Summary

| Stage | Name | Input | Output | Runtime | Frequency |
|-------|------|-------|--------|---------|-----------|
| **0** | UMLS Loading | UMLS RRF files | UMLS concepts DB | 30-45 min | One-time per UMLS version |
| **1** | Preprocessing | kg_clean.txt | Entities, clusters | 5-10 min | Per dataset |
| **2 Setup** | SapBERT + TF-IDF | UMLS concepts | Embeddings, indexes | 2-3 hrs (GPU) | One-time per UMLS version |
| **2** | Candidate Gen | Entities | 128 candidates/entity | 15-30 min | Per dataset |
| **3** | Aggregation | Candidates | 64 candidates/entity | 5-10 min | Per dataset |
| **4** | Hard Neg Filter | Candidates | 32 candidates/entity | 2-5 min | Per dataset |
| **5** | Reranking | Candidates | Ranked candidates | 2-5 min | Per dataset |
| **6** | Final Output | Candidates | Final mappings | 1-2 min | Per dataset |

**Total Time:**
- **First Run:** 3-5 hours (includes one-time setup)
- **Subsequent Runs:** 30-60 minutes (cached)
- **With Optimization:** 15-30 minutes (FP16 + IVF-PQ)

---

## 2. PREREQUISITES

### 2.1. System Requirements

#### Minimum Requirements:
```
CPU: 8 cores
RAM: 32 GB
GPU: NVIDIA GPU with 8+ GB VRAM (for SapBERT encoding)
      CPU-only mode available but 10x slower
Disk: 50 GB free space
OS: Linux (Ubuntu 18.04+), macOS, Windows 10+
```

#### Recommended Requirements:
```
CPU: 16+ cores
RAM: 64+ GB
GPU: NVIDIA V100/A100 with 16+ GB VRAM
      Or multiple GPUs for faster processing
Disk: 100 GB SSD
OS: Linux (Ubuntu 20.04+)
```

### 2.2. Software Dependencies

#### Python Version:
```bash
Python 3.8 or higher (3.9 recommended)
```

#### Required Libraries:
```bash
# Core dependencies
torch>=1.10.0
transformers>=4.20.0
faiss-cpu>=1.7.0  # or faiss-gpu for GPU support
scikit-learn>=1.0.0
numpy>=1.21.0
scipy>=1.7.0
tqdm>=4.62.0
pyyaml>=6.0

# Optional (for optimization)
faiss-gpu>=1.7.0  # GPU-accelerated FAISS
```

### 2.3. UMLS License

**IMPORTANT:** UMLS requires a license from NLM.

1. Register at: https://uts.nlm.nih.gov/uts/signup-login
2. Request UMLS license
3. Download UMLS Metathesaurus (Full Release)
4. Extract to data directory

---

## 3. INSTALLATION

### 3.1. Clone Repository

```bash
git clone <repository-url>
cd GFM
```

### 3.2. Create Virtual Environment

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n umls-mapping python=3.9
conda activate umls-mapping
```

### 3.3. Install Dependencies

```bash
# Install PyTorch (adjust for your CUDA version)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install -r requirements.txt

# Install FAISS
# For GPU (recommended):
pip install faiss-gpu

# For CPU only:
pip install faiss-cpu
```

### 3.4. Verify Installation

```bash
python3 -c "
import torch
import transformers
import faiss
import sklearn
print('✓ All dependencies installed')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'FAISS GPU support: {hasattr(faiss, \"StandardGpuResources\")}')
"
```

Expected output:
```
✓ All dependencies installed
PyTorch version: 2.0.0
CUDA available: True
FAISS GPU support: True
```

---

## 4. DATA PREPARATION

### 4.1. Directory Structure

Create the following directory structure:

```bash
GFM/
├── data/
│   ├── umls/
│   │   └── 2024AB/
│   │       └── META/
│   │           ├── MRCONSO.RRF
│   │           ├── MRSTY.RRF
│   │           └── MRDEF.RRF
│   └── kg_clean.txt          # Your input knowledge graph
├── outputs/                  # Will be created automatically
├── config/
│   └── umls_mapping.yaml
├── scripts/
└── gfmrag/
```

### 4.2. Download UMLS Data

```bash
# 1. Download UMLS from https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html
# 2. Extract the archive
# 3. Copy META directory to data/umls/2024AB/

mkdir -p data/umls/2024AB
unzip umls-2024AB-full.zip
cp -r 2024AB-full/META data/umls/2024AB/

# Verify files exist
ls -lh data/umls/2024AB/META/MRCONSO.RRF
ls -lh data/umls/2024AB/META/MRSTY.RRF
ls -lh data/umls/2024AB/META/MRDEF.RRF
```

Expected sizes:
```
MRCONSO.RRF: ~7 GB
MRSTY.RRF:   ~350 MB
MRDEF.RRF:   ~100 MB
```

### 4.3. Prepare Input Knowledge Graph

Your input knowledge graph should be a text file with one triple per line:

```bash
# Format: subject | predicate | object
# Example content of kg_clean.txt:

diabetes | causes | hyperglycemia
type 2 diabetes mellitus | is_a | diabetes
insulin resistance | associated_with | diabetes
metformin | treats | type 2 diabetes
glucose | elevated_in | diabetes
```

Place your KG file at:
```bash
data/kg_clean.txt
```

---

## 5. CONFIGURATION

### 5.1. Create Configuration File

```bash
# Copy template
cp config/umls_mapping.yaml config/my_project.yaml
```

### 5.2. Edit Configuration

Edit `config/my_project.yaml`:

```yaml
# =============================================================================
# INPUT/OUTPUT PATHS
# =============================================================================

# Input knowledge graph
kg_clean_path: "./data/kg_clean.txt"

# UMLS data directory
umls_data_dir: "./data/umls/2024AB/META"

# UMLS RRF files
mrconso_path: "./data/umls/2024AB/META/MRCONSO.RRF"
mrsty_path: "./data/umls/2024AB/META/MRSTY.RRF"
mrdef_path: "./data/umls/2024AB/META/MRDEF.RRF"

# Output directory
output_root: "./outputs"

# UMLS cache directory
umls_cache_dir: "./outputs/umls_cache"

# =============================================================================
# STAGE PARAMETERS
# =============================================================================

# Stage 0: UMLS Loading
umls_language: "ENG"

# Stage 2: Candidate Generation
sapbert_model: "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
sapbert_batch_size: 256        # Increase to 2048 for optimization
sapbert_device: "cuda"         # or "cpu"
sapbert_top_k: 64

tfidf_ngram_range: [3, 3]
ensemble_final_k: 128

# Stage 3: Cluster Aggregation
cluster_output_k: 64

# Stage 4: Hard Negative Filtering
hard_neg_similarity_threshold: 0.7
hard_neg_output_k: 32

# Stage 5: Cross-Encoder
cross_encoder_model: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
cross_encoder_device: "cuda"

# Stage 6: Confidence
confidence_high_threshold: 0.75
propagation_min_agreement: 0.8

# =============================================================================
# GENERAL SETTINGS
# =============================================================================

num_processes: 10
force_recompute: false
save_intermediate: true
device: "cuda"
```

### 5.3. Validate Configuration

```bash
python3 -c "
from gfmrag.umls_mapping import load_config
config, runtime = load_config('config/my_project.yaml')
print('✓ Configuration valid')
print(f'  Output: {config.output_root}')
print(f'  Device: {config.device}')
"
```

---

## 6. PIPELINE EXECUTION

### 6.1. Option A: Run Complete Pipeline (Recommended)

```bash
# Run entire pipeline with one command
python scripts/run_umls_mapping.py --config config/my_project.yaml
```

This will automatically:
1. Run all stages in order
2. Track progress
3. Handle errors
4. Save outputs
5. Generate validation report

### 6.2. Option B: Run Individual Stages

If you prefer to run stages individually:

#### Stage 0: UMLS Loading (ONE-TIME)

```bash
python scripts/run_umls_mapping.py \
    --config config/my_project.yaml \
    --stages stage0_umls_loading
```

**Expected Output:**
```
outputs/umls_cache/
├── umls_concepts.pkl     (~500 MB - 4.2M concepts)
├── umls_aliases.pkl      (~800 MB - 10M+ aliases)
└── umls_stats.json       (~5 KB - statistics)
```

**Validation:**
```bash
python3 -c "
import pickle
with open('outputs/umls_cache/umls_concepts.pkl', 'rb') as f:
    concepts = pickle.load(f)
print(f'✓ Loaded {len(concepts):,} UMLS concepts')
"
```

#### Stage 1: Preprocessing

```bash
python scripts/run_umls_mapping.py \
    --config config/my_project.yaml \
    --stages stage1_preprocessing
```

**Expected Output:**
```
outputs/stage31_preprocessing/
├── entities.txt              (All entities, sorted)
├── synonym_clusters.json     (Synonym groupings)
└── normalized_entities.json  (Normalized forms)
```

**Validation:**
```bash
python scripts/validate_stage1.py
```

#### Stage 2 Setup: SapBERT + TF-IDF (ONE-TIME)

```bash
# SapBERT encoding (2-3 hours GPU, 4-6 hours CPU)
python scripts/run_umls_mapping.py \
    --config config/my_project.yaml \
    --stages stage2_setup_sapbert

# TF-IDF building (10-15 minutes)
python scripts/run_umls_mapping.py \
    --config config/my_project.yaml \
    --stages stage2_setup_tfidf
```

**For Optimized Performance (3-6x faster):**
```bash
# Use optimized script
python scripts/task_2_1_sapbert_setup_optimized.py
python scripts/task_2_2_tfidf_setup.py

# Optional: Build IVF-PQ index for 10-50x faster queries
python scripts/build_faiss_ivfpq.py
```

**Expected Output:**
```
outputs/
├── umls_embeddings.pkl      (~12 GB - 4.2M embeddings)
├── umls_faiss.index         (~12 GB - FAISS index)
├── umls_cui_order.pkl       (~10 MB - CUI order)
├── tfidf_vectorizer.pkl     (~1 MB - vectorizer)
├── tfidf_matrix.pkl         (~500 MB - TF-IDF matrix)
├── alias_to_cuis.pkl        (~50 MB - reverse index)
└── all_aliases.pkl          (~50 MB - all aliases)
```

**Validation:**
```bash
python scripts/validate_stage2_setup.py
```

#### Stage 2: Candidate Generation

```bash
python scripts/run_umls_mapping.py \
    --config config/my_project.yaml \
    --stages stage2_candidate_generation
```

**Expected Output:**
```
outputs/stage2_candidates.json  (~100-500 MB depending on dataset)
```

**Format:**
```json
{
  "diabetes": [
    {
      "cui": "C0011849",
      "rrf_score": 0.95,
      "methods": ["sapbert", "tfidf"],
      "avg_score": 0.92,
      "preferred_name": "Diabetes Mellitus",
      "semantic_types": ["T047"]
    },
    ...
  ]
}
```

#### Stage 3: Cluster Aggregation

```bash
python scripts/run_umls_mapping.py \
    --config config/my_project.yaml \
    --stages stage3_cluster_aggregation
```

**Note:** Output is intermediate, passed to Stage 4

#### Stage 4: Hard Negative Filtering

```bash
python scripts/run_umls_mapping.py \
    --config config/my_project.yaml \
    --stages stage4_hard_negative_filtering
```

**Expected Output:**
```
outputs/stage4_filtered.json  (~50-200 MB)
```

#### Stage 5: Cross-Encoder Reranking

```bash
python scripts/run_umls_mapping.py \
    --config config/my_project.yaml \
    --stages stage5_cross_encoder_reranking
```

**Expected Output:**
```
outputs/stage5_reranked.json  (~50-200 MB)
```

#### Stage 6: Final Output

```bash
python scripts/run_umls_mapping.py \
    --config config/my_project.yaml \
    --stages stage6_final_output
```

**Expected Output:**
```
outputs/
├── final_umls_mappings.json    (Final mappings with confidence)
├── umls_mapping_triples.txt    (Triple format)
└── mapping_statistics.json     (Statistics)
```

### 6.3. Resume After Failure

If pipeline fails, resume from last successful stage:

```bash
python scripts/run_umls_mapping.py \
    --config config/my_project.yaml \
    --resume
```

### 6.4. Force Rerun

To rerun all stages (ignore cache):

```bash
python scripts/run_umls_mapping.py \
    --config config/my_project.yaml \
    --force
```

### 6.5. Check Pipeline Status

```bash
python scripts/run_umls_mapping.py \
    --config config/my_project.yaml \
    --status
```

Output:
```
Pipeline Status:
  Last run: 2025-12-30T15:30:00
  Last successful stage: stage2_candidate_generation
  Completed stages: 5
  Failed stages: 0

Completed stages:
  ✓ stage0_umls_loading
  ✓ stage1_preprocessing
  ✓ stage2_setup_sapbert
  ✓ stage2_setup_tfidf
  ✓ stage2_candidate_generation
```

---

## 7. VALIDATION

### 7.1. Stage 1 Validation

```bash
python scripts/validate_stage1.py
```

**Checks:**
- ✓ File existence (entities.txt, clusters.json, normalized.json)
- ✓ Entity count and coverage
- ✓ Cluster quality (singleton rate, size distribution)
- ✓ Normalization quality (expansion rate)
- ✓ UMLS data structure

**Expected Output:**
```
STAGE 1 VALIDATION
======================================================================

1. Checking file existence:
  ✓ entities.txt: 150.5 KB
  ✓ synonym_clusters.json: 250.3 KB
  ✓ normalized_entities.json: 300.8 KB
  ✓ umls_concepts.pkl: 512.5 MB
  ✓ umls_aliases.pkl: 823.2 MB
  ✓ umls_stats.json: 4.2 KB

2. Validating entities:
  ✓ Total entities: 12,345
  ✓ No duplicates found
  ✓ No empty entries
  ✓ Sorted correctly (case-insensitive)

3. Validating synonym clusters:
  ✓ Total clusters: 8,234
  ✓ All entities covered
  ✓ Singleton rate: 45.2% (within range)
  ✓ Average cluster size: 1.5 (within range)

4. Validating normalized entities:
  ✓ Structure correct (original, normalized, expanded)
  ✓ Expansion rate: 15.3% (within range)
  ✓ Test case passed: T2DM → type 2 diabetes mellitus

5. Validating UMLS data:
  ✓ Concepts: 4,234,567
  ✓ Aliases: 10,523,891
  ✓ Test CUI C0011860 found

6. Cross-validating with KG:
  ✓ Coverage: 98.5% (12,156 / 12,345)

======================================================================
✅ VALIDATION PASSED - ALL CHECKS SUCCESSFUL!
======================================================================
```

### 7.2. Stage 2 Setup Validation

```bash
python scripts/validate_stage2_setup.py
```

**Checks:**
- ✓ All 7 files exist (embeddings, FAISS, TF-IDF)
- ✓ File sizes reasonable
- ✓ FAISS index structure (>4M vectors)
- ✓ TF-IDF matrix dimensions (>10M × ~100K)
- ✓ Embedding dimensions (768)

### 7.3. Final Pipeline Validation

```bash
python scripts/final_validation.py
```

**Checks:**
- ✓ All output files exist
- ✓ Final mappings structure
- ✓ Statistics within expected ranges
- ✓ Confidence tier distribution

**Expected Output:**
```
FINAL PIPELINE VALIDATION
======================================================================

1. Checking output files:
  ✓ entities.txt: 150.5 KB
  ✓ synonym_clusters.json: 250.3 KB
  ✓ normalized_entities.json: 300.8 KB
  ✓ umls_concepts.pkl: 512.5 MB
  ✓ umls_embeddings.pkl: 11.8 GB
  ✓ umls_faiss.index: 12.1 GB
  ✓ umls_cui_order.pkl: 10.2 MB
  ✓ tfidf_vectorizer.pkl: 1.2 MB
  ✓ tfidf_matrix.pkl: 485.3 MB
  ✓ alias_to_cuis.pkl: 52.8 MB
  ✓ all_aliases.pkl: 48.9 MB
  ✓ stage2_candidates.json: 125.6 MB
  ✓ stage4_filtered.json: 62.8 MB
  ✓ stage5_reranked.json: 63.2 MB
  ✓ final_umls_mappings.json: 45.3 MB
  ✓ umls_mapping_triples.txt: 2.1 MB
  ✓ mapping_statistics.json: 1.8 KB

2. Validating final mappings:
  ✓ Loaded 12,345 mappings
  ✓ Field 'entity' present
  ✓ Field 'cui' present
  ✓ Field 'preferred_name' present
  ✓ Field 'confidence' present
  ✓ Field 'confidence_tier' present

3. Mapping statistics:

  Total entities mapped: 12,345

  By confidence tier:
    HIGH: 8,456 (68.5%)
    MEDIUM: 2,789 (22.6%)
    LOW: 1,100 (8.9%)

  Average confidence: 0.763
  Real UMLS coverage: 11,391 (92.3%)

  ✓ High confidence rate within target range (65-75%)

4. Validating triples:
  ✓ 12,345 triples generated
  ✓ Triple format correct

======================================================================
✅ VALIDATION PASSED - ALL CHECKS SUCCESSFUL!
🎉 PIPELINE HOÀN TẤT!
======================================================================
```

### 7.4. Quality Metrics

Check final mapping quality:

```bash
python3 -c "
import json

with open('outputs/mapping_statistics.json') as f:
    stats = json.load(f)

print('Quality Metrics:')
print(f'  High confidence: {stats[\"by_tier\"][\"high\"][\"percentage\"]}%')
print(f'  Average confidence: {stats[\"avg_confidence\"]}')
print(f'  UMLS coverage: {stats[\"real_umls_coverage_percentage\"]}%')

# Check against targets
high_pct = stats['by_tier']['high']['percentage']
if high_pct >= 65 and high_pct <= 75:
    print('  ✓ Quality targets met!')
else:
    print(f'  ⚠️  High confidence rate outside target (65-75%): {high_pct}%')
"
```

---

## 8. TROUBLESHOOTING

### 8.1. Common Issues

#### Issue 1: Out of Memory (OOM)

**Symptom:** `RuntimeError: CUDA out of memory`

**Solutions:**

```bash
# Option 1: Reduce batch size
# Edit config/my_project.yaml:
sapbert_batch_size: 128  # Instead of 256

# Option 2: Use CPU
device: "cpu"

# Option 3: Use FP16 (saves 50% memory)
# Use optimized script:
python scripts/task_2_1_sapbert_setup_optimized.py
```

#### Issue 2: UMLS Files Not Found

**Symptom:** `FileNotFoundError: MRCONSO.RRF not found`

**Solution:**

```bash
# Check file paths
ls -l data/umls/2024AB/META/MRCONSO.RRF

# If missing, verify download and extraction
# Update paths in config:
mrconso_path: "/correct/path/to/MRCONSO.RRF"
```

#### Issue 3: Slow Performance

**Symptom:** Stage 2 Setup takes >5 hours

**Solutions:**

```bash
# Use optimized scripts (3-6x faster)
python scripts/task_2_1_sapbert_setup_optimized.py

# Use GPU instead of CPU
device: "cuda"

# Use multiple GPUs if available (automatic)
```

#### Issue 4: Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'torch'`

**Solution:**

```bash
# Reinstall dependencies
pip install -r requirements.txt

# Check installation
python -c "import torch; print(torch.__version__)"
```

#### Issue 5: Low Accuracy

**Symptom:** High confidence rate < 60%

**Solutions:**

```bash
# Check data quality
python scripts/validate_stage1.py

# Increase candidate count
ensemble_final_k: 256  # Instead of 128

# Tune parameters
sapbert_top_k: 128     # Instead of 64
```

### 8.2. Debugging

Enable verbose logging:

```bash
export PYTHONUNBUFFERED=1

python scripts/run_umls_mapping.py \
    --config config/my_project.yaml \
    2>&1 | tee pipeline.log
```

Check logs:
```bash
# Pipeline logs
tail -f outputs/pipeline.log

# Check for errors
grep ERROR outputs/pipeline.log

# Check warnings
grep WARNING outputs/pipeline.log
```

### 8.3. Reset Pipeline

If pipeline is corrupted:

```bash
# Reset status
python scripts/run_umls_mapping.py \
    --config config/my_project.yaml \
    --reset

# Clear outputs
rm -rf outputs/*

# Rerun from scratch
python scripts/run_umls_mapping.py \
    --config config/my_project.yaml \
    --force
```

---

## 9. PERFORMANCE TUNING

### 9.1. GPU Optimization

Use optimized scripts for 3-6x speedup:

```bash
# Replace original with optimized
cp scripts/task_2_1_sapbert_setup_optimized.py scripts/task_2_1_sapbert_setup.py

# Features:
# - FP16 mixed precision (2x faster)
# - Large batches (2048 vs 256)
# - Multi-GPU support
# - 50% less memory
```

### 9.2. FAISS Optimization

Use IVF-PQ for 10-50x faster queries:

```bash
# After Stage 2 Setup, build approximate index
python scripts/build_faiss_ivfpq.py

# Update Stage 2 candidate generation to use it
# In stage2_generate_candidates.py:
index = faiss.read_index("./outputs/umls_faiss_ivfpq.index")
index.nprobe = 32  # Tune for speed/accuracy
```

### 9.3. Parallel Processing

Enable parallel UMLS parsing (10-15x faster):

```bash
# Edit config:
num_processes: 16  # Use all CPU cores
```

### 9.4. Caching

Reuse one-time setup across datasets:

```bash
# Setup once
python scripts/run_umls_mapping.py \
    --config config/umls_2024ab.yaml \
    --stages stage0_umls_loading stage2_setup_sapbert stage2_setup_tfidf

# Reuse for multiple datasets
cp -r outputs/umls_cache /shared/umls_cache
cp outputs/umls_*.pkl /shared/
cp outputs/*.index /shared/

# In new project config:
umls_cache_dir: "/shared/umls_cache"
```

---

## 10. PRODUCTION DEPLOYMENT

### 10.1. Batch Processing

Process multiple datasets:

```bash
#!/bin/bash
# batch_process.sh

datasets=(dataset1 dataset2 dataset3)

for ds in "${datasets[@]}"; do
    echo "Processing $ds..."

    # Create config
    sed "s|KG_PATH|./data/${ds}/kg.txt|g" \
        config/template.yaml > config/${ds}.yaml
    sed -i "s|OUTPUT|./results/${ds}|g" config/${ds}.yaml

    # Run pipeline
    python scripts/run_umls_mapping.py --config config/${ds}.yaml

    echo "✓ $ds completed"
done
```

### 10.2. Docker Deployment

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code
COPY . .

# Run pipeline
ENTRYPOINT ["python", "scripts/run_umls_mapping.py"]
CMD ["--config", "config/umls_mapping.yaml"]
```

Build and run:
```bash
docker build -t umls-mapping .

docker run --gpus all -v $(pwd)/data:/app/data \
    -v $(pwd)/outputs:/app/outputs \
    umls-mapping
```

### 10.3. Monitoring

Monitor pipeline progress:

```bash
# Terminal 1: Run pipeline
python scripts/run_umls_mapping.py --config config/my_project.yaml

# Terminal 2: Monitor logs
tail -f outputs/pipeline.log

# Terminal 3: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 4: Monitor status
watch -n 5 'python scripts/run_umls_mapping.py --config config/my_project.yaml --status'
```

### 10.4. Backup and Recovery

```bash
# Backup outputs
tar -czf outputs-$(date +%Y%m%d).tar.gz outputs/

# Backup to cloud
aws s3 cp outputs-$(date +%Y%m%d).tar.gz s3://my-bucket/backups/

# Restore
tar -xzf outputs-20251230.tar.gz
```

---

## 11. QUICK START CHECKLIST

### Before Running:

- [ ] UMLS license obtained
- [ ] UMLS data downloaded and extracted
- [ ] Dependencies installed
- [ ] Configuration file created
- [ ] Input KG file prepared
- [ ] Sufficient disk space (50+ GB)
- [ ] GPU available (recommended)

### First Run:

```bash
# 1. Verify installation
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# 2. Validate configuration
python -c "from gfmrag.umls_mapping import load_config; load_config('config/my_project.yaml'); print('✓ Config OK')"

# 3. Run pipeline
python scripts/run_umls_mapping.py --config config/my_project.yaml

# 4. Monitor progress
tail -f outputs/pipeline.log

# 5. Validate results
python scripts/final_validation.py
```

### Expected Timeline:

```
First Run (with all setup):
├── Stage 0: UMLS Loading     30-45 min
├── Stage 1: Preprocessing    5-10 min
├── Stage 2 Setup: SapBERT    2-3 hours (or 25-40 min optimized)
├── Stage 2 Setup: TF-IDF     10-15 min
├── Stage 2: Candidates       15-30 min
├── Stages 3-6: Processing    10-15 min
└── Total: 3-5 hours (or 1-2 hours optimized)

Subsequent Runs (cached):
├── Stage 1: Preprocessing    5-10 min
├── Stage 2: Candidates       15-30 min
├── Stages 3-6: Processing    10-15 min
└── Total: 30-60 min
```

---

## 12. SUPPORT AND RESOURCES

### Documentation:

- **Pipeline Overview:** `docs/UMLS_MAPPING_PIPELINE.md`
- **Optimization Guide:** `docs/OPTIMIZATION_ANALYSIS.md`
- **Quick Optimization:** `docs/QUICK_OPTIMIZATION_GUIDE.md`
- **Implementation Guide:** `docs/OPTIMIZATION_IMPLEMENTATION_GUIDE.md`

### Scripts:

- **Main Pipeline:** `scripts/run_umls_mapping.py`
- **Validation:** `scripts/validate_stage1.py`, `scripts/validate_stage2_setup.py`, `scripts/final_validation.py`
- **Optimization:** `scripts/task_2_1_sapbert_setup_optimized.py`, `scripts/build_faiss_ivfpq.py`

### Troubleshooting:

1. Check logs: `outputs/pipeline.log`
2. Validate stages: `scripts/validate_*.py`
3. Check status: `--status` flag
4. Reset if needed: `--reset` flag

---

**Questions or Issues?** Check troubleshooting section or review documentation!

**Ready to start?** Follow the Quick Start Checklist above! 🚀
