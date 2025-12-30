# UMLS Mapping Pipeline - Automated Workflow

**Complete end-to-end UMLS entity mapping với workflow tự động**

## 🎯 Overview

Pipeline tự động hóa hoàn toàn quá trình mapping biomedical entities sang UMLS CUIs qua 6 stages với:

✅ **Orchestration tự động** - Chạy toàn bộ pipeline một lệnh duy nhất
✅ **Resume capability** - Tiếp tục từ stage bị fail
✅ **Progress tracking** - Theo dõi status từng stage
✅ **Error handling** - Xử lý lỗi và recovery tự động
✅ **Config management** - YAML config dễ customize
✅ **Logging tập trung** - Tất cả logs ở một chỗ

---

## 📁 Architecture

```
┌─────────────────────────────────────────────────┐
│         CLI: run_umls_mapping.py                │
│  python scripts/run_umls_mapping.py --config    │
│  Options: --stages, --resume, --force, --status│
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│   Pipeline Orchestrator (UMLSMappingPipeline)   │
│  - Auto stage execution                         │
│  - Status tracking (.pipeline_status.json)      │
│  - Resume from failures                         │
│  - Unified logging (pipeline.log)               │
└─────────────────────────────────────────────────┘
                       ↓
┌──────────┬──────────┬──────────┬─────────────────┐
│ Stage 0  │ Stage 1  │ Stage 2  │ Stages 3-6     │
│ UMLS Load│ Preproc  │ CandGen  │ Agg→Rerank→Out │
└──────────┴──────────┴──────────┴─────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│        Configuration (umls_mapping.yaml)        │
│  Paths, models, parameters - all customizable  │
└─────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Setup Configuration

```bash
# Copy default config
cp config/umls_mapping.yaml config/my_project.yaml

# Edit config (customize paths, parameters)
vim config/my_project.yaml
```

### 2. Run Complete Pipeline

```bash
# ONE command to run entire pipeline!
python scripts/run_umls_mapping.py --config config/my_project.yaml
```

That's it! Pipeline sẽ tự động:
- Load UMLS database
- Extract và normalize entities
- Setup SapBERT + TF-IDF (lần đầu)
- Generate candidates
- Aggregate, filter, rerank
- Output final mappings

---

## 📖 Usage Guide

### Basic Usage

```bash
# Run complete pipeline
python scripts/run_umls_mapping.py --config config/umls_mapping.yaml

# Check pipeline status
python scripts/run_umls_mapping.py --config config/umls_mapping.yaml --status

# List available stages
python scripts/run_umls_mapping.py --list-stages
```

### Advanced Usage

```bash
# Run specific stages only
python scripts/run_umls_mapping.py --config config.yaml \
    --stages stage2_candidate_generation stage3_cluster_aggregation

# Resume from last successful stage (if pipeline failed)
python scripts/run_umls_mapping.py --config config.yaml --resume

# Force rerun all stages (ignore cache)
python scripts/run_umls_mapping.py --config config.yaml --force

# Reset pipeline status
python scripts/run_umls_mapping.py --config config.yaml --reset
```

### Python API Usage

```python
from gfmrag.umls_mapping import load_config, UMLSMappingPipeline

# Load config
config, runtime_options = load_config('config/umls_mapping.yaml')

# Create pipeline
pipeline = UMLSMappingPipeline(config)

# Run complete pipeline
success = pipeline.run()

# Run specific stages
success = pipeline.run(stages=['stage2_candidate_generation'])

# Resume from failure
success = pipeline.run(resume=True)

# Check status
status = pipeline.get_status()
print(status['completed_stages'])
print(status['last_successful_stage'])

# Reset
pipeline.reset()
```

---

## ⚙️ Configuration

### Config File Structure

```yaml
# config/umls_mapping.yaml

# Input/Output Paths
kg_clean_path: "./data/kg_clean.txt"
umls_data_dir: "./data/umls/2024AB/META"
output_root: "./outputs"

# UMLS Files
mrconso_path: "./data/umls/2024AB/META/MRCONSO.RRF"
mrsty_path: "./data/umls/2024AB/META/MRSTY.RRF"
mrdef_path: "./data/umls/2024AB/META/MRDEF.RRF"

# Stage Parameters
sapbert_model: "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
sapbert_top_k: 64
ensemble_final_k: 128
cluster_output_k: 64
hard_neg_output_k: 32

# Runtime
device: "cuda"  # or "cpu"
num_processes: 10
save_intermediate: true
```

### Config Fields

| Field | Default | Description |
|-------|---------|-------------|
| `kg_clean_path` | Required | Input knowledge graph file |
| `output_root` | Required | Output directory for all results |
| `umls_data_dir` | Required | UMLS data directory |
| `sapbert_model` | SapBERT-PubMedBERT | Model for semantic encoding |
| `ensemble_final_k` | 128 | Candidates after ensemble |
| `cluster_output_k` | 64 | Candidates after aggregation |
| `hard_neg_output_k` | 32 | Candidates after filtering |
| `device` | "cuda" | Device for models (cuda/cpu) |

See `config/umls_mapping.yaml` for all options.

---

## 📊 Pipeline Stages

### Stage 0: UMLS Database Loading (ONE-TIME)
**Runtime:** 30-45 minutes
**Inputs:** MRCONSO.RRF, MRSTY.RRF, MRDEF.RRF
**Outputs:** umls_concepts.pkl, umls_aliases.pkl, umls_stats.json

### Stage 1: Preprocessing
**Runtime:** 5-10 minutes
**Inputs:** kg_clean.txt
**Outputs:** entities.txt, synonym_clusters.json, normalized_entities.json

### Stage 2 Setup: SapBERT + TF-IDF (ONE-TIME)
**Runtime:** 2-3 hours (GPU)
**Outputs:**
- umls_embeddings.pkl (~12 GB)
- umls_faiss.index (~12 GB)
- tfidf_vectorizer.pkl
- tfidf_matrix.pkl (~500 MB)

### Stage 2: Candidate Generation
**Runtime:** 15-30 minutes
**Outputs:** stage2_candidates.json (128 candidates per entity)

### Stage 3: Cluster Aggregation
**Runtime:** 5-10 minutes
**Outputs:** In-memory (64 candidates per entity)

### Stage 4: Hard Negative Filtering
**Runtime:** 2-5 minutes
**Outputs:** stage4_filtered.json (32 candidates per entity)

### Stage 5: Cross-Encoder Reranking
**Runtime:** 2-5 minutes
**Outputs:** stage5_reranked.json

### Stage 6: Final Output
**Runtime:** 1-2 minutes
**Outputs:**
- final_umls_mappings.json
- umls_mapping_triples.txt
- mapping_statistics.json

---

## 🔄 Resume & Error Handling

### Automatic Resume

Pipeline tự động track status trong `.pipeline_status.json`:

```json
{
  "completed_stages": [
    "stage0_umls_loading",
    "stage1_preprocessing",
    "stage2_setup_sapbert"
  ],
  "failed_stages": ["stage2_setup_tfidf"],
  "last_run": "2025-12-30T10:30:45",
  "last_successful_stage": "stage2_setup_sapbert"
}
```

Nếu pipeline fail:

```bash
# Simply resume with --resume
python scripts/run_umls_mapping.py --config config.yaml --resume
```

Pipeline sẽ skip các stages đã completed và tiếp tục từ stage tiếp theo!

### Manual Stage Control

```bash
# Run only failed stage
python scripts/run_umls_mapping.py --config config.yaml \
    --stages stage2_setup_tfidf

# Force rerun specific stages
python scripts/run_umls_mapping.py --config config.yaml \
    --stages stage2_candidate_generation --force
```

---

## 📝 Logging

### Log Files

All logs saved to `{output_root}/pipeline.log`:

```
2025-12-30 10:30:00 - INFO - UMLS Mapping Pipeline Initialized
2025-12-30 10:30:00 - INFO - Output directory: ./outputs
2025-12-30 10:30:05 - INFO - Running: stage0_umls_loading
2025-12-30 10:45:12 - INFO - ✓ stage0_umls_loading completed successfully
2025-12-30 10:45:15 - INFO - Running: stage1_preprocessing
...
```

### Progress Monitoring

```bash
# Check pipeline status
python scripts/run_umls_mapping.py --config config.yaml --status

# Tail logs in real-time
tail -f outputs/pipeline.log

# Check completed stages
cat outputs/.pipeline_status.json | jq '.completed_stages'
```

---

## 🎛️ Stage Selection

### Run All Stages

```bash
python scripts/run_umls_mapping.py --config config.yaml
```

### ONE-TIME Setup Only

```bash
# Run UMLS loading + SapBERT/TF-IDF setup (once per UMLS version)
python scripts/run_umls_mapping.py --config config.yaml \
    --stages stage0_umls_loading stage2_setup_sapbert stage2_setup_tfidf
```

### Per-Dataset Processing

```bash
# Run entity extraction + candidate generation + final output
python scripts/run_umls_mapping.py --config config.yaml \
    --stages stage1_preprocessing stage2_candidate_generation \
             stage3_cluster_aggregation stage4_hard_negative_filtering \
             stage5_cross_encoder_reranking stage6_final_output
```

### Partial Rerun

```bash
# Rerun from stage 3 onwards (e.g., to tune parameters)
python scripts/run_umls_mapping.py --config config.yaml \
    --stages stage3_cluster_aggregation stage4_hard_negative_filtering \
             stage5_cross_encoder_reranking stage6_final_output --force
```

---

## 🎯 Best Practices

### 1. ONE-TIME vs PER-DATASET Stages

**ONE-TIME (run once, reuse forever):**
- `stage0_umls_loading` - Parse UMLS database
- `stage2_setup_sapbert` - Build SapBERT embeddings + FAISS
- `stage2_setup_tfidf` - Build TF-IDF vectorizer

**PER-DATASET (run for each dataset):**
- `stage1_preprocessing` - Extract entities from KG
- `stage2_candidate_generation` - Generate candidates
- `stage3_cluster_aggregation` - Aggregate by clusters
- `stage4_hard_negative_filtering` - Filter hard negatives
- `stage5_cross_encoder_reranking` - Rerank
- `stage6_final_output` - Final mappings

### 2. Config Management

```bash
# Separate configs for different projects
config/
├── umls_mapping.yaml          # Default template
├── project_A.yaml             # Project A
├── project_B.yaml             # Project B
└── production.yaml            # Production settings
```

### 3. Error Recovery

1. **Check logs:** `outputs/pipeline.log`
2. **Check status:** `--status`
3. **Resume:** `--resume`
4. **Force rerun specific stage:** `--stages <stage> --force`

### 4. Performance Tuning

```yaml
# For large datasets
sapbert_batch_size: 512  # Larger batch = faster (if GPU has memory)
num_processes: 20        # More parallel processing

# For limited resources
sapbert_batch_size: 64   # Smaller batch = less memory
device: "cpu"            # Use CPU if no GPU
```

---

## 🛠️ Troubleshooting

### Pipeline Hangs

```bash
# Check if any subprocess is running
ps aux | grep python | grep umls

# Kill if needed
pkill -f run_umls_mapping
```

### Out of Memory

```yaml
# Reduce batch size in config
sapbert_batch_size: 64  # Default: 256
```

### Permission Errors

```bash
# Ensure scripts are executable
chmod +x scripts/*.py
```

### Config Errors

```bash
# Validate config
python -c "from gfmrag.umls_mapping import load_config; load_config('config/umls_mapping.yaml')"
```

---

## 📈 Monitoring Progress

### Real-time Progress

```bash
# Terminal 1: Run pipeline
python scripts/run_umls_mapping.py --config config.yaml

# Terminal 2: Monitor logs
tail -f outputs/pipeline.log

# Terminal 3: Check status
watch -n 5 'python scripts/run_umls_mapping.py --config config.yaml --status'
```

### Check Outputs

```bash
# List output files
ls -lh outputs/

# Check intermediate results
ls -lh outputs/stage*/

# View statistics
cat outputs/mapping_statistics.json | jq
```

---

## 🔗 Integration

### With Existing Workflows

```python
# In your workflow
from gfmrag.umls_mapping import load_config, UMLSMappingPipeline

def run_entity_mapping(kg_path, output_dir):
    # Create config
    config_dict = {
        'kg_clean_path': kg_path,
        'output_root': output_dir,
        # ... other settings
    }

    # Save config
    import yaml
    with open('temp_config.yaml', 'w') as f:
        yaml.dump(config_dict, f)

    # Load and run
    config, _ = load_config('temp_config.yaml')
    pipeline = UMLSMappingPipeline(config)
    success = pipeline.run()

    return success

# Use it
run_entity_mapping('./data/my_kg.txt', './results')
```

### Batch Processing

```bash
#!/bin/bash
# Process multiple datasets

for dataset in dataset1 dataset2 dataset3; do
    echo "Processing $dataset..."

    # Create config for this dataset
    cp config/template.yaml config/${dataset}.yaml
    sed -i "s|KG_PATH|./data/${dataset}/kg.txt|g" config/${dataset}.yaml
    sed -i "s|OUTPUT|./results/${dataset}|g" config/${dataset}.yaml

    # Run pipeline
    python scripts/run_umls_mapping.py --config config/${dataset}.yaml

    if [ $? -eq 0 ]; then
        echo "✓ $dataset completed"
    else
        echo "✗ $dataset failed"
    fi
done
```

---

## 🎓 Examples

See `examples/` directory for complete examples:
- `examples/basic_usage.py` - Basic pipeline usage
- `examples/custom_config.py` - Custom configuration
- `examples/batch_processing.sh` - Batch processing script
- `examples/api_integration.py` - API integration

---

## 📚 Related Documentation

- [STAGE2_SETUP_README.md](STAGE2_SETUP_README.md) - SapBERT + TF-IDF setup details
- [STAGE3_METRICS_GUIDE.md](STAGE3_METRICS_GUIDE.md) - Metrics tracking
- [API Reference](API.md) - Complete API documentation

---

## 💡 Tips

1. **Use --resume frequently** - Pipeline can fail for many reasons (OOM, network, etc). Always resume instead of restarting!

2. **Monitor disk space** - FAISS index + embeddings ~25 GB. Ensure enough space.

3. **GPU recommended** - Stage 2 setup 10x faster on GPU (2-3 hours vs 20-30 hours).

4. **Config version control** - Keep configs in git for reproducibility.

5. **Test on small dataset first** - Verify pipeline works before running on full dataset.

---

## ✅ Checklist

Before running pipeline:
- [ ] UMLS data downloaded và extracted
- [ ] Config file created và validated
- [ ] Enough disk space (~30 GB)
- [ ] Dependencies installed (torch, transformers, faiss, etc.)
- [ ] Input KG file exists và valid

After pipeline completes:
- [ ] Check final_validation output
- [ ] Review mapping_statistics.json
- [ ] Verify high confidence rate (65-75%)
- [ ] Check logs for warnings
- [ ] Backup results và config

---

**Questions?** Check troubleshooting section or open an issue!
