# Workflow - Simplified Pipeline Runners

Simplified entry points for running GFM-RAG pipelines with embedded configurations.

## Stage 3: UMLS Mapping Pipeline

### Quick Start

**Option 1: Direct Python execution**
```bash
python workflow/stage3umlsmapping.py
```

**Option 2: Python module execution**
```bash
python -m workflow.stage3umlsmapping
```

**Option 3: Root-level runner (Recommended)**
```bash
python run_umls_pipeline.py
```

### Features

✅ **Zero Configuration Required** - All configs embedded with sensible defaults
✅ **Auto Path Detection** - Automatically finds data and output directories
✅ **Optimized by Default** - FP16 + Large Batches (3-6x speedup)
✅ **Resume Capability** - Continue from last checkpoint
✅ **Status Tracking** - Check pipeline progress anytime

### Common Commands

```bash
# Run complete pipeline (default)
python run_umls_pipeline.py

# Resume from checkpoint
python run_umls_pipeline.py --resume

# Check status
python run_umls_pipeline.py --status

# Run specific stages
python run_umls_pipeline.py --stages stage1_preprocessing stage2_setup_sapbert

# Custom paths
python run_umls_pipeline.py --umls-dir /path/to/umls --kg-file /path/to/kg_clean.txt

# Performance tuning
python run_umls_pipeline.py --batch-size 4096  # Larger batches (needs more VRAM)
python run_umls_pipeline.py --no-amp           # Disable FP16 (slower but more compatible)

# Reset pipeline
python run_umls_pipeline.py --reset

# Force rerun everything
python run_umls_pipeline.py --force
```

### Prerequisites

Before running, ensure you have:

1. **UMLS Files** in `data/umls/`:
   - MRCONSO.RRF
   - MRSTY.RRF
   - MRDEF.RRF

2. **Knowledge Graph** at `data/kg_clean.txt`

3. **Dependencies** installed:
   ```bash
   pip install torch transformers faiss-cpu scikit-learn tqdm
   # Or for GPU: pip install faiss-gpu
   ```

### Default Configuration

The pipeline uses these optimized defaults:

```python
- SapBERT Batch Size: 2048 (8x larger, optimized)
- Mixed Precision: FP16 (3-6x faster)
- Multi-GPU: Enabled (auto-detect)
- Output Directory: ./tmp/umls_mapping
- Cache Directory: ./tmp/umls_mapping/cache
```

### Expected Runtime

With optimizations enabled (default):

```
Stage 0: UMLS Loading         → 5-10 minutes
Stage 1: Preprocessing         → 2-5 minutes
Stage 2: SapBERT Setup         → 25-40 minutes (3-6x faster!)
Stage 2: TF-IDF Setup          → 3-5 minutes
Stage 2: Generate Candidates   → 5-10 minutes
Stages 3-4: Aggregate + Filter → 3-5 minutes
Stage 5: Reranking             → 2-3 minutes
Stage 6: Final Output          → 1-2 minutes

TOTAL: ~60-90 minutes
```

### Output Files

After successful run, find results in `tmp/umls_mapping/`:

```
tmp/umls_mapping/
├── final_umls_mappings.json          # Main output
├── stage6_with_confidence.json       # With confidence scores
├── stage5_reranked.json              # After reranking
├── stage4_filtered.json              # After filtering
├── stage3_aggregated.json            # After aggregation
├── stage2_candidates.json            # Initial candidates
└── cache/
    ├── umls_concepts.pkl             # UMLS cache
    ├── umls_embeddings.pkl           # SapBERT embeddings
    └── umls_faiss.index              # FAISS index
```

### Troubleshooting

**Issue: Out of Memory (GPU)**
```bash
# Reduce batch size
python run_umls_pipeline.py --batch-size 1024

# Or disable multi-GPU
python run_umls_pipeline.py --no-multi-gpu
```

**Issue: UMLS files not found**
```bash
# Check path
ls -la data/umls/

# Or specify custom path
python run_umls_pipeline.py --umls-dir /custom/path/to/umls
```

**Issue: Pipeline stuck/crashed**
```bash
# Resume from checkpoint
python run_umls_pipeline.py --resume

# Check status first
python run_umls_pipeline.py --status
```

**Issue: Want to start fresh**
```bash
# Reset and rerun
python run_umls_pipeline.py --reset
python run_umls_pipeline.py --force
```

### Validation

After pipeline completes, validate results:

```bash
# Final validation
python scripts/final_validation.py

# Stage 1 validation
python scripts/validate_stage1.py

# Stage 2 setup validation
python scripts/validate_stage2_setup.py
```

### Documentation

For detailed documentation, see:

- `docs/DEPLOYMENT_GUIDE.md` - Complete deployment guide
- `docs/QUICK_OPTIMIZATION_GUIDE.md` - Optimization guide
- `docs/UMLS_MAPPING_PIPELINE.md` - Pipeline architecture
- `docs/STAGE3_METRICS_GUIDE.md` - Metrics documentation

### Support

For issues or questions:
1. Check troubleshooting section in `docs/DEPLOYMENT_GUIDE.md`
2. Review pipeline logs in `tmp/umls_mapping/pipeline.log`
3. Use `--status` to check pipeline state
