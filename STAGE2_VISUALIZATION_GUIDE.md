# Stage 2 Entity Resolution - Visualization Guide

## Overview

This guide shows how to generate visualization plots from Stage 2 Entity Resolution results **without re-running the entire pipeline**.

## Prerequisites

Install visualization libraries:
```bash
pip install matplotlib seaborn
```

## Quick Start

### Option 1: Using the standalone script (Recommended)

```bash
python run_stage2_visualization.py [output_dir]
```

**Example:**
```bash
# Use default directory (tmp/entity_resolution)
python run_stage2_visualization.py

# Or specify custom directory
python run_stage2_visualization.py path/to/your/output
```

### Option 2: Using Python directly

```python
from pathlib import Path
from gfmrag.workflow.stage2_visualization import visualize_stage2_metrics

# Specify output directory containing Stage 2 results
output_dir = Path("tmp/entity_resolution")

# Generate all plots
visualize_stage2_metrics(output_dir)
```

## Required Files

The visualization script expects these files in the output directory:

**Required for most plots:**
- `stage0_entity_types.json` - Entity type inference results
- `stage1_embeddings.npy` - SapBERT embeddings
- `stage1_entity_ids.json` - Entity ID mappings

**Optional (for cluster plots):**
- `stage1b_synonym_clusters.json` - Synonym clusters

If some files are missing, the script will skip those specific plots and generate others.

## Generated Plots

The visualization script creates the following plots in `{output_dir}/visualizations/`:

### 1. **type_distribution.png**
- Bar chart showing count of each entity type (drug, disease, gene, etc.)
- Helps understand entity type composition in your dataset

### 2. **tier_distribution.png**
- Bar chart + pie chart showing 3-tier cascading distribution
- Shows how many entities were resolved by:
  - Tier 1: Keyword matching (fast)
  - Tier 2: SapBERT kNN (medium)
  - Tier 3: LLM inference (slow but accurate)

### 3. **confidence_distribution.png**
- Histogram + box plot of confidence scores
- Shows mean/median confidence across all entities
- Helps identify low-confidence entities that may need manual review

### 4. **cluster_size_distribution.png**
- Histogram of synonym cluster sizes
- Statistics: total clusters, singletons, min/max/mean sizes
- Helps understand entity resolution quality

### 5. **embedding_similarity_heatmap.png**
- Heatmap showing cosine similarity between sample entities
- Uses first 50 entities for visualization
- Helps validate embedding quality

## Example Output Structure

```
tmp/entity_resolution/
├── stage0_entity_types.json
├── stage1_embeddings.npy
├── stage1_entity_ids.json
├── stage2_candidate_pairs.jsonl
├── stage3_scored_pairs.jsonl
├── stage4_equivalent_pairs.jsonl
├── stage5_clusters.json
├── kg_clean.txt
└── visualizations/              # Created by visualization script
    ├── type_distribution.png
    ├── tier_distribution.png
    ├── confidence_distribution.png
    ├── cluster_size_distribution.png
    └── embedding_similarity_heatmap.png
```

## Troubleshooting

### Error: "Matplotlib/Seaborn not installed"

**Solution:** Install visualization libraries
```bash
pip install matplotlib seaborn
```

### Error: "Directory not found"

**Solution:** Verify the output directory path exists and contains Stage 2 results
```bash
ls tmp/entity_resolution/
```

### Warning: "Some files are missing"

**Solution:** This is normal if you haven't run all stages. The script will generate plots for available data and skip missing ones.

### Error: "No module named 'gfmrag'"

**Solution:** Make sure you're running from the project root directory
```bash
cd /path/to/GFM
python run_stage2_visualization.py
```

## Use Cases

### 1. After Stage 2 Completion (Server Without Display)

If you run Stage 2 on a server without matplotlib installed:
```bash
# On server - run Stage 2 (no visualization)
python -m gfmrag.workflow.stage2_entity_resolution

# Download results to local machine
scp -r server:path/to/tmp/entity_resolution ./tmp/

# On local machine - install libs and visualize
pip install matplotlib seaborn
python run_stage2_visualization.py tmp/entity_resolution
```

### 2. Re-generate Plots After Analysis

After analyzing metrics, regenerate plots anytime:
```bash
python run_stage2_visualization.py
```

### 3. Custom Output Directory

If you used a custom output directory in Stage 2:
```bash
python run_stage2_visualization.py custom/output/path
```

## Integration with Stage 2 Pipeline

The visualization is automatically called at the end of Stage 2 if matplotlib/seaborn are installed:

```python
# In stage2_entity_resolution.py main()
try:
    from gfmrag.workflow.stage2_visualization import visualize_stage2_metrics
    visualize_stage2_metrics(cfg.output_dir)
except ImportError:
    logger.warning("Matplotlib not installed. Skipping visualization.")
```

## Notes

- Plots use high DPI (300) for publication quality
- Non-interactive backend (`Agg`) works on servers without display
- Large datasets: embedding heatmap samples first 50 entities only
- All plots use consistent color schemes via seaborn

## See Also

- `gfmrag/workflow/stage2_visualization.py` - Visualization implementation
- `gfmrag/workflow/stage2_entity_resolution.py` - Main Stage 2 pipeline
