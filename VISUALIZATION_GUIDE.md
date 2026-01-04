# Visualization Guide - Stage 2 & Stage 3

## 📊 Tổng quan

Cả Stage 2 (Entity Resolution) và Stage 3 (UMLS Mapping) đều có visualization tự động.

## 🎨 Stage 2: Entity Resolution Visualization

**Location**: `tmp/entity_resolution/visualizations/`

### Plots được tạo tự động:

1. **`type_distribution.png`**
   - Bar chart: Phân phối entity types (drug, disease, procedure, etc.)
   - Shows: Số lượng entities cho mỗi type
   - Example: 200 drugs, 150 diseases, 100 procedures

2. **`tier_distribution.png`**
   - 2 charts: Bar chart + Pie chart
   - Shows: Phân phối qua 3 tiers (Tier 1: Keywords, Tier 2: SapBERT, Tier 3: LLM)
   - Useful for: Đánh giá hiệu quả của early stopping
   - Example: 60% Tier 1 (fast), 30% Tier 2 (medium), 10% Tier 3 (slow)

3. **`confidence_distribution.png`**
   - 2 charts: Histogram + Box plot
   - Shows: Phân phối confidence scores
   - Statistics: Mean, median, quartiles
   - Useful for: Đánh giá chất lượng type inference

4. **`cluster_size_distribution.png`**
   - Histogram + Statistics panel
   - Shows: Kích thước của synonym clusters
   - Statistics: Total clusters, singletons, min/max/mean size
   - Useful for: Đánh giá synonym resolution quality

5. **`embedding_similarity_heatmap.png`**
   - Heatmap (50x50 sample)
   - Shows: Cosine similarity giữa SapBERT embeddings
   - Colors: Red (high similarity) to Blue (low similarity)
   - Useful for: Visual inspection của embedding quality

### Cách sử dụng:

```bash
# Chạy Stage 2
python -m gfmrag.workflow.stage2_entity_resolution stage=2

# Visualizations tự động được tạo tại:
ls tmp/entity_resolution/visualizations/

# Xem plots:
# - Linux: xdg-open tmp/entity_resolution/visualizations/type_distribution.png
# - Mac: open tmp/entity_resolution/visualizations/type_distribution.png
# - Windows: explorer tmp\entity_resolution\visualizations\
```

---

## 🎨 Stage 3: UMLS Mapping Visualization

**Location**: `tmp/umls_mapping/visualizations/`

### Plots được tạo tự động:

1. **`candidate_reduction_funnel.png`**
   - Funnel chart: Số candidates qua các stages
   - Shows: Stage 3.2 (128) → 3.3 (64) → 3.4 (32) → 3.5 (reranked)
   - Useful for: Hiểu candidate filtering process

2. **`confidence_distribution.png`**
   - Histogram: Phân phối final confidence scores
   - Shows: Distribution across High/Medium/Low tiers
   - Example: 70% high confidence, 20% medium, 10% low

3. **`score_progression.png`**
   - Line chart: Score evolution qua các stages
   - Shows: Cách scores thay đổi từ Stage 3.2 → 3.6
   - Useful for: Đánh giá impact của mỗi stage

4. **`stage_timing.png`**
   - Bar chart: Thời gian chạy của mỗi stage
   - Shows: Stage 3.0 (7s), 3.1 (0.01s), 3.2 (2min), etc.
   - Useful for: Tìm bottlenecks

5. **`cluster_statistics.png`**
   - Histogram: Cluster size distribution
   - Shows: Số synonym cluster theo kích thước
   - Statistics: Singletons, average cluster size

6. **`metric_heatmap.png`**
   - Heatmap: All metrics across all stages
   - Shows: Overview toàn bộ pipeline metrics
   - Useful for: Quick health check

7. **`quality_metrics.png`**
   - Multi-panel: Precision, recall, confidence calibration
   - Shows: Quality metrics if gold standard available
   - Useful for: Evaluation against ground truth

### Cách sử dụng:

```bash
# Chạy Stage 3
python -m gfmrag.workflow.stage3_umls_mapping

# Visualizations tự động được tạo tại:
ls tmp/umls_mapping/visualizations/

# Xem plots:
# - Linux: xdg-open tmp/umls_mapping/visualizations/confidence_distribution.png
# - Mac: open tmp/umls_mapping/visualizations/
```

---

## 📦 Cài đặt Dependencies

```bash
# Required cho visualization:
pip install matplotlib seaborn

# Hoặc với conda:
conda install matplotlib seaborn
```

**Lưu ý:** Nếu không có matplotlib/seaborn, pipeline vẫn chạy bình thường nhưng skip visualization.

---

## 🔍 Interpretation Guide

### Stage 2 Visualizations

**Type Distribution:**
- Balanced distribution → Good entity diversity
- One type dominates → Check if dataset is specialized

**Tier Distribution:**
- High Tier 1% → Keywords working well (fast)
- High Tier 3% → Many hard cases (slow, need LLM)
- Target: 60-70% Tier 1+2

**Confidence Distribution:**
- Mean > 0.7 → Good type inference
- Many low confidence → Review keywords/examples
- Bimodal distribution → Clear high/low confidence cases

**Cluster Size:**
- Many singletons (>80%) → Good, most entities unique
- Large clusters (>10 entities) → Check for over-clustering

**Embedding Similarity:**
- Block diagonal pattern → Good entity grouping
- Random pattern → Check embedding quality

### Stage 3 Visualizations

**Candidate Funnel:**
- Smooth reduction → Pipeline working as designed
- Drastic cuts → Check if thresholds too aggressive

**Confidence Distribution:**
- 60-80% high confidence → Excellent
- >30% low confidence → Review candidate generation

**Score Progression:**
- Scores improve through stages → Good refinement
- Scores decrease → Check stage logic

**Stage Timing:**
- Stage 3.2 dominant → Expected (SapBERT encoding)
- Stage 3.5 slow → Check cross-encoder batch size

---

## 🎯 Best Practices

### When to Check Visualizations:

1. **After first run** - Baseline understanding
2. **After config changes** - Verify impact
3. **For debugging** - Identify issues
4. **For reporting** - Show results to stakeholders

### What to Look For:

**Stage 2:**
- Type distribution matches domain expectations
- Confidence distribution skewed towards high
- Cluster sizes mostly singletons + few multi-entity clusters

**Stage 3:**
- Most entities have high confidence mappings (≥0.75)
- Candidate funnel reduces smoothly
- Stage timing acceptable for your use case

### Troubleshooting:

**Low confidence in Stage 2:**
- Add more keywords to Tier 1
- Add more labeled examples to Tier 2
- Check LLM prompts in Tier 3

**Low confidence in Stage 3:**
- Increase candidate pool (ensemble.final_k)
- Check if entities are in UMLS
- Review hard negative filtering

---

## 📊 Output Format

All plots are saved as:
- **Format:** PNG
- **DPI:** 300 (high quality)
- **Size:** Optimized for reports/presentations

You can include these plots directly in:
- Research papers
- Technical reports
- Presentations
- Documentation

---

## 🔗 Related Files

**Stage 2:**
- Code: `gfmrag/workflow/stage2_visualization.py`
- Pipeline: `gfmrag/workflow/stage2_entity_resolution.py`

**Stage 3:**
- Code: `gfmrag/umls_mapping/visualization.py`
- Pipeline: `gfmrag/workflow/stage3_umls_mapping.py`

**Metrics:**
- Stage 2: `tmp/entity_resolution/stage*.json`
- Stage 3: `tmp/umls_mapping/pipeline_metrics.json`

---

**Last Updated:** 2026-01-04
**Version:** 1.0.0
