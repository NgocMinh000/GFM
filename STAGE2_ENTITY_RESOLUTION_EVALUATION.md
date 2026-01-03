# Stage 2: Entity Resolution Evaluation Guide

## 📊 Overview

This document provides comprehensive guidance on evaluating the quality of **Stage 2: Entity Resolution** in the GFM-RAG knowledge graph construction pipeline. Stage 2 is responsible for extracting entities and relations from documents using Open Information Extraction (OpenIE) and resolving duplicate entities through semantic similarity matching.

## 🎯 Evaluation Objectives

The evaluation framework assesses:

1. **OpenIE Extraction Quality** - How well triples are extracted from text
2. **Entity Linking Performance** - How accurately similar entities are identified
3. **Cluster Quality** - How well synonym groups are formed
4. **Graph Structure** - Overall quality of the knowledge graph
5. **Processing Efficiency** - Computational performance and throughput

## 📚 Academic References

This evaluation methodology is based on recent research in entity resolution:

### Primary References

1. **"How to Evaluate Entity Resolution Systems"** (2024)
   - arXiv:2404.05622
   - Proposes entity-centric evaluation framework
   - Introduces cluster precision/recall metrics
   - [Paper Link](https://arxiv.org/pdf/2404.05622)

2. **"Unsupervised Graph-Based Entity Resolution for Complex Entities"** (ACM TKDD)
   - Graph-based clustering algorithms for entity resolution
   - Precision, recall, and F-measure evaluation
   - [Paper Link](https://dl.acm.org/doi/10.1145/3533016)

3. **"Network metrics for assessing the quality of entity resolution"** (2020)
   - Network-based quality metrics for entity resolution
   - Cluster size distribution analysis
   - [Paper Link](https://www.researchgate.net/publication/347379650)

4. **ColBERT: Efficient and Effective Passage Search** (SIGIR 2020)
   - Late interaction model for semantic similarity
   - Token-level embeddings and MaxSim operation
   - [Paper Link](https://github.com/stanford-futuredata/ColBERT)

## 📈 Metrics Categories

### 1. OpenIE Extraction Metrics

Evaluates the quality of triple extraction from raw text.

| Metric | Description | Target | Interpretation |
|--------|-------------|--------|----------------|
| **Clean Triple Rate** | % of correctly formatted triples | ≥ 85% | Higher is better. Low values indicate OpenIE model issues. |
| **Formatting Error Rate** | % of malformed triples | ≤ 10% | Lower is better. High values suggest parsing errors. |
| **NER Error Rate** | % of triples missing NER entities | ≤ 15% | Lower is better. Indicates entity extraction quality. |
| **Unique Triple Ratio** | unique_triples / total_triples | 0.6 - 0.9 | Measures redundancy. Too low suggests duplication. |

**Interpretation:**
- **Clean Triple Rate ≥ 90%**: Excellent extraction quality
- **85% ≤ Clean Triple Rate < 90%**: Good quality, minor issues
- **70% ≤ Clean Triple Rate < 85%**: Acceptable, needs improvement
- **Clean Triple Rate < 70%**: Poor quality, investigate OpenIE model

### 2. Entity Linking Metrics

Evaluates semantic similarity matching for entity resolution.

| Metric | Description | Target | Interpretation |
|--------|-------------|--------|----------------|
| **Avg Similarity Score** | Mean similarity of matched pairs | ≥ 0.85 | Higher indicates more confident matches. |
| **Median Similarity Score** | Median similarity | ≥ 0.87 | Robust central tendency measure. |
| **Score Std Dev** | Standard deviation of scores | 0.05 - 0.10 | Lower suggests consistent matching quality. |
| **Entities Indexed** | Number of entities indexed | - | Total entities available for matching. |
| **Synonym Pairs Found** | Number of synonym pairs | - | Higher suggests good entity coverage. |

**Similarity Score Distribution:**
- **High Confidence (≥ 0.9)**: Very strong semantic match
- **Medium Confidence (0.8 - 0.9)**: Good match, reliable
- **Low Confidence (threshold - 0.8)**: Weaker match, review recommended

**Target Distribution:**
- High Confidence: ≥ 60%
- Medium Confidence: 20-30%
- Low Confidence: ≤ 15%

### 3. Cluster Analysis Metrics

Evaluates the quality of synonym clustering using Union-Find algorithm.

| Metric | Description | Target | Interpretation |
|--------|-------------|--------|----------------|
| **Number of Clusters** | Total entity clusters | - | Lower suggests better consolidation. |
| **Avg Cluster Size** | Mean entities per cluster | 2-5 | Indicates clustering effectiveness. |
| **Max Cluster Size** | Largest cluster size | < 20 | Very large clusters may indicate over-merging. |
| **Singleton Ratio** | % clusters with 1 entity | 30-50% | Some singletons are expected (unique entities). |

**Cluster Size Distribution Analysis:**
- **Size 1 (Singletons)**: 30-50% - Normal for unique entities
- **Size 2-3**: 30-40% - Most common, good synonym pairs
- **Size 4-10**: 15-25% - Reasonable synonym groups
- **Size >10**: < 5% - Review for over-merging

**Interpretation:**
- **Avg Cluster Size 2-3**: Optimal - mostly pairs and triplets
- **Avg Cluster Size 3-5**: Good - reasonable synonym groups
- **Avg Cluster Size > 5**: Review - possible over-merging
- **Avg Cluster Size < 2**: Review - possible under-merging

### 4. Coverage Metrics

Measures how well entities are linked to synonyms.

| Metric | Description | Target | Interpretation |
|--------|-------------|--------|----------------|
| **Entity Coverage** | % entities with ≥ 1 synonym | 50-70% | Higher suggests better entity linking. |
| **Entities with Synonyms** | Count of linked entities | - | Absolute number of successful links. |
| **Entities without Synonyms** | Count of singletons | - | Unique entities or missed links. |

**Interpretation:**
- **Coverage ≥ 70%**: Excellent linking performance
- **50% ≤ Coverage < 70%**: Good performance
- **30% ≤ Coverage < 50%**: Acceptable, consider tuning
- **Coverage < 30%**: Poor, check similarity threshold

### 5. Graph Structure Metrics

Evaluates the overall knowledge graph quality.

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Total Edges** | All edges in graph | Larger graphs = more information |
| **Synonymy Edges** | Edges of type "equivalent" | More synonyms = better entity resolution |
| **Relation Edges** | Non-synonym edges | More relations = richer semantic content |
| **Unique Relations** | Number of distinct relations | Higher diversity = more expressive graph |
| **Synonym Ratio** | synonymy_edges / total_edges | 0.1 - 0.3 typical range |

### 6. Efficiency Metrics

Measures computational performance.

| Metric | Description | Target |
|--------|-------------|--------|
| **Total Time** | End-to-end pipeline time | Depends on dataset size |
| **OpenIE Time** | Triple extraction time | Largest component typically |
| **Graph Creation Time** | Graph building time | - |
| **Entity Linking Time** | Similarity computation time | - |
| **Entities/Second** | Processing throughput | Higher is better |
| **Triples/Second** | Extraction throughput | Higher is better |

## 🔍 Quality Assessment Guidelines

### Overall Quality Score

Combine metrics into an overall assessment:

**Excellent (A: 90-100%)**
- Clean Triple Rate ≥ 90%
- Avg Similarity ≥ 0.88
- High Confidence ≥ 65%
- Coverage ≥ 65%
- Avg Cluster Size: 2-4

**Good (B: 80-90%)**
- Clean Triple Rate ≥ 85%
- Avg Similarity ≥ 0.85
- High Confidence ≥ 55%
- Coverage ≥ 50%
- Avg Cluster Size: 2-5

**Acceptable (C: 70-80%)**
- Clean Triple Rate ≥ 75%
- Avg Similarity ≥ 0.82
- High Confidence ≥ 45%
- Coverage ≥ 40%
- Avg Cluster Size: 1.5-6

**Needs Improvement (D/F: < 70%)**
- Clean Triple Rate < 75%
- Avg Similarity < 0.82
- High Confidence < 45%
- Coverage < 40%
- Review all components

## 📊 Visualization Guide

The evaluation automatically generates 7 visualization plots:

### 1. **Quality Dashboard** (`00_quality_dashboard.png`)

Comprehensive overview with:
- OpenIE quality score
- Similarity scores (avg, median, max)
- Entity coverage percentage
- Confidence distribution
- Cluster metrics
- Processing time breakdown
- Summary statistics box

**Use:** Quick overall health check

### 2. **OpenIE Quality** (`01_openie_quality.png`)

Shows:
- Clean triples vs. errors (bar chart)
- Error rates (formatting, NER)

**Interpretation:**
- Green bars should dominate
- Red/orange bars should be small
- Error rates should be < 10-15%

### 3. **Similarity Distribution** (`02_similarity_distribution.png`)

Shows:
- Histogram of similarity scores
- Confidence tier pie chart

**Interpretation:**
- Distribution should be right-skewed (concentrated near 1.0)
- High confidence slice should be largest in pie chart
- Vertical lines mark confidence thresholds

### 4. **Cluster Analysis** (`03_cluster_analysis.png`)

Shows:
- Cluster size distribution (bar chart)
- Cluster statistics (clusters, avg size, max size, singletons)
- Coverage pie chart

**Interpretation:**
- Most clusters should be size 2-3
- Singleton bar (red) should be present but not dominant
- Coverage pie should show ≥ 50% green

### 5. **Graph Structure** (`04_graph_structure.png`)

Shows:
- Edge type distribution
- Graph statistics summary

**Interpretation:**
- Both synonymy and relation edges should be present
- Relation edges typically outnumber synonymy edges

### 6. **Efficiency Metrics** (`05_efficiency_metrics.png`)

Shows:
- Processing time per stage
- Throughput (entities/sec, triples/sec)

**Interpretation:**
- OpenIE typically takes longest
- Compare throughput across runs for performance tracking

### 7. **Entity Statistics** (`06_entity_statistics.png`)

Shows:
- Entity counts (total, unique, phrases)
- Linking results (indexed, synonym pairs)

**Interpretation:**
- Unique counts should be significantly lower than total (deduplication)
- Synonym pairs should be substantial fraction of unique entities

## 🛠️ Troubleshooting Guide

### Problem: Low Clean Triple Rate (< 75%)

**Possible Causes:**
- OpenIE model not suited for domain
- Input text quality issues
- Parsing errors

**Solutions:**
1. Check sample extracted triples manually
2. Try different OpenIE model
3. Preprocess text (cleaning, normalization)
4. Adjust OpenIE parameters

### Problem: Low Similarity Scores (Avg < 0.82)

**Possible Causes:**
- Similarity threshold too low
- Entity linking model not suitable
- Entities very diverse (expected)

**Solutions:**
1. Increase threshold (default: 0.8)
2. Try different embedding model (ColBERT vs. DPR)
3. Check if low scores are expected for domain

### Problem: Low Coverage (< 30%)

**Possible Causes:**
- Threshold too high
- Max neighbors too low
- Entities truly unique

**Solutions:**
1. Lower threshold to 0.75-0.78
2. Increase max_sim_neighbors from 100 to 200
3. Verify entities are actually similar

### Problem: Very Large Clusters (Max Size > 30)

**Possible Causes:**
- Threshold too low
- Over-generalization

**Solutions:**
1. Increase threshold to 0.85-0.88
2. Review large clusters manually
3. Consider domain-specific filtering

### Problem: Too Many Singletons (> 70%)

**Possible Causes:**
- Threshold too high
- Poor entity matching
- Entities truly unique

**Solutions:**
1. Lower threshold slightly
2. Check if singletons are actually unique
3. Try different similarity model

## 📁 Output Files

After running Stage 2, you'll find these files in `tmp/kg_construction/<fingerprint>/`:

### Metrics Files

1. **`entity_resolution_metrics.json`**
   - Complete metrics in JSON format
   - Machine-readable for analysis
   - Structured by category

2. **`entity_resolution_details.json`**
   - Raw data:
     - All similarity scores (array)
     - Cluster sizes (array)
     - All synonym pairs with scores

3. **`entity_resolution_report.txt`**
   - Human-readable text report
   - Formatted tables
   - Summary statistics

### Visualization Files

Located in `visualizations/` subdirectory:

- `00_quality_dashboard.png` - Overall quality overview
- `01_openie_quality.png` - Extraction quality
- `02_similarity_distribution.png` - Score distribution
- `03_cluster_analysis.png` - Clustering results
- `04_graph_structure.png` - Graph statistics
- `05_efficiency_metrics.png` - Performance metrics
- `06_entity_statistics.png` - Entity counts

### Graph Files

1. **`passage_info.json`**
   - Document-level entity information
   - Clean and noisy triples per document

2. **`openie_results.jsonl`**
   - Raw OpenIE extraction results
   - One JSON object per line

## 🚀 Usage

### Running with Evaluation

The metrics and visualizations are automatically generated when running Stage 2:

```bash
python gfmrag/workflow/stage1_index_dataset.py \
    dataset=your_dataset \
    kg_constructor.force=true
```

### Viewing Results

After completion:

```bash
# View text report
cat tmp/kg_construction/<fingerprint>/entity_resolution_report.txt

# View metrics JSON
jq '.' tmp/kg_construction/<fingerprint>/entity_resolution_metrics.json

# Open visualizations
open tmp/kg_construction/<fingerprint>/visualizations/
```

### Customizing Thresholds

In your config file (`config/stage1_index_dataset.yaml`):

```yaml
kg_constructor:
  threshold: 0.80          # Similarity threshold (0.75-0.88)
  max_sim_neighbors: 100   # Max synonyms per entity (50-200)
  cosine_sim_edges: true   # Enable entity resolution

  el_model:
    _target_: gfmrag.kg_construction.entity_linking_model.ColbertELModel
    model_name_or_path: "colbert-ir/colbertv2.0"
    # Or use DPR:
    # _target_: gfmrag.kg_construction.entity_linking_model.DPRELModel
    # model_name: "BAAI/bge-large-en-v1.5"
```

### Programmatic Access

```python
from gfmrag.kg_construction import KGConstructor
from gfmrag.kg_construction.metrics import EntityResolutionMetrics
from gfmrag.kg_construction.visualization import visualize_entity_resolution

# After running KGConstructor
constructor = KGConstructor.from_config(cfg)
constructor.create_kg(data_root="data", data_name="my_dataset")

# Metrics are automatically collected in constructor.metrics
metrics = constructor.metrics.get_all_metrics()

# Generate custom visualizations
visualize_entity_resolution("tmp/kg_construction/<fingerprint>")
```

## 📖 Best Practices

### 1. Baseline Establishment

Run evaluation on a small sample first:
- Manually verify 50-100 synonym pairs
- Establish domain-specific quality targets
- Calibrate thresholds based on results

### 2. Iterative Tuning

1. Start with default threshold (0.8)
2. Review similarity distribution
3. Adjust based on high/medium/low confidence ratio
4. Re-run and compare metrics

### 3. Model Selection

- **ColBERT**: Better for long entities, contextual matching
- **DPR**: Faster, good for short entities, less memory
- **Test both** on your domain and compare

### 4. Quality vs. Coverage Trade-off

- **High threshold (0.85-0.88)**: High precision, low coverage
- **Medium threshold (0.80-0.83)**: Balanced
- **Low threshold (0.75-0.78)**: High coverage, lower precision

Choose based on downstream task requirements.

### 5. Monitoring Over Time

Track metrics across dataset updates:
```bash
# Compare metrics between runs
diff tmp/kg_construction/run1/entity_resolution_metrics.json \
     tmp/kg_construction/run2/entity_resolution_metrics.json
```

## 🔗 Integration with Stage 3

Stage 2 entity resolution feeds into Stage 3 UMLS mapping:

1. **Clean Entities** → Used as queries for UMLS linking
2. **Synonym Clusters** → Help improve UMLS candidate generation
3. **Graph Structure** → Provides context for concept disambiguation

**Quality Targets for Stage 3 Readiness:**
- Clean Triple Rate ≥ 80%
- Coverage ≥ 45%
- Avg Similarity ≥ 0.83

## 📚 References

1. **How to Evaluate Entity Resolution Systems** (2024)
   https://arxiv.org/pdf/2404.05622

2. **Unsupervised Graph-Based Entity Resolution**
   https://dl.acm.org/doi/10.1145/3533016

3. **Network metrics for entity resolution quality**
   https://www.researchgate.net/publication/347379650

4. **ColBERT: Efficient Passage Search** (SIGIR 2020)
   https://github.com/stanford-futuredata/ColBERT

5. **Papers with Code: Entity Resolution**
   https://paperswithcode.com/task/entity-resolution

6. **Entity Resolution in Knowledge Graphs (Neo4j)**
   https://neo4j.com/blog/developer/entity-resolved-knowledge-graphs/

## 💡 Tips

1. **Always review visualizations** - Numbers alone don't tell the full story
2. **Check cluster extremes** - Look at largest and smallest clusters manually
3. **Sample synonym pairs** - Verify a random sample are actually synonyms
4. **Monitor processing time** - Track efficiency for production deployments
5. **Document threshold choices** - Record rationale for parameter selection
6. **Version control configs** - Track config changes alongside metrics

## ❓ FAQ

**Q: What similarity threshold should I use?**
A: Start with 0.80. Increase to 0.83-0.85 if you see false positive synonyms. Decrease to 0.75-0.78 if coverage is too low.

**Q: How many synonym pairs is "good"?**
A: Depends on your domain. Aim for 30-60% entity coverage. Medical/scientific domains typically have higher synonym rates.

**Q: Should I use ColBERT or DPR?**
A: ColBERT for biomedical/technical domains with long entities. DPR for general domains with short entities. Test both.

**Q: My clusters are too large. What do I do?**
A: Increase threshold to 0.85+, reduce max_sim_neighbors to 50, or review the large clusters for over-merging patterns.

**Q: Why are there so many singletons?**
A: Some entities are truly unique. 30-50% singletons is normal. If > 70%, try lowering threshold.

---

**Last Updated**: 2026-01-03
**Version**: 1.0
**Maintainer**: GFM-RAG Development Team
