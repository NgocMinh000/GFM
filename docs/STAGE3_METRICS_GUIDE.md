# Stage 3 UMLS Mapping - Metrics Guide

Comprehensive metrics tracking and evaluation for each stage of the UMLS mapping pipeline.

## Overview

The metrics system provides:
- **Real-time tracking** of progress through each stage
- **Quality control** with automatic warnings and error detection
- **Performance monitoring** with timing and throughput metrics
- **Human-readable reports** for analysis and debugging

## Output Files

After running the pipeline, you'll find:

```
tmp/umls_mapping/
├── pipeline_metrics.json      # Complete metrics in JSON format
├── pipeline_report.txt         # Human-readable summary report
├── final_umls_mappings.json   # Final mappings with confidence
├── umls_mapping_triples.txt   # KG triples to add
├── mapping_statistics.json    # Overall statistics
└── manual_review_queue.json   # Low-confidence cases
```

---

## Stage 3.0: UMLS Data Loading

**Purpose**: Load and index UMLS concepts from RRF files

### Metrics Tracked

| Metric | Description | Target Range |
|--------|-------------|--------------|
| `total_concepts` | Number of UMLS concepts loaded | ~4M for full UMLS |
| `total_unique_names` | Unique concept names indexed | ~10M+ |
| `avg_names_per_concept` | Avg synonyms per concept | 2-5 |
| `concepts_with_definitions` | Concepts with text definitions | 20-30% |
| `avg_semantic_types_per_concept` | Avg semantic types per concept | 1-2 |

### What to Monitor

✅ **Good Signs**:
- Loading completes in reasonable time (5-15 min for full UMLS)
- ~4M concepts loaded
- No errors during parsing

⚠️ **Warning Signs**:
- Very low concept count (< 100K)
- No concepts with definitions (MRDEF missing)
- Parse errors in logs

---

## Stage 3.1: Preprocessing & Entity Extraction

**Purpose**: Extract entities from KG and build synonym clusters

### Metrics Tracked

| Metric | Description | Target Range |
|--------|-------------|--------------|
| `total_entities` | Entities extracted from kg_clean.txt | Depends on corpus |
| `total_clusters` | Synonym clusters built | < total_entities |
| `singleton_clusters` | Clusters with 1 entity | 40-60% |
| `max_cluster_size` | Largest cluster size | Monitor for outliers |
| `avg_cluster_size` | Average cluster size | 1.5-3.0 |
| `median_cluster_size` | Median cluster size | 1 (many singletons) |

### What to Monitor

✅ **Good Signs**:
- Reasonable entity count for your corpus
- 40-60% singleton clusters (many entities have no synonyms)
- No extremely large clusters (>100 entities)

⚠️ **Warning Signs**:
- Very large clusters (>100) may indicate overly aggressive synonym grouping
- 0% singleton clusters means something wrong with synonym extraction
- Very low entity count compared to input file

---

## Stage 3.2: Candidate Generation

**Purpose**: Generate top-K UMLS candidates using SapBERT + TF-IDF ensemble

### Metrics Tracked

| Metric | Description | Target Range |
|--------|-------------|--------------|
| `entities_with_candidates` | Entities with ≥1 candidate | Should = input count |
| `avg_candidates_per_entity` | Avg candidates returned | ~128 (config.ensemble_final_k) |
| `entities_with_no_candidates` | Entities with 0 candidates | <5% |
| `avg_top1_score` | Avg score of top-1 candidate | 0.6-0.9 |
| `min_candidate_score` | Lowest candidate score | 0.0-0.3 |
| `max_candidate_score` | Highest candidate score | 0.8-1.0 |
| `avg_candidate_score` | Overall avg score | 0.3-0.6 |

### Warnings Generated

- **No candidates found for: `<entity>`** - Entity couldn't be matched to any UMLS concept

### What to Monitor

✅ **Good Signs**:
- <5% entities with no candidates
- High avg_top1_score (>0.6)
- Reasonable score distribution

⚠️ **Warning Signs**:
- >20% entities with no candidates → Check preprocessing/normalization
- Very low avg_top1_score (<0.3) → May need better models or data
- All scores near 1.0 → Potential bug

---

## Stage 3.3: Cluster Aggregation

**Purpose**: Aggregate candidates across synonym clusters using voting

### Metrics Tracked

| Metric | Description | Target Range |
|--------|-------------|--------------|
| `clusters_processed` | Synonym clusters aggregated | = total_clusters from Stage 3.1 |
| `avg_top1_score_after_aggregation` | Avg top-1 score after voting | Should ≥ Stage 3.2 |
| `avg_outliers_per_cluster` | Avg outlier candidates per cluster | <5 |
| `avg_cluster_support` | Avg entities agreeing on CUI | 1-3 |

### What to Monitor

✅ **Good Signs**:
- Score improves vs Stage 3.2 (consensus helps)
- Low outlier count (<20% of candidates)
- Good cluster support (>1 for multi-entity clusters)

⚠️ **Warning Signs**:
- Score drops significantly → Voting may be too aggressive
- High outlier rate (>50%) → Cluster members disagree
- Very low cluster support → Check synonym clustering

---

## Stage 3.4: Hard Negative Filtering

**Purpose**: Filter hard negatives and check semantic type consistency

### Metrics Tracked

| Metric | Description | Target Range |
|--------|-------------|--------------|
| `entities_filtered` | Entities processed | = input count |
| `avg_top1_score_after_filtering` | Avg top-1 score after filter | May drop slightly |
| `type_match_rate` | % candidates matching semantic type | 60-90% |
| `avg_hard_negative_penalty` | Avg penalty for hard negatives | <0.2 |
| `candidates_with_penalties` | Candidates penalized | 5-20% |

### What to Monitor

✅ **Good Signs**:
- High type_match_rate (>60%)
- Low avg penalty (<0.2)
- Reasonable penalty count (5-20%)

⚠️ **Warning Signs**:
- Very low type_match_rate (<30%) → Semantic type inference may be off
- High penalties everywhere (>0.5) → Too aggressive hard negative detection
- 0% penalties → Hard negative detection not working

---

## Stage 3.5: Cross-Encoder Reranking

**Purpose**: Rerank candidates using PubMedBERT cross-encoder

### Metrics Tracked

| Metric | Description | Target Range |
|--------|-------------|--------------|
| `entities_reranked` | Entities processed | = input count |
| `avg_final_score` | Final score after reranking | Should improve |
| `avg_cross_encoder_score` | Avg cross-encoder score | 0.4-0.8 |
| `avg_previous_score` | Avg score before reranking | From Stage 3.4 |
| `score_improvement` | Improvement vs previous stage | >0 |

### What to Monitor

✅ **Good Signs**:
- Positive score_improvement
- avg_cross_encoder_score in reasonable range (0.4-0.8)
- Final scores are better discriminated

⚠️ **Warning Signs**:
- Negative score_improvement → Cross-encoder may need fine-tuning
- All cross-encoder scores near 0.5 → Model not confident
- Very slow processing → Consider batch size tuning

---

## Stage 3.6: Confidence Scoring & Propagation

**Purpose**: Compute multi-factor confidence and propagate through clusters

### Metrics Tracked

| Metric | Description | Target Goal |
|--------|-------------|-------------|
| `total_mappings` | Total entities mapped | = input count |
| `high_confidence` | Mappings with confidence ≥0.75 | **Target: >60%** |
| `medium_confidence` | Mappings with 0.50 ≤ confidence <0.75 | 20-30% |
| `low_confidence` | Mappings with confidence <0.50 | <20% |
| `high_confidence_pct` | % high confidence | **>60%** |
| `medium_confidence_pct` | % medium confidence | 20-30% |
| `low_confidence_pct` | % low confidence | **<20%** |
| `propagated_count` | Mappings from propagation | 10-30% |
| `propagated_pct` | % propagated | 10-30% |
| `avg_confidence` | Mean confidence | **>0.65** |
| `median_confidence` | Median confidence | 0.6-0.8 |
| `min_confidence` | Lowest confidence | 0.0-0.4 |
| `max_confidence` | Highest confidence | 0.9-1.0 |
| `avg_score_margin` | Avg gap between top-1 and top-2 | >0.2 |
| `avg_cluster_consensus` | Avg cluster agreement | >0.7 |

### Warnings Generated

- **Low confidence mapping: `<entity>` → `<CUI>` (`<score>`)** - Mapping needs manual review

### What to Monitor

✅ **Good Signs** (Target Performance):
- **≥60% high confidence** (main quality metric)
- **<20% low confidence**
- **avg_confidence >0.65**
- avg_score_margin >0.2 (clear winner)
- avg_cluster_consensus >0.7

⚠️ **Warning Signs**:
- <40% high confidence → Pipeline may need tuning
- >30% low confidence → Many uncertain mappings
- avg_score_margin <0.1 → Hard to pick winner
- Many propagation warnings → Cluster quality issues

---

## Overall Pipeline Metrics

### Summary Statistics

Located in `pipeline_metrics.json`:

```json
{
  "summary": {
    "total_stages": 7,
    "total_duration_seconds": 3600.5,
    "total_warnings": 45,
    "total_errors": 0,
    "stage_durations": {
      "Stage 3.0: UMLS Data Loading": "120.5s",
      "Stage 3.1: Preprocessing": "15.2s",
      "Stage 3.2: Candidate Generation": "1800.3s",
      "Stage 3.3: Cluster Aggregation": "45.8s",
      "Stage 3.4: Hard Negative Filtering": "120.1s",
      "Stage 3.5: Cross-Encoder Reranking": "1400.6s",
      "Stage 3.6: Confidence & Propagation": "98.0s"
    }
  }
}
```

### Report Format

Located in `pipeline_report.txt`:

```
================================================================================
STAGE 3 UMLS MAPPING - PIPELINE REPORT
================================================================================

OVERALL SUMMARY
--------------------------------------------------------------------------------
Total Duration: 3600.50s
Total Warnings: 45
Total Errors: 0

================================================================================
Stage 3.0: UMLS Data Loading
================================================================================
Duration: 120.50s
Input Count: 0
Output Count: 4000000
Throughput: 33195.02 items/s

Metrics:
  - total_concepts: 4000000
  - total_unique_names: 12500000
  - avg_names_per_concept: 3.125
  ...
```

---

## Troubleshooting Guide

### Issue: Low High-Confidence Rate (<40%)

**Possible Causes**:
1. Poor candidate generation (check Stage 3.2 avg_top1_score)
2. Hard negative issues (check Stage 3.4 metrics)
3. Cross-encoder not working well (check Stage 3.5 score_improvement)

**Solutions**:
- Tune SapBERT/TF-IDF weights in candidate generation
- Adjust confidence thresholds
- Fine-tune cross-encoder on domain data
- Check semantic type inference rules

### Issue: Many "No Candidates Found" Warnings

**Possible Causes**:
1. Entity normalization too aggressive
2. UMLS coverage doesn't match domain
3. Preprocessing issues

**Solutions**:
- Review normalization rules in `utils.py`
- Check if entities are medical/biological
- Add domain-specific abbreviations
- Consider using subset of UMLS relevant to domain

### Issue: Very Slow Processing

**Bottlenecks**:
- Stage 3.2 (SapBERT encoding): Use GPU, adjust batch size
- Stage 3.5 (Cross-encoder): Use GPU, increase batch size
- Stage 3.0 (UMLS loading): Use cached files

**Solutions**:
- Set `device: cuda` in config
- Increase batch sizes if GPU memory allows
- Use `force_recompute: false` after first run

### Issue: High Outlier Rate in Clusters

**Possible Causes**:
1. Synonym clustering too aggressive
2. Entities in cluster are actually different
3. Low-quality candidates

**Solutions**:
- Review synonym relationships in input KG
- Adjust cluster aggregation weights
- Check Stage 3.2 candidate quality

---

## Best Practices

### 1. Monitor Key Metrics

Focus on these critical metrics:
- **Stage 3.2**: `avg_top1_score` (should be >0.6)
- **Stage 3.6**: `high_confidence_pct` (should be >60%)
- **Stage 3.6**: `low_confidence_pct` (should be <20%)
- **Overall**: `total_errors` (should be 0)

### 2. Iterative Tuning

1. Run pipeline on small sample first
2. Check metrics at each stage
3. Identify bottlenecks or quality issues
4. Adjust config parameters
5. Re-run and compare metrics

### 3. Manual Review

Always review:
- `manual_review_queue.json` for low-confidence mappings
- Warnings in `pipeline_report.txt`
- Random sample of high-confidence mappings

### 4. Save Intermediate Results

Set `save_intermediate: true` in config to enable debugging:
- `stage32_candidates.json` - Raw candidates
- `stage33_aggregated.json` - After cluster voting
- `stage34_filtered.json` - After hard negative filtering
- `stage35_reranked.json` - After cross-encoder

---

## Target Performance (Overall)

**Production Quality**:
- High Confidence: ≥60%
- Medium Confidence: 20-30%
- Low Confidence: <20%
- Average Confidence: >0.65
- Processing Time: <1 hour for 10K entities

**Research Quality** (if gold standard available):
- Top-1 Accuracy: 75-85%
- Top-5 Accuracy: 85-95%
- Mean Reciprocal Rank: >0.80

---

## References

- UMLS Documentation: https://www.nlm.nih.gov/research/umls/
- SapBERT Paper: https://arxiv.org/abs/2010.11784
- Metrics calculation: `gfmrag/umls_mapping/metrics.py`
