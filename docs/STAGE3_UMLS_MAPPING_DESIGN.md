# Stage 3: UMLS Mapping - Implementation Design

## Overview

**Purpose:** Map biomedical entities from KG to UMLS CUIs (Concept Unique Identifiers)

**Input:** `kg_clean.txt` from Stage 2 (với synonyms_of edges)

**Output:** `umls_mappings.json` (entities + CUIs + confidence scores)

**Target Accuracy:** 85-90%

---

## Architecture

### Pipeline Stages

```
STAGE 3.0: Setup & UMLS Data Loading
  ├─ Download UMLS files (MRCONSO, MRSTY, MRDEF)
  ├─ Parse and index UMLS concepts
  └─ Build search indices

STAGE 3.1: Preprocessing & Data Preparation
  ├─ Extract entities from kg_clean.txt
  ├─ Build synonym clusters from synonyms_of edges
  ├─ Normalize text
  └─ Create entity metadata

STAGE 3.2: Candidate Generation (Ensemble)
  ├─ Method A: SapBERT semantic similarity
  ├─ Method B: TF-IDF character n-grams
  ├─ Ensemble fusion (RRF)
  └─ Output: top-128 candidates per entity

STAGE 3.3: Synonym Cluster Aggregation
  ├─ Aggregate candidates across synonym clusters
  ├─ Compute cluster consensus scores
  ├─ Detect outliers
  └─ Output: top-64 refined candidates

STAGE 3.4: Hard Negative Filtering
  ├─ Identify hard negatives (similar strings, different CUIs)
  ├─ Infer semantic types from KG context
  ├─ Filter by semantic type consistency
  └─ Output: top-32 filtered candidates

STAGE 3.5: Cross-Encoder Reranking
  ├─ Load/train PubMedBERT cross-encoder
  ├─ Score each (entity, candidate) pair
  ├─ Rerank by cross-encoder scores
  └─ Output: top-1 CUI with scores

STAGE 3.6: Confidence & Propagation
  ├─ Compute multi-factor confidence scores
  ├─ Propagate CUIs through synonym clusters
  ├─ Classify by confidence tiers (high/medium/low)
  └─ Output: Final mappings with confidence
```

---

## File Structure

```
gfmrag/
├── umls_mapping/                          # NEW: UMLS mapping logic
│   ├── __init__.py
│   ├── config.py                          # UMLSMappingConfig dataclass
│   ├── umls_loader.py                     # Stage 3.0: Load UMLS data
│   ├── preprocessor.py                    # Stage 3.1: Preprocessing
│   ├── candidate_generator.py             # Stage 3.2: SapBERT + TF-IDF
│   ├── cluster_aggregator.py              # Stage 3.3: Synonym aggregation
│   ├── hard_negative_filter.py            # Stage 3.4: Hard negative filtering
│   ├── cross_encoder_reranker.py          # Stage 3.5: Cross-encoder
│   ├── confidence_propagator.py           # Stage 3.6: Confidence + propagation
│   └── utils.py                           # Shared utilities
│
├── workflow/
│   ├── stage3_umls_mapping.py             # NEW: Main workflow script
│   └── config/
│       └── stage3_umls_mapping.yaml       # NEW: Config file
│
└── data/
    └── umls/                              # UMLS data directory
        ├── MRCONSO.RRF                    # User downloads
        ├── MRSTY.RRF
        ├── MRDEF.RRF
        └── processed/                     # Processed indices
            ├── umls_concepts.pkl
            ├── umls_embeddings.pkl
            ├── umls_faiss.index
            └── tfidf_vectorizer.pkl
```

---

## Config File Design

**File:** `gfmrag/workflow/config/stage3_umls_mapping.yaml`

```yaml
# Stage 3: UMLS Mapping Configuration

hydra:
  run:
    dir: outputs/umls_mapping/${now:%Y-%m-%d}/${now:%H-%M-%S}

# Input/Output
input:
  kg_clean_path: tmp/kg_construction/*/hotpotqa/kg_clean.txt  # From Stage 2
  umls_data_dir: data/umls  # UMLS RRF files

output:
  root_dir: tmp/umls_mapping
  final_mappings: outputs/final_umls_mappings.json

# Stage 3.0: UMLS Setup
umls:
  files:
    mrconso: ${input.umls_data_dir}/MRCONSO.RRF
    mrsty: ${input.umls_data_dir}/MRSTY.RRF
    mrdef: ${input.umls_data_dir}/MRDEF.RRF
  language: ENG  # English only
  cache_dir: ${input.umls_data_dir}/processed

# Stage 3.1: Preprocessing
preprocessing:
  normalize:
    lowercase: true
    remove_punctuation: true
    expand_abbreviations: true
  abbreviation_dict: config/medical_abbreviations.json

# Stage 3.2: Candidate Generation
candidate_generation:
  sapbert:
    model: cambridgeltl/SapBERT-from-PubMedBERT-fulltext
    batch_size: 256
    device: cuda
    top_k: 64
  tfidf:
    ngram_range: [3, 3]  # Character trigrams
    min_df: 2
    top_k: 64
  ensemble:
    method: reciprocal_rank_fusion  # RRF
    k_constant: 60
    final_k: 128

# Stage 3.3: Cluster Aggregation
cluster_aggregation:
  score_weights:
    avg_score: 0.6
    cluster_consensus: 0.3
    method_diversity: 0.1
  outlier_threshold: 0.5  # Entities với top-1 khác majority
  output_k: 64

# Stage 3.4: Hard Negative Filtering
hard_negative_filtering:
  similarity_threshold: 0.7  # Hard negatives nếu > threshold
  semantic_type_groups:
    disease: [Disease or Syndrome, Neoplastic Process]
    drug: [Pharmacologic Substance, Antibiotic]
    procedure: [Therapeutic or Preventive Procedure]
    anatomy: [Body Part Organ or Organ Component, Tissue]
  type_match_weight: 0.2
  output_k: 32

# Stage 3.5: Cross-Encoder Reranking
cross_encoder:
  model: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
  pretrained_path: null  # Set nếu có trained model
  training:
    enabled: false  # Set true nếu muốn train
    dataset: medmentions  # medmentions/bc5cdr/ncbi_disease
    epochs: 10
    batch_size: 16
    learning_rate: 2e-5
  inference:
    batch_size: 32
    device: cuda
  score_weights:
    cross_encoder: 0.7
    previous_stage: 0.3

# Stage 3.6: Confidence & Propagation
confidence:
  factors:
    score_margin: 0.35  # Gap top-1 vs top-2
    absolute_score: 0.25
    cluster_consensus: 0.25
    method_agreement: 0.15
  tiers:
    high: 0.75  # >= 0.75
    medium: 0.50  # 0.50-0.75
    low: 0.0  # < 0.50
  propagation:
    min_cluster_agreement: 0.8  # Propagate nếu >= 80% agree
    confidence_penalty: 0.9  # Propagated = best * 0.9

# General
general:
  num_processes: 10
  force_recompute: false
  save_intermediate: true
  device: cuda
```

---

## Dependencies

**New packages cần thêm vào `pyproject.toml`:**

```toml
[tool.poetry.dependencies]
# ... existing dependencies ...

# UMLS Mapping
faiss-cpu = "^1.8.0"  # hoặc faiss-gpu nếu có GPU
scikit-learn = "^1.3.0"  # TF-IDF vectorizer
tqdm = "^4.66.0"  # Progress bars (đã có)
torch = ">=2.4.1"  # (đã có)
transformers = "^4.46.1"  # (đã có - cho SapBERT/PubMedBERT)
```

---

## Implementation Steps

### Step 1: Create Config Dataclass

**File:** `gfmrag/umls_mapping/config.py`

```python
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class UMLSMappingConfig:
    """Configuration for UMLS mapping pipeline"""

    # Input/Output
    kg_clean_path: str
    umls_data_dir: str
    output_root: str

    # UMLS files
    mrconso_path: str
    mrsty_path: str
    mrdef_path: str

    # Stage 3.2: Candidate Generation
    sapbert_model: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    sapbert_batch_size: int = 256
    sapbert_device: str = "cuda"
    sapbert_top_k: int = 64

    tfidf_ngram_range: tuple = (3, 3)
    tfidf_min_df: int = 2
    tfidf_top_k: int = 64

    ensemble_k: int = 128
    rrf_k_constant: int = 60

    # Stage 3.3: Cluster Aggregation
    cluster_score_weights: Dict[str, float] = field(default_factory=lambda: {
        "avg_score": 0.6,
        "cluster_consensus": 0.3,
        "method_diversity": 0.1
    })
    cluster_output_k: int = 64

    # Stage 3.4: Hard Negative Filtering
    hard_neg_similarity_threshold: float = 0.7
    semantic_type_match_weight: float = 0.2
    hard_neg_output_k: int = 32

    # Stage 3.5: Cross-Encoder
    cross_encoder_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    cross_encoder_device: str = "cuda"
    cross_encoder_score_weight: float = 0.7

    # Stage 3.6: Confidence
    confidence_weights: Dict[str, float] = field(default_factory=lambda: {
        "score_margin": 0.35,
        "absolute_score": 0.25,
        "cluster_consensus": 0.25,
        "method_agreement": 0.15
    })
    min_cluster_agreement: float = 0.8

    # General
    num_processes: int = 10
    force: bool = False
    save_intermediate: bool = True
```

---

### Step 2: Main Pipeline Class

**File:** `gfmrag/workflow/stage3_umls_mapping.py`

```python
"""
Stage 3: UMLS Mapping Pipeline
================================

Maps biomedical entities to UMLS CUIs through 6-stage pipeline:
1. Preprocessing
2. Candidate generation (SapBERT + TF-IDF ensemble)
3. Synonym cluster aggregation
4. Hard negative filtering
5. Cross-encoder reranking
6. Confidence scoring & propagation

Target accuracy: 85-90%
"""

import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv

from gfmrag.umls_mapping.config import UMLSMappingConfig
from gfmrag.umls_mapping import (
    UMLSLoader,
    Preprocessor,
    CandidateGenerator,
    ClusterAggregator,
    HardNegativeFilter,
    CrossEncoderReranker,
    ConfidencePropagator
)

load_dotenv()
logger = logging.getLogger(__name__)


class UMLSMappingPipeline:
    """Main UMLS mapping pipeline"""

    def __init__(self, config: UMLSMappingConfig):
        self.config = config
        self.output_dir = Path(config.output_root)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Stage outputs
        self.stage_paths = {
            "stage0_umls_concepts": self.output_dir / "stage0_umls_concepts.pkl",
            "stage1_entities": self.output_dir / "stage1_entities.json",
            "stage1_clusters": self.output_dir / "stage1_synonym_clusters.json",
            "stage2_candidates": self.output_dir / "stage2_candidates.json",
            "stage3_refined": self.output_dir / "stage3_refined_candidates.json",
            "stage4_filtered": self.output_dir / "stage4_filtered_candidates.json",
            "stage5_reranked": self.output_dir / "stage5_reranked.json",
            "stage6_final": self.output_dir / "stage6_final_mappings.json",
        }

    def run(self):
        """Execute full pipeline"""

        # Stage 0: Load UMLS
        logger.info("="*80)
        logger.info("STAGE 0: LOADING UMLS DATA")
        logger.info("="*80)
        umls_data = self.stage0_load_umls()

        # Stage 1: Preprocessing
        logger.info("\n" + "="*80)
        logger.info("STAGE 1: PREPROCESSING")
        logger.info("="*80)
        entities, clusters = self.stage1_preprocess(umls_data)

        # Stage 2: Candidate Generation
        logger.info("\n" + "="*80)
        logger.info("STAGE 2: CANDIDATE GENERATION")
        logger.info("="*80)
        candidates = self.stage2_generate_candidates(entities, umls_data)

        # Stage 3: Cluster Aggregation
        logger.info("\n" + "="*80)
        logger.info("STAGE 3: CLUSTER AGGREGATION")
        logger.info("="*80)
        refined = self.stage3_aggregate_clusters(candidates, clusters)

        # Stage 4: Hard Negative Filtering
        logger.info("\n" + "="*80)
        logger.info("STAGE 4: HARD NEGATIVE FILTERING")
        logger.info("="*80)
        filtered = self.stage4_filter_hard_negatives(refined, umls_data)

        # Stage 5: Cross-Encoder Reranking
        logger.info("\n" + "="*80)
        logger.info("STAGE 5: CROSS-ENCODER RERANKING")
        logger.info("="*80)
        reranked = self.stage5_rerank(filtered, umls_data)

        # Stage 6: Confidence & Propagation
        logger.info("\n" + "="*80)
        logger.info("STAGE 6: CONFIDENCE & PROPAGATION")
        logger.info("="*80)
        final = self.stage6_confidence_propagation(reranked, clusters)

        logger.info("\n" + "="*80)
        logger.info("UMLS MAPPING COMPLETE")
        logger.info("="*80)
        logger.info(f"Output: {self.stage_paths['stage6_final']}")

        return final

    def stage0_load_umls(self):
        """Load and index UMLS data"""
        loader = UMLSLoader(self.config)
        return loader.load()

    def stage1_preprocess(self, umls_data):
        """Extract entities and build synonym clusters"""
        preprocessor = Preprocessor(self.config)
        return preprocessor.process()

    def stage2_generate_candidates(self, entities, umls_data):
        """Generate top-128 candidates using ensemble"""
        generator = CandidateGenerator(self.config, umls_data)
        return generator.generate(entities)

    def stage3_aggregate_clusters(self, candidates, clusters):
        """Aggregate candidates across synonym clusters"""
        aggregator = ClusterAggregator(self.config)
        return aggregator.aggregate(candidates, clusters)

    def stage4_filter_hard_negatives(self, refined, umls_data):
        """Filter hard negatives and semantic type mismatches"""
        filter = HardNegativeFilter(self.config, umls_data)
        return filter.filter(refined)

    def stage5_rerank(self, filtered, umls_data):
        """Rerank with cross-encoder"""
        reranker = CrossEncoderReranker(self.config, umls_data)
        return reranker.rerank(filtered)

    def stage6_confidence_propagation(self, reranked, clusters):
        """Compute confidence and propagate through clusters"""
        propagator = ConfidencePropagator(self.config)
        return propagator.propagate(reranked, clusters)


@hydra.main(config_path="config", config_name="stage3_umls_mapping", version_base=None)
def main(cfg: DictConfig):
    """Main entry point"""

    # Convert OmegaConf to UMLSMappingConfig
    config = UMLSMappingConfig(
        kg_clean_path=cfg.input.kg_clean_path,
        umls_data_dir=cfg.input.umls_data_dir,
        output_root=cfg.output.root_dir,
        mrconso_path=cfg.umls.files.mrconso,
        mrsty_path=cfg.umls.files.mrsty,
        mrdef_path=cfg.umls.files.mrdef,
        # ... map all config values ...
    )

    # Run pipeline
    pipeline = UMLSMappingPipeline(config)
    results = pipeline.run()

    logger.info(f"✅ UMLS mapping completed successfully")


if __name__ == "__main__":
    main()
```

---

## Next Steps

1. **Create directory structure**
2. **Implement each stage module** (6 files in `gfmrag/umls_mapping/`)
3. **Create config file**
4. **Download UMLS data** (requires UMLS license)
5. **Test incrementally**

---

## Estimated Timeline

- **Setup & UMLS download**: 1 day
- **Stage 0-1 (Loading + Preprocessing)**: 2-3 days
- **Stage 2 (Candidate Generation)**: 3-4 days
- **Stage 3-4 (Aggregation + Filtering)**: 2-3 days
- **Stage 5 (Cross-Encoder)**: 3-5 days (including training)
- **Stage 6 (Confidence)**: 1-2 days
- **Testing & Metrics**: 2-3 days

**Total**: ~15-20 days for full implementation

---

## Key Challenges

1. **UMLS Data Size**: 4M+ concepts, ~12GB embeddings → Need efficient indexing
2. **Cross-Encoder Training**: Need annotated data (can use public datasets)
3. **Performance**: GPU highly recommended
4. **Testing**: Need gold standard annotations for evaluation

---

**Status**: Design phase complete, ready for implementation approval
