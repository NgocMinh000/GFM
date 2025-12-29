"""
Configuration dataclass for UMLS Mapping Pipeline
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class UMLSMappingConfig:
    """Configuration for 6-stage UMLS mapping pipeline"""

    # Input/Output Paths
    kg_clean_path: str
    umls_data_dir: str
    output_root: str
    mrconso_path: str
    mrsty_path: str
    mrdef_path: Optional[str] = None

    # Stage 3.0: UMLS Setup
    umls_language: str = "ENG"
    umls_cache_dir: str = "data/umls/processed"
    precompute_embeddings: bool = True

    # Stage 3.2: Candidate Generation
    sapbert_model: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    sapbert_batch_size: int = 256
    sapbert_device: str = "cuda"
    sapbert_top_k: int = 64
    tfidf_ngram_range: tuple = (3, 3)
    ensemble_final_k: int = 128

    # Stage 3.3: Cluster Aggregation
    cluster_output_k: int = 64

    # Stage 3.4: Hard Negative Filtering
    hard_neg_similarity_threshold: float = 0.7
    hard_neg_output_k: int = 32

    # Stage 3.5: Cross-Encoder
    cross_encoder_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    cross_encoder_device: str = "cuda"

    # Stage 3.6: Confidence
    confidence_high_threshold: float = 0.75
    propagation_min_agreement: float = 0.8

    # General
    num_processes: int = 10
    force_recompute: bool = False
    save_intermediate: bool = True
    device: str = "cuda"
