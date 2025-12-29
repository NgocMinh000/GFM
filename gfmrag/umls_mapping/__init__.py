"""
UMLS Mapping Package
====================

6-stage pipeline for mapping biomedical entities to UMLS CUIs.

Stages:
    0. UMLS Data Loading
    1. Preprocessing & Synonym Clustering
    2. Candidate Generation (SapBERT + TF-IDF)
    3. Synonym Cluster Aggregation
    4. Hard Negative Filtering
    5. Cross-Encoder Reranking
    6. Confidence Scoring & Propagation
"""

from .config import UMLSMappingConfig
from .umls_loader import UMLSLoader
from .preprocessor import Preprocessor
from .candidate_generator import CandidateGenerator
from .cluster_aggregator import ClusterAggregator
from .hard_negative_filter import HardNegativeFilter
from .cross_encoder_reranker import CrossEncoderReranker
from .confidence_propagator import ConfidencePropagator
from .metrics import (
    MetricsTracker,
    Stage0Metrics,
    Stage1Metrics,
    Stage2Metrics,
    Stage3Metrics,
    Stage4Metrics,
    Stage5Metrics,
    Stage6Metrics,
)

__all__ = [
    "UMLSMappingConfig",
    "UMLSLoader",
    "Preprocessor",
    "CandidateGenerator",
    "ClusterAggregator",
    "HardNegativeFilter",
    "CrossEncoderReranker",
    "ConfidencePropagator",
    "MetricsTracker",
    "Stage0Metrics",
    "Stage1Metrics",
    "Stage2Metrics",
    "Stage3Metrics",
    "Stage4Metrics",
    "Stage5Metrics",
    "Stage6Metrics",
]

__version__ = "1.0.0"
