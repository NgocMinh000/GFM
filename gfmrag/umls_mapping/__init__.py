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
from .config_loader import load_config, save_config, create_default_config
from .pipeline import UMLSMappingPipeline, PipelineStatus
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
from .validation import Stage1Validator
from .visualization import PipelineVisualizer, visualize_pipeline_metrics

__all__ = [
    "UMLSMappingConfig",
    "load_config",
    "save_config",
    "create_default_config",
    "UMLSMappingPipeline",
    "PipelineStatus",
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
    "Stage1Validator",
    "PipelineVisualizer",
    "visualize_pipeline_metrics",
]

__version__ = "1.0.0"
