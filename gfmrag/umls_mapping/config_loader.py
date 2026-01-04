"""
Configuration loader for UMLS Mapping Pipeline

Loads YAML config and converts to UMLSMappingConfig dataclass
"""

import yaml
from pathlib import Path
from typing import Union, Dict, Any

from .config import UMLSMappingConfig


def load_config(config_path: Union[str, Path]) -> UMLSMappingConfig:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to YAML config file

    Returns:
        UMLSMappingConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Validate required fields
    required_fields = [
        'kg_clean_path',
        'umls_data_dir',
        'output_root',
        'mrconso_path',
        'mrsty_path'
    ]

    missing_fields = [f for f in required_fields if f not in config_dict]
    if missing_fields:
        raise ValueError(f"Missing required config fields: {missing_fields}")

    # Convert tfidf_ngram_range from list to tuple
    if 'tfidf_ngram_range' in config_dict and isinstance(config_dict['tfidf_ngram_range'], list):
        config_dict['tfidf_ngram_range'] = tuple(config_dict['tfidf_ngram_range'])

    # Remove runtime options (not part of UMLSMappingConfig)
    runtime_fields = ['stages', 'resume', 'force']
    runtime_options = {k: config_dict.pop(k) for k in runtime_fields if k in config_dict}

    # Create config
    try:
        config = UMLSMappingConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Invalid config: {e}")

    return config, runtime_options


def save_config(config: UMLSMappingConfig, output_path: Union[str, Path]):
    """
    Save configuration to YAML file

    Args:
        config: UMLSMappingConfig instance
        output_path: Path to save YAML file
    """
    output_path = Path(output_path)

    # Convert dataclass to dict
    config_dict = {
        'kg_clean_path': config.kg_clean_path,
        'umls_data_dir': config.umls_data_dir,
        'output_root': config.output_root,
        'mrconso_path': config.mrconso_path,
        'mrsty_path': config.mrsty_path,
        'mrdef_path': config.mrdef_path,
        'umls_language': config.umls_language,
        'umls_cache_dir': config.umls_cache_dir,
        'precompute_embeddings': config.precompute_embeddings,
        'sapbert_model': config.sapbert_model,
        'sapbert_batch_size': config.sapbert_batch_size,
        'sapbert_device': config.sapbert_device,
        'sapbert_top_k': config.sapbert_top_k,
        'tfidf_ngram_range': list(config.tfidf_ngram_range),
        'ensemble_final_k': config.ensemble_final_k,
        'cluster_output_k': config.cluster_output_k,
        'hard_neg_similarity_threshold': config.hard_neg_similarity_threshold,
        'hard_neg_output_k': config.hard_neg_output_k,
        'cross_encoder_model': config.cross_encoder_model,
        'cross_encoder_device': config.cross_encoder_device,
        'confidence_high_threshold': config.confidence_high_threshold,
        'propagation_min_agreement': config.propagation_min_agreement,
        'num_processes': config.num_processes,
        'force_recompute': config.force_recompute,
        'save_intermediate': config.save_intermediate,
        'device': config.device,
    }

    # Save to YAML
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def create_default_config(output_path: Union[str, Path]):
    """
    Create a default configuration file

    Args:
        output_path: Path to save default config
    """
    output_path = Path(output_path)

    # Create default config
    default_config = UMLSMappingConfig(
        kg_clean_path="./data/kg_clean.txt",
        umls_data_dir="./data/umls/2024AB/META",
        output_root="./outputs",
        mrconso_path="./data/umls/2024AB/META/MRCONSO.RRF",
        mrsty_path="./data/umls/2024AB/META/MRSTY.RRF",
        mrdef_path="./data/umls/2024AB/META/MRDEF.RRF",
    )

    save_config(default_config, output_path)
    print(f"Default config created: {output_path}")
