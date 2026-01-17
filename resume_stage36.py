#!/usr/bin/env python3
"""
Resume Stage 3 pipeline from Stage 3.6 (Confidence Scoring)

This script:
1. Recalculates Stage 3.5 scores with new formula (0.4/0.6) and pre-filtering
2. Loads necessary data from previous stages
3. Runs Stage 3.6 (Confidence Scoring & Propagation)
4. Saves final outputs

Usage:
    python resume_stage36.py
    python resume_stage36.py --config gfmrag/workflow/config/stage3_umls_mapping.yaml
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List
from dataclasses import asdict
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from gfmrag.umls_mapping import (
    UMLSMappingConfig,
    UMLSLoader,
    Preprocessor,
    ConfidencePropagator,
    MetricsTracker,
    Stage6Metrics,
)
from gfmrag.umls_mapping.cross_encoder_reranker import RerankedCandidate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def recalculate_stage35_scores(
    input_file: Path,
    min_prev_score: float = 0.6,
    min_cross_score: float = 0.5,
    cross_weight: float = 0.4,
) -> Dict[str, List[RerankedCandidate]]:
    """
    Recalculate scores from stage35_reranked.json

    Returns:
        Dictionary mapping entity -> list of RerankedCandidate objects
    """

    logger.info(f"Loading stage 3.5 results from: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} entities")
    logger.info(f"Recalculation parameters:")
    logger.info(f"  - Min previous_score: {min_prev_score}")
    logger.info(f"  - Min cross_encoder_score: {min_cross_score}")
    logger.info(f"  - Cross-encoder weight: {cross_weight}")
    logger.info(f"  - Previous score weight: {1.0 - cross_weight}")

    # Statistics
    total_filtered = 0
    total_candidates_before = 0

    # Recalculate for each entity
    updated_data = {}

    for entity, candidates in tqdm(data.items(), desc="Recalculating scores"):
        if not candidates:
            updated_data[entity] = []
            continue

        total_candidates_before += len(candidates)
        filtered_candidates = []

        for candidate in candidates:
            # Extract scores
            prev_score = candidate.get('previous_score', candidate.get('score', 0.0))
            cross_score = candidate.get('cross_encoder_score', 0.0)

            # Pre-filtering
            if prev_score < min_prev_score or cross_score < min_cross_score:
                total_filtered += 1
                continue

            # Recalculate final score with new formula
            new_score = cross_score * cross_weight + prev_score * (1.0 - cross_weight)

            # Create RerankedCandidate object
            reranked = RerankedCandidate(
                cui=candidate['cui'],
                name=candidate['name'],
                score=new_score,
                cross_encoder_score=cross_score,
                previous_score=prev_score,
                method='cross_encoder_reranked_v2'
            )

            filtered_candidates.append(reranked)

        # Re-sort by new score
        filtered_candidates.sort(key=lambda x: x.score, reverse=True)
        updated_data[entity] = filtered_candidates

    logger.info(f"\nFiltered {total_filtered}/{total_candidates_before} candidates ({total_filtered/total_candidates_before*100:.2f}%)")
    logger.info(f"Remaining: {total_candidates_before - total_filtered} candidates")

    return updated_data


def load_preprocessor_data(stage31_path: Path) -> tuple:
    """
    Load entities and synonym clusters from stage 3.1 preprocessing

    Returns:
        (entities dict, synonym_clusters dict)
    """

    logger.info(f"Loading preprocessing data from: {stage31_path}")

    # The preprocessing stage should have saved entity data
    # We need to reconstruct the entity objects
    # For now, we'll work with a simplified version

    # Load the entities (this might need adjustment based on actual file structure)
    with open(stage31_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract entity -> synonym_group mapping
    entities = {}
    synonym_clusters = {}

    for entity, entity_data in data.items():
        # Create simplified entity object
        from dataclasses import dataclass

        @dataclass
        class SimpleEntity:
            original: str
            normalized: str
            synonym_group: frozenset

        # Get synonym group
        if isinstance(entity_data, dict):
            synonym_group = frozenset(entity_data.get('synonym_group', [entity]))
        else:
            synonym_group = frozenset([entity])

        entities[entity] = SimpleEntity(
            original=entity,
            normalized=entity_data.get('normalized', entity) if isinstance(entity_data, dict) else entity,
            synonym_group=synonym_group
        )

        # Build synonym_clusters (cluster -> members)
        cluster_key = tuple(sorted(synonym_group))
        if cluster_key not in synonym_clusters:
            synonym_clusters[cluster_key] = list(synonym_group)

    logger.info(f"Loaded {len(entities)} entities in {len(synonym_clusters)} synonym clusters")

    return entities, synonym_clusters


def main():
    parser = argparse.ArgumentParser(
        description='Resume Stage 3 pipeline from Stage 3.6'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='tmp/umls_mapping',
        help='Output directory (default: tmp/umls_mapping)'
    )
    parser.add_argument(
        '--stage35-file',
        type=str,
        default=None,
        help='Stage 3.5 results file (default: <output-dir>/stage35_reranked.json)'
    )
    parser.add_argument(
        '--stage31-file',
        type=str,
        default=None,
        help='Stage 3.1 preprocessing file (default: <output-dir>/stage31_preprocessing.json)'
    )
    parser.add_argument(
        '--min-prev-score',
        type=float,
        default=0.6,
        help='Minimum previous_score threshold (default: 0.6)'
    )
    parser.add_argument(
        '--min-cross-score',
        type=float,
        default=0.5,
        help='Minimum cross_encoder_score threshold (default: 0.5)'
    )
    parser.add_argument(
        '--cross-weight',
        type=float,
        default=0.4,
        help='Weight for cross_encoder_score (default: 0.4)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='gfmrag/workflow/config/stage3_umls_mapping.yaml',
        help='Config file path'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    stage35_file = Path(args.stage35_file) if args.stage35_file else output_dir / "stage35_reranked.json"
    stage31_file = Path(args.stage31_file) if args.stage31_file else output_dir / "stage31_preprocessing.json"

    logger.info("=" * 80)
    logger.info("RESUMING STAGE 3 PIPELINE FROM STAGE 3.6")
    logger.info("=" * 80)

    # Step 1: Recalculate Stage 3.5 scores
    logger.info("\n[Step 1] Recalculating Stage 3.5 scores...")
    reranked_candidates = recalculate_stage35_scores(
        input_file=stage35_file,
        min_prev_score=args.min_prev_score,
        min_cross_score=args.min_cross_score,
        cross_weight=args.cross_weight
    )

    # Save recalculated results
    logger.info("\nSaving recalculated Stage 3.5 results...")
    recalc_path = output_dir / "stage35_reranked_recalculated.json"
    with open(recalc_path, 'w', encoding='utf-8') as f:
        json_data = {}
        for entity, candidates in reranked_candidates.items():
            json_data[entity] = [asdict(c) for c in candidates]
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved to: {recalc_path}")

    # Step 2: Load preprocessing data
    logger.info("\n[Step 2] Loading preprocessing data...")
    try:
        entities, synonym_clusters = load_preprocessor_data(stage31_file)
    except Exception as e:
        logger.error(f"Failed to load preprocessing data: {e}")
        logger.info("You may need to manually provide the entity and synonym cluster data")
        logger.info("Or re-run stages 3.1-3.5 with the updated config")
        return

    # Step 3: Load config
    logger.info("\n[Step 3] Loading config...")
    import hydra
    from hydra import compose, initialize
    from omegaconf import DictConfig

    # Load config using hydra
    try:
        with initialize(version_base=None, config_path="gfmrag/workflow/config"):
            cfg = compose(config_name="stage3_umls_mapping")

        # Convert to UMLSMappingConfig
        config = UMLSMappingConfig(
            kg_clean_path=cfg.paths.kg_clean_path,
            umls_data_dir=cfg.paths.umls_data_dir,
            output_root=cfg.paths.root_dir,
            mrconso_path=cfg.paths.mrconso_path,
            mrsty_path=cfg.paths.mrsty_path,
            mrdef_path=cfg.paths.get('mrdef_path'),
            cross_encoder_weight=0.4,
            cross_encoder_min_prev_score=0.6,
            cross_encoder_min_cross_score=0.5
        )
    except Exception as e:
        logger.warning(f"Failed to load config via Hydra: {e}")
        logger.info("Creating default config...")
        config = UMLSMappingConfig(
            kg_clean_path="tmp/kg_construction/*/hotpotqa/kg_clean.txt",
            umls_data_dir="data/umls",
            output_root=str(output_dir),
            mrconso_path="data/umls/2024AB/META/MRCONSO.RRF",
            mrsty_path="data/umls/2024AB/META/MRSTY.RRF",
            cross_encoder_weight=0.4,
            cross_encoder_min_prev_score=0.6,
            cross_encoder_min_cross_score=0.5
        )

    # Step 4: Initialize Stage 3.6 components
    logger.info("\n[Step 4] Initializing Stage 3.6 components...")

    # Initialize UMLS loader (needed for confidence propagator)
    logger.info("Loading UMLS data...")
    umls_loader = UMLSLoader(config)
    umls_loader.load()

    # Initialize confidence propagator
    confidence_propagator = ConfidencePropagator(config)

    # Initialize metrics tracker
    metrics = MetricsTracker(output_dir)

    # Step 5: Run Stage 3.6
    logger.info("\n[Stage 3.6] Computing confidence & propagating...")
    metrics.start_stage("Stage 3.6: Confidence & Propagation", input_count=len(reranked_candidates))

    # Compute initial mappings
    final_mappings = {}
    for entity, candidates in tqdm(reranked_candidates.items(), desc="Computing confidence"):
        if entity not in entities:
            logger.warning(f"Entity not found in preprocessing data: {entity}")
            cluster_members = frozenset([entity])
        else:
            cluster_members = entities[entity].synonym_group

        mapping = confidence_propagator.compute_confidence(
            entity,
            candidates,
            cluster_members
        )
        final_mappings[entity] = mapping

        # Track low confidence cases
        if mapping.tier == 'low':
            metrics.add_warning(f"Low confidence mapping: {entity} -> {mapping.cui} ({mapping.confidence:.3f})")

    # Propagate through clusters
    logger.info("Propagating through synonym clusters...")
    final_mappings = confidence_propagator.finalize_all_mappings(
        final_mappings,
        synonym_clusters
    )

    # Compute metrics
    stage6_metrics = Stage6Metrics.compute(final_mappings)
    for key, value in stage6_metrics.items():
        metrics.add_metric(key, value)

    metrics.end_stage(output_count=len(final_mappings))

    # Step 6: Save final results
    logger.info("\n[Step 6] Saving final outputs...")

    # Save final mappings JSON
    json_path = output_dir / "final_umls_mappings_v2.json"
    json_data = {}
    for entity, mapping in final_mappings.items():
        json_data[entity] = {
            'cui': mapping.cui,
            'name': mapping.name,
            'confidence': mapping.confidence,
            'tier': mapping.tier,
            'alternatives': [
                {'cui': cui, 'name': name, 'score': score}
                for cui, name, score in mapping.alternatives
            ],
            'cluster_size': len(mapping.cluster_members),
            'is_propagated': mapping.is_propagated,
            'confidence_factors': mapping.confidence_factors
        }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, default=str)
    logger.info(f"Saved final mappings to: {json_path}")

    # Save triples
    triples_path = output_dir / "umls_mapping_triples_v2.txt"
    with open(triples_path, 'w', encoding='utf-8') as f:
        for entity, mapping in final_mappings.items():
            if mapping.confidence >= 0.5:
                f.write(f"{entity}|mapped_to_cui|{mapping.cui}\n")
    logger.info(f"Saved KG triples to: {triples_path}")

    # Save statistics
    stats_path = output_dir / "mapping_statistics_v2.json"
    total = len(final_mappings)
    high = sum(1 for m in final_mappings.values() if m.tier == 'high')
    medium = sum(1 for m in final_mappings.values() if m.tier == 'medium')
    low = sum(1 for m in final_mappings.values() if m.tier == 'low')
    propagated = sum(1 for m in final_mappings.values() if m.is_propagated)

    stats = {
        'total_entities': total,
        'high_confidence': high,
        'medium_confidence': medium,
        'low_confidence': low,
        'propagated': propagated,
        'high_confidence_pct': f"{high/total*100:.2f}%",
        'medium_confidence_pct': f"{medium/total*100:.2f}%",
        'low_confidence_pct': f"{low/total*100:.2f}%",
        'propagated_pct': f"{propagated/total*100:.2f}%",
        'recalculation_params': {
            'min_prev_score': args.min_prev_score,
            'min_cross_score': args.min_cross_score,
            'cross_weight': args.cross_weight,
            'prev_weight': 1.0 - args.cross_weight
        }
    }

    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved statistics to: {stats_path}")

    # Save metrics
    metrics.save_metrics()

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 3.6 COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nOutput files:")
    logger.info(f"  - Final mappings: {json_path}")
    logger.info(f"  - KG triples: {triples_path}")
    logger.info(f"  - Statistics: {stats_path}")
    logger.info(f"  - Metrics: {output_dir / 'pipeline_metrics.json'}")
    logger.info(f"\nQuality Summary:")
    logger.info(f"  - High confidence: {high} ({high/total*100:.1f}%)")
    logger.info(f"  - Medium confidence: {medium} ({medium/total*100:.1f}%)")
    logger.info(f"  - Low confidence: {low} ({low/total*100:.1f}%)")
    logger.info(f"  - Propagated: {propagated} ({propagated/total*100:.1f}%)")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
