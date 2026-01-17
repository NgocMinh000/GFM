#!/usr/bin/env python3
"""
Recalculate Stage 3.5 scores with new formula and pre-filtering

This script:
1. Loads existing stage35_reranked.json results
2. Applies pre-filtering (previous_score >= 0.6 AND cross_encoder_score >= 0.5)
3. Recalculates scores with new formula (0.4 cross-encoder + 0.6 previous)
4. Saves updated results back to stage35_reranked.json

Usage:
    python recalculate_stage35_scores.py
    python recalculate_stage35_scores.py --input tmp/umls_mapping/stage35_reranked.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def recalculate_scores(
    input_file: Path,
    min_prev_score: float = 0.6,
    min_cross_score: float = 0.5,
    cross_weight: float = 0.4,
    prev_weight: float = 0.6
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Recalculate scores for stage 3.5 results

    Args:
        input_file: Path to stage35_reranked.json
        min_prev_score: Minimum previous_score threshold (default: 0.6)
        min_cross_score: Minimum cross_encoder_score threshold (default: 0.5)
        cross_weight: Weight for cross_encoder_score (default: 0.4)
        prev_weight: Weight for previous_score (default: 0.6)

    Returns:
        Updated results dictionary
    """

    logger.info(f"Loading stage 3.5 results from: {input_file}")

    if not input_file.exists():
        logger.error(f"File not found: {input_file}")
        logger.info("Available files in directory:")
        for f in input_file.parent.glob("*.json"):
            logger.info(f"  - {f.name}")
        raise FileNotFoundError(f"File not found: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} entities")
    logger.info(f"\nRecalculation parameters:")
    logger.info(f"  - Min previous_score: {min_prev_score}")
    logger.info(f"  - Min cross_encoder_score: {min_cross_score}")
    logger.info(f"  - Cross-encoder weight: {cross_weight}")
    logger.info(f"  - Previous score weight: {prev_weight}")

    # Statistics
    total_entities = 0
    total_candidates_before = 0
    total_candidates_after = 0
    total_filtered_prev = 0
    total_filtered_cross = 0
    entities_with_no_candidates = []

    # Recalculate for each entity
    updated_data = {}

    for entity, candidates in data.items():
        if not candidates:
            updated_data[entity] = []
            continue

        total_entities += 1
        total_candidates_before += len(candidates)

        filtered_candidates = []
        filtered_prev = 0
        filtered_cross = 0

        for candidate in candidates:
            # Extract scores
            prev_score = candidate.get('previous_score', candidate.get('score', 0.0))
            cross_score = candidate.get('cross_encoder_score', 0.0)

            # Pre-filtering
            if prev_score < min_prev_score:
                filtered_prev += 1
                continue

            if cross_score < min_cross_score:
                filtered_cross += 1
                continue

            # Recalculate final score with new formula
            new_score = cross_score * cross_weight + prev_score * prev_weight

            # Update candidate
            updated_candidate = candidate.copy()
            updated_candidate['score'] = new_score
            updated_candidate['previous_score'] = prev_score
            updated_candidate['cross_encoder_score'] = cross_score
            updated_candidate['method'] = 'cross_encoder_reranked_v2'  # Mark as recalculated

            filtered_candidates.append(updated_candidate)

        # Re-sort by new score
        filtered_candidates.sort(key=lambda x: x['score'], reverse=True)

        updated_data[entity] = filtered_candidates

        # Update statistics
        total_candidates_after += len(filtered_candidates)
        total_filtered_prev += filtered_prev
        total_filtered_cross += filtered_cross

        if len(filtered_candidates) == 0 and len(candidates) > 0:
            entities_with_no_candidates.append(entity)

        # Log sample for first few entities
        if total_entities <= 3:
            logger.info(f"\n--- Entity: '{entity}' ---")
            logger.info(f"  Before: {len(candidates)} candidates")
            logger.info(f"  Filtered (low prev_score): {filtered_prev}")
            logger.info(f"  Filtered (low cross_score): {filtered_cross}")
            logger.info(f"  After: {len(filtered_candidates)} candidates")
            if filtered_candidates:
                top = filtered_candidates[0]
                logger.info(f"  Top candidate: {top['name']} (CUI: {top['cui']})")
                logger.info(f"    - Previous score: {top['previous_score']:.4f}")
                logger.info(f"    - Cross-encoder score: {top['cross_encoder_score']:.4f}")
                logger.info(f"    - New final score: {top['score']:.4f}")

    # Print summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("RECALCULATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total entities: {total_entities}")
    logger.info(f"Total candidates before: {total_candidates_before}")
    logger.info(f"Total candidates after: {total_candidates_after}")
    logger.info(f"Filtered (low previous_score < {min_prev_score}): {total_filtered_prev}")
    logger.info(f"Filtered (low cross_encoder_score < {min_cross_score}): {total_filtered_cross}")
    logger.info(f"Total filtered: {total_filtered_prev + total_filtered_cross}")
    logger.info(f"Filtering rate: {(total_filtered_prev + total_filtered_cross) / total_candidates_before * 100:.2f}%")
    logger.info(f"Entities with no candidates after filtering: {len(entities_with_no_candidates)}")

    if entities_with_no_candidates:
        logger.warning(f"\nEntities with ALL candidates filtered out ({len(entities_with_no_candidates)}):")
        for entity in entities_with_no_candidates[:10]:
            logger.warning(f"  - {entity}")
        if len(entities_with_no_candidates) > 10:
            logger.warning(f"  ... and {len(entities_with_no_candidates) - 10} more")

    logger.info("=" * 80)

    return updated_data


def main():
    parser = argparse.ArgumentParser(
        description='Recalculate Stage 3.5 scores with new formula and pre-filtering'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='tmp/umls_mapping/stage35_reranked.json',
        help='Input file path (default: tmp/umls_mapping/stage35_reranked.json)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: overwrites input file)'
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create backup of original file before overwriting'
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
        '--prev-weight',
        type=float,
        default=0.6,
        help='Weight for previous_score (default: 0.6)'
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output) if args.output else input_file

    # Recalculate scores
    updated_data = recalculate_scores(
        input_file=input_file,
        min_prev_score=args.min_prev_score,
        min_cross_score=args.min_cross_score,
        cross_weight=args.cross_weight,
        prev_weight=args.prev_weight
    )

    # Create backup if requested
    if args.backup and output_file == input_file:
        backup_file = input_file.with_suffix('.json.backup')
        logger.info(f"\nCreating backup: {backup_file}")
        import shutil
        shutil.copy2(input_file, backup_file)

    # Save updated results
    logger.info(f"\nSaving updated results to: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ Successfully saved updated results!")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Verify the results look correct")
    logger.info(f"  2. Continue with Stage 3.6 (confidence scoring)")
    logger.info(f"  3. Run: python -m gfmrag.workflow.stage3_umls_mapping --start-from stage36")


if __name__ == '__main__':
    main()
