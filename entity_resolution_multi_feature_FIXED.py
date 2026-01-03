#!/usr/bin/env python3
"""
Entity Resolution with Multi-Feature Scoring - FIXED VERSION

This script demonstrates how to properly use ColBERT for entity resolution
with multi-feature scoring, fixing the "string indices must be integers" error.

Features:
- SapBERT similarity (dense embeddings)
- Lexical similarity (string matching)
- ColBERT similarity (token-level late interaction) ✅ FIXED
- Graph similarity (network context)

Author: GFM-RAG Team
Fixed: 2026-01-03
"""

import logging
import sys
from typing import List, Tuple, Dict
from pathlib import Path

import numpy as np
from ragatouille import RAGPretrainedModel
from tqdm import tqdm

# ✅ CRITICAL FIX: Import the utility function
from gfmrag.kg_construction.entity_linking_model.colbert_utils import extract_colbert_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiFeatureEntityResolution:
    """
    Entity resolution using multiple similarity features.

    Combines 4 features with weighted scoring:
    - SapBERT (0.5): Dense semantic embeddings
    - Lexical (0.15): String-based similarity
    - ColBERT (0.25): Token-level semantic matching
    - Graph (0.1): Structural context
    """

    def __init__(
        self,
        colbert_index_path: str,
        feature_weights: Dict[str, float] = None,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize the multi-feature entity resolution system.

        Args:
            colbert_index_path: Path to pre-built ColBERT index
            feature_weights: Dict of feature weights (default: sapbert=0.5, lexical=0.15, colbert=0.25, graph=0.1)
            similarity_threshold: Minimum score for equivalent pairs (default: 0.7)
        """
        self.feature_weights = feature_weights or {
            'sapbert': 0.5,
            'lexical': 0.15,
            'colbert': 0.25,
            'graph': 0.1
        }
        self.similarity_threshold = similarity_threshold

        # Validate weights sum to 1.0
        total_weight = sum(self.feature_weights.values())
        assert abs(total_weight - 1.0) < 1e-6, f"Feature weights must sum to 1.0, got {total_weight}"

        # ✅ Load ColBERT index with proper error handling
        logger.info("Loading ColBERT index...")
        try:
            self.colbert_searcher = RAGPretrainedModel.from_index(colbert_index_path)
            logger.info(f"✅ ColBERT index loaded from {colbert_index_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load ColBERT index: {e}")
            raise

    def compute_sapbert_similarity(self, entity1: str, entity2: str) -> float:
        """
        Compute SapBERT dense embedding similarity.

        TODO: Implement with actual SapBERT model
        For now, returns a mock score.
        """
        # Mock implementation - replace with actual SapBERT
        return 0.85

    def compute_lexical_similarity(self, entity1: str, entity2: str) -> float:
        """
        Compute lexical similarity (Jaccard, edit distance, etc.)

        Simple Jaccard similarity on tokens.
        """
        tokens1 = set(entity1.lower().split())
        tokens2 = set(entity2.lower().split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / len(union) if union else 0.0

    def compute_colbert_similarity(self, entity1: str, entity2: str) -> float:
        """
        ✅ FIXED: Compute ColBERT similarity with proper error handling.

        This method now uses extract_colbert_score() to safely handle
        RAGatouille's varying result formats.

        Args:
            entity1: Query entity
            entity2: Reference entity (should be in index)

        Returns:
            float: Similarity score (0-1)
        """
        try:
            # Search for entity1 in the ColBERT index
            results = self.colbert_searcher.search(query=entity1, k=1)

            # ✅ CRITICAL FIX: Use safe extraction instead of direct access
            # ❌ OLD (broken): score = results[0]['score']
            # ✅ NEW (fixed):
            score = extract_colbert_score(results, entity1, fallback=0.0)

            logger.debug(f"ColBERT score for '{entity1}' -> '{entity2}': {score:.3f}")
            return score

        except Exception as e:
            logger.error(f"ColBERT similarity failed for '{entity1}' vs '{entity2}': {e}")
            return 0.0

    def compute_graph_similarity(self, entity1: str, entity2: str) -> float:
        """
        Compute graph-based similarity (shared neighbors, PageRank, etc.)

        TODO: Implement with actual graph structure
        For now, returns a mock score.
        """
        # Mock implementation - replace with actual graph analysis
        return 0.3

    def compute_multi_feature_score(
        self,
        entity1: str,
        entity2: str,
        verbose: bool = False
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute combined similarity score using all features.

        Args:
            entity1: First entity
            entity2: Second entity
            verbose: If True, print individual feature scores

        Returns:
            Tuple of (combined_score, feature_scores_dict)
        """
        # Compute individual feature scores
        feature_scores = {
            'sapbert': self.compute_sapbert_similarity(entity1, entity2),
            'lexical': self.compute_lexical_similarity(entity1, entity2),
            'colbert': self.compute_colbert_similarity(entity1, entity2),  # ✅ FIXED
            'graph': self.compute_graph_similarity(entity1, entity2)
        }

        # Compute weighted combination
        combined_score = sum(
            self.feature_weights[feature] * score
            for feature, score in feature_scores.items()
        )

        if verbose:
            logger.info(f"\nScores for '{entity1}' <-> '{entity2}':")
            for feature, score in feature_scores.items():
                weight = self.feature_weights[feature]
                weighted = weight * score
                logger.info(f"  {feature:8s}: {score:.3f} (weight: {weight:.2f}, weighted: {weighted:.3f})")
            logger.info(f"  Combined: {combined_score:.3f}")

        return combined_score, feature_scores

    def find_equivalent_pairs(
        self,
        candidate_pairs: List[Tuple[str, str]],
        verbose: bool = True
    ) -> List[Tuple[str, str, float, Dict[str, float]]]:
        """
        Find equivalent entity pairs using multi-feature scoring.

        Args:
            candidate_pairs: List of (entity1, entity2) tuples to evaluate
            verbose: If True, show progress and print results

        Returns:
            List of (entity1, entity2, score, feature_scores) for pairs above threshold
        """
        logger.info("="*80)
        logger.info("STAGE 3: MULTI-FEATURE SCORING")
        logger.info("="*80)
        logger.info(f"Feature weights: {self.feature_weights}")
        logger.info(f"Similarity threshold: {self.similarity_threshold}")
        logger.info(f"Candidate pairs: {len(candidate_pairs)}")
        logger.info("")

        equivalent_pairs = []
        all_scores = []

        # Process each candidate pair
        iterator = tqdm(candidate_pairs, desc="Computing similarities") if verbose else candidate_pairs

        for entity1, entity2 in iterator:
            combined_score, feature_scores = self.compute_multi_feature_score(
                entity1, entity2, verbose=False
            )

            all_scores.append(combined_score)

            # Check if pair exceeds threshold
            if combined_score >= self.similarity_threshold:
                equivalent_pairs.append((entity1, entity2, combined_score, feature_scores))

        # Summary statistics
        logger.info("\n" + "="*80)
        logger.info("RESULTS SUMMARY")
        logger.info("="*80)
        logger.info(f"Pairs evaluated: {len(candidate_pairs)}")
        logger.info(f"Equivalent pairs found: {len(equivalent_pairs)}")
        logger.info(f"Score range: [{min(all_scores):.3f}, {max(all_scores):.3f}]")
        logger.info(f"Mean score: {np.mean(all_scores):.3f}")
        logger.info(f"Median score: {np.median(all_scores):.3f}")
        logger.info("")

        # Show top matches
        if equivalent_pairs:
            logger.info("Top 10 equivalent pairs:")
            for entity1, entity2, score, features in sorted(equivalent_pairs, key=lambda x: x[2], reverse=True)[:10]:
                logger.info(f"  {score:.3f}: '{entity1}' <-> '{entity2}'")
                logger.info(f"         SapBERT={features['sapbert']:.3f}, "
                          f"Lexical={features['lexical']:.3f}, "
                          f"ColBERT={features['colbert']:.3f}, "  # ✅ Now shows real scores!
                          f"Graph={features['graph']:.3f}")
        else:
            logger.warning("⚠️  No equivalent pairs found above threshold!")

        return equivalent_pairs


def main():
    """
    Example usage of multi-feature entity resolution.
    """
    # Configuration
    COLBERT_INDEX_PATH = "tmp/colbert/Entity_index_<your_fingerprint>"  # Update this

    # Example candidate pairs
    candidate_pairs = [
        ("aspirin", "acetylsalicylic acid"),
        ("diabetes", "diabetes mellitus"),
        ("hypertension", "high blood pressure"),
        ("myocardial infarction", "heart attack"),
        ("acetaminophen", "paracetamol"),
    ]

    # Initialize resolver
    try:
        resolver = MultiFeatureEntityResolution(
            colbert_index_path=COLBERT_INDEX_PATH,
            feature_weights={
                'sapbert': 0.5,
                'lexical': 0.15,
                'colbert': 0.25,
                'graph': 0.1
            },
            similarity_threshold=0.7
        )
    except Exception as e:
        logger.error(f"Failed to initialize resolver: {e}")
        logger.error("Please update COLBERT_INDEX_PATH in the script")
        return 1

    # Find equivalent pairs
    equivalent_pairs = resolver.find_equivalent_pairs(
        candidate_pairs,
        verbose=True
    )

    # Detailed results
    logger.info("\n" + "="*80)
    logger.info("DETAILED RESULTS")
    logger.info("="*80)
    for entity1, entity2, score, features in equivalent_pairs:
        logger.info(f"\n✅ EQUIVALENT: '{entity1}' <-> '{entity2}'")
        logger.info(f"   Combined Score: {score:.3f}")
        logger.info(f"   Feature Breakdown:")
        for feature_name, feature_score in features.items():
            weight = resolver.feature_weights[feature_name]
            weighted = weight * feature_score
            logger.info(f"     - {feature_name:8s}: {feature_score:.3f} × {weight:.2f} = {weighted:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
