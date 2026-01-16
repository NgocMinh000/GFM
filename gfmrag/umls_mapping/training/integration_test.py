"""
Integration Testing for Fine-Tuned Cross-Encoder Pipeline

Tests the complete pipeline integration:
1. Load fine-tuned model
2. Load calibrator
3. Load adaptive thresholds
4. Test end-to-end UMLS mapping with improvements

Usage:
    python -m gfmrag.umls_mapping.training.integration_test \
        --model_path models/cross_encoder_finetuned/checkpoint-best \
        --test_entities "diabetic neuropathy,myocardial infarction,hypertension"
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from gfmrag.umls_mapping.umls_loader import UMLSLoader
from gfmrag.umls_mapping.cross_encoder_finetuned import FineTunedCrossEncoderReranker
from gfmrag.umls_mapping.hard_negative_filter import FilteredCandidate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class IntegrationTester:
    """
    Tests fine-tuned cross-encoder integration.
    """

    def __init__(
        self,
        model_path: str,
        calibrator_path: str = None,
        threshold_tuner_path: str = None,
    ):
        """
        Initialize integration tester.

        Args:
            model_path: Path to fine-tuned model
            calibrator_path: Path to calibrator
            threshold_tuner_path: Path to threshold tuner
        """
        self.model_path = model_path
        self.calibrator_path = calibrator_path
        self.threshold_tuner_path = threshold_tuner_path

        # Components
        self.reranker = None
        self.umls_loader = None

    def setup(self):
        """Setup components."""
        logger.info("Setting up integration test...")

        # Load UMLS
        logger.info("Loading UMLS...")
        self.umls_loader = UMLSLoader()

        # Load fine-tuned reranker
        logger.info("Loading fine-tuned reranker...")
        self.reranker = FineTunedCrossEncoderReranker(
            model_path=self.model_path,
            calibrator_path=self.calibrator_path,
            threshold_tuner_path=self.threshold_tuner_path,
            device="cuda",
            use_calibration=True,
            use_adaptive_thresholds=True,
        )

        # Load components (test that they exist)
        self.reranker._load_components()

        logger.info("Setup complete!")
        logger.info("")

        # Print statistics
        stats = self.reranker.get_statistics()
        self._print_statistics(stats)

    def test_single_entity(
        self,
        entity: str,
        top_k: int = 10,
    ) -> Dict:
        """
        Test mapping for a single entity.

        Args:
            entity: Entity text
            top_k: Number of top candidates to return

        Returns:
            Test results
        """
        logger.info(f"Testing entity: '{entity}'")

        # Step 1: Get candidate CUIs (simplified - in real pipeline this comes from earlier stages)
        # For testing, we'll search UMLS directly
        candidates = self._get_test_candidates(entity, k=20)

        if not candidates:
            logger.warning(f"No candidates found for '{entity}'")
            return {
                "entity": entity,
                "candidates": [],
                "top_cui": None,
                "top_score": 0.0,
            }

        # Step 2: Rerank with fine-tuned cross-encoder
        reranked = self.reranker.rerank(
            entity=entity,
            candidates=candidates,
            entity_type=None,  # Would come from KG in real pipeline
            return_all=True,
        )

        # Step 3: Analyze results
        if reranked:
            top_candidate = reranked[0]
            score_margin = self.reranker.compute_score_margin(reranked)
            confidence_tier = self.reranker.get_confidence_tier(
                top_candidate.cross_encoder_score
            )

            logger.info(f"  Top match: {top_candidate.name} (CUI: {top_candidate.cui})")
            logger.info(f"  Score: {top_candidate.cross_encoder_score:.4f}")
            logger.info(f"  Score margin: {score_margin:.4f}")
            logger.info(f"  Confidence: {confidence_tier}")
        else:
            logger.info(f"  No candidates passed threshold")

        return {
            "entity": entity,
            "candidates": [
                {
                    "cui": c.cui,
                    "name": c.name,
                    "score": c.cross_encoder_score,
                    "rank": i + 1,
                }
                for i, c in enumerate(reranked[:top_k])
            ],
            "top_cui": reranked[0].cui if reranked else None,
            "top_score": reranked[0].cross_encoder_score if reranked else 0.0,
            "score_margin": self.reranker.compute_score_margin(reranked) if len(reranked) >= 2 else 0.0,
            "num_above_threshold": len(reranked),
        }

    def test_batch_entities(
        self,
        entities: List[str],
    ) -> Dict:
        """
        Test mapping for multiple entities.

        Args:
            entities: List of entity texts

        Returns:
            Batch test results
        """
        logger.info(f"Testing {len(entities)} entities...")
        logger.info("")

        results = []
        for entity in tqdm(entities, desc="Testing entities"):
            result = self.test_single_entity(entity, top_k=5)
            results.append(result)

        # Aggregate statistics
        scores = [r["top_score"] for r in results if r["top_cui"]]
        margins = [r["score_margin"] for r in results if r["score_margin"] > 0]
        high_confidence = [r for r in results if r["top_score"] >= 0.7]

        stats = {
            "num_entities": len(entities),
            "num_mapped": len([r for r in results if r["top_cui"]]),
            "num_unmapped": len([r for r in results if not r["top_cui"]]),
            "avg_score": np.mean(scores) if scores else 0.0,
            "median_score": np.median(scores) if scores else 0.0,
            "avg_margin": np.mean(margins) if margins else 0.0,
            "median_margin": np.median(margins) if margins else 0.0,
            "high_confidence_pct": len(high_confidence) / len(entities) * 100 if entities else 0.0,
            "results": results,
        }

        self._print_batch_statistics(stats)

        return stats

    def test_improvements(
        self,
        test_entities: List[str],
    ) -> Dict:
        """
        Compare fine-tuned vs baseline performance.

        Args:
            test_entities: List of test entities

        Returns:
            Comparison results
        """
        logger.info("Testing improvements over baseline...")

        # Test with fine-tuned model
        logger.info("\n=== Fine-Tuned Model ===")
        finetuned_results = self.test_batch_entities(test_entities)

        # TODO: Compare with baseline (would need to load baseline model)
        # For now, just show expected improvements

        logger.info("")
        logger.info("=" * 80)
        logger.info("EXPECTED IMPROVEMENTS (from Phase 2 Plan)")
        logger.info("=" * 80)
        logger.info("High Confidence Mappings:")
        logger.info("  Baseline:   5-10%")
        logger.info("  Fine-Tuned: 40-60%")
        logger.info("  Improvement: +30-50%")
        logger.info("")
        logger.info("Cross-Encoder Score:")
        logger.info("  Baseline:   0.58")
        logger.info("  Fine-Tuned: 0.85+")
        logger.info("  Improvement: +47%")
        logger.info("")
        logger.info("Score Margin:")
        logger.info("  Baseline:   0.10-0.12")
        logger.info("  Fine-Tuned: 0.25+")
        logger.info("  Improvement: +100-150%")
        logger.info("=" * 80)
        logger.info("")

        logger.info("ACTUAL RESULTS (Fine-Tuned):")
        logger.info(f"  Avg Score:      {finetuned_results['avg_score']:.4f}")
        logger.info(f"  Avg Margin:     {finetuned_results['avg_margin']:.4f}")
        logger.info(f"  High Conf %:    {finetuned_results['high_confidence_pct']:.1f}%")
        logger.info("")

        return {
            "finetuned": finetuned_results,
        }

    def _get_test_candidates(
        self,
        entity: str,
        k: int = 20,
    ) -> List[FilteredCandidate]:
        """
        Get test candidates for an entity.

        Simplified version - in real pipeline this comes from CandidateGenerator.

        Args:
            entity: Entity text
            k: Number of candidates

        Returns:
            List of FilteredCandidate objects
        """
        # Simple fuzzy search in UMLS
        entity_lower = entity.lower()

        matches = []
        for cui, concept in self.umls_loader.concepts.items():
            # Check if entity appears in concept names
            for name in [concept.preferred_name] + concept.synonyms[:5]:
                if entity_lower in name.lower() or name.lower() in entity_lower:
                    matches.append((cui, concept.preferred_name, 0.8))
                    break

        # Take top k
        matches = matches[:k]

        # Convert to FilteredCandidate
        candidates = [
            FilteredCandidate(
                cui=cui,
                name=name,
                score=score,
                previous_score=score,
                filter_reason="test_candidate",
            )
            for cui, name, score in matches
        ]

        return candidates

    def _print_statistics(self, stats: Dict):
        """Print reranker statistics."""
        logger.info("=" * 80)
        logger.info("RERANKER STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Model loaded:              {stats['model_loaded']}")
        logger.info(f"Calibration enabled:       {stats['calibration_enabled']}")
        logger.info(f"Calibrator loaded:         {stats['calibrator_loaded']}")
        logger.info(f"Adaptive thresholds:       {stats['adaptive_thresholds_enabled']}")
        logger.info(f"Threshold tuner loaded:    {stats['threshold_tuner_loaded']}")
        logger.info(f"Device:                    {stats['device']}")

        if "threshold_statistics" in stats:
            ts = stats["threshold_statistics"]
            logger.info("")
            logger.info("Threshold Statistics:")
            logger.info(f"  Num types:        {ts.get('num_types', 0)}")
            logger.info(f"  Num tuned:        {ts.get('num_tuned', 0)}")
            logger.info(f"  Threshold range:  {ts.get('threshold_min', 0):.3f} - {ts.get('threshold_max', 0):.3f}")
            logger.info(f"  Threshold median: {ts.get('threshold_median', 0):.3f}")

        logger.info("=" * 80)
        logger.info("")

    def _print_batch_statistics(self, stats: Dict):
        """Print batch test statistics."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("BATCH TEST RESULTS")
        logger.info("=" * 80)
        logger.info(f"Entities tested:         {stats['num_entities']}")
        logger.info(f"Successfully mapped:     {stats['num_mapped']} ({stats['num_mapped']/stats['num_entities']*100:.1f}%)")
        logger.info(f"Failed to map:           {stats['num_unmapped']} ({stats['num_unmapped']/stats['num_entities']*100:.1f}%)")
        logger.info("")
        logger.info(f"Average score:           {stats['avg_score']:.4f}")
        logger.info(f"Median score:            {stats['median_score']:.4f}")
        logger.info(f"Average margin:          {stats['avg_margin']:.4f}")
        logger.info(f"Median margin:           {stats['median_margin']:.4f}")
        logger.info(f"High confidence (≥0.7):  {stats['high_confidence_pct']:.1f}%")
        logger.info("=" * 80)
        logger.info("")


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test fine-tuned cross-encoder integration")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--calibrator_path", type=str, default=None, help="Path to calibrator")
    parser.add_argument("--threshold_tuner_path", type=str, default=None, help="Path to threshold tuner")
    parser.add_argument("--test_entities", type=str, default=None,
                       help="Comma-separated list of test entities")
    parser.add_argument("--test_file", type=str, default=None,
                       help="File with test entities (one per line)")
    parser.add_argument("--output", type=str, default="results/integration_test.json",
                       help="Output file for results")
    args = parser.parse_args()

    # Get test entities
    if args.test_entities:
        test_entities = [e.strip() for e in args.test_entities.split(",")]
    elif args.test_file:
        with open(args.test_file) as f:
            test_entities = [line.strip() for line in f if line.strip()]
    else:
        # Default test entities
        test_entities = [
            "diabetic neuropathy",
            "myocardial infarction",
            "hypertension",
            "type 2 diabetes mellitus",
            "chronic kidney disease",
            "rheumatoid arthritis",
            "alzheimer disease",
            "parkinson disease",
            "breast cancer",
            "lung cancer",
        ]

    logger.info(f"Testing {len(test_entities)} entities")

    # Initialize tester
    tester = IntegrationTester(
        model_path=args.model_path,
        calibrator_path=args.calibrator_path,
        threshold_tuner_path=args.threshold_tuner_path,
    )

    # Setup
    tester.setup()

    # Run tests
    results = tester.test_improvements(test_entities)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
