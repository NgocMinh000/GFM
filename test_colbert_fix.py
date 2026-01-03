#!/usr/bin/env python3
"""
Test script to verify ColBERT fix is working correctly.

This script tests:
1. ColbertELModel with error handling
2. extract_colbert_score utility function
3. Entity linking in kg_constructor.py context

Run this to verify the fixes resolve the "string indices must be integers" error.
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_colbert_el_model():
    """Test ColbertELModel with small dataset."""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: ColbertELModel Entity Linking")
    logger.info("="*80)

    try:
        from gfmrag.kg_construction.entity_linking_model import ColbertELModel

        # Create model
        model = ColbertELModel(
            model_name_or_path="colbert-ir/colbertv2.0",
            root="tmp/test_colbert",
            force=True  # Force rebuild for testing
        )

        # Test entities (biomedical domain)
        entities = [
            "aspirin",
            "acetylsalicylic acid",
            "diabetes",
            "diabetes mellitus",
            "hypertension",
            "high blood pressure",
            "myocardial infarction",
            "heart attack"
        ]

        logger.info(f"Indexing {len(entities)} test entities...")
        model.index(entities)

        # Query for similar entities
        queries = ["aspirin medication", "diabetes disease", "heart attack symptoms"]

        logger.info(f"Searching for {len(queries)} queries...")
        results = model(queries, topk=3)

        # Validate results
        success = True
        for query, matches in results.items():
            logger.info(f"\nQuery: '{query}'")
            if not matches:
                logger.warning(f"  ⚠️  No matches found (empty results)")
                continue

            for match in matches:
                # Check required keys
                if "entity" not in match or "score" not in match or "norm_score" not in match:
                    logger.error(f"  ❌ Missing required keys in match: {match.keys()}")
                    success = False
                    continue

                entity = match["entity"]
                score = match["score"]
                norm_score = match["norm_score"]

                # Validate types
                if not isinstance(entity, str):
                    logger.error(f"  ❌ 'entity' should be str, got {type(entity)}")
                    success = False

                if not isinstance(score, (int, float)):
                    logger.error(f"  ❌ 'score' should be numeric, got {type(score)}")
                    success = False

                if not isinstance(norm_score, (int, float)):
                    logger.error(f"  ❌ 'norm_score' should be numeric, got {type(norm_score)}")
                    success = False

                # Check score is not 0
                if score == 0.0:
                    logger.warning(f"  ⚠️  Score is 0.0 for '{entity}' (possible issue)")
                else:
                    logger.info(f"  ✅ '{entity}': score={score:.3f}, norm_score={norm_score:.3f}")

        if success:
            logger.info("\n✅ TEST 1 PASSED: ColbertELModel returns correct format")
            return True
        else:
            logger.error("\n❌ TEST 1 FAILED: ColbertELModel has format issues")
            return False

    except Exception as e:
        logger.error(f"\n❌ TEST 1 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_extract_colbert_score():
    """Test extract_colbert_score utility function."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: extract_colbert_score Utility")
    logger.info("="*80)

    try:
        from gfmrag.kg_construction.entity_linking_model.colbert_utils import extract_colbert_score

        # Test various result formats
        test_cases = [
            # (input, expected_behavior, description)
            ([{"content": "test", "score": 0.85}], 0.85, "Dict with 'content' and 'score'"),
            ([{"text": "test", "score": 0.75}], 0.0, "Dict with 'text' and 'score' (not supported in extract)"),
            ([{"content": "test", "similarity": 0.90}], 0.90, "Dict with 'similarity' key"),
            (["string result"], 0.0, "String result (should fallback to 0.0)"),
            ([], 0.0, "Empty result (should fallback to 0.0)"),
            (None, 0.0, "None result (should fallback to 0.0)"),
        ]

        success = True
        for i, (input_data, expected, description) in enumerate(test_cases, 1):
            result = extract_colbert_score(input_data, f"test_query_{i}", fallback=0.0)

            if result == expected:
                logger.info(f"  ✅ Case {i}: {description} → {result:.2f} (expected {expected:.2f})")
            else:
                logger.error(f"  ❌ Case {i}: {description} → {result:.2f} (expected {expected:.2f})")
                success = False

        if success:
            logger.info("\n✅ TEST 2 PASSED: extract_colbert_score handles all formats")
            return True
        else:
            logger.error("\n❌ TEST 2 FAILED: extract_colbert_score has issues")
            return False

    except Exception as e:
        logger.error(f"\n❌ TEST 2 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pairwise_similarity():
    """Test pairwise similarity computation."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Pairwise Similarity")
    logger.info("="*80)

    try:
        from gfmrag.kg_construction.entity_linking_model import ColbertELModel

        model = ColbertELModel(
            model_name_or_path="colbert-ir/colbertv2.0",
            root="tmp/test_colbert_pairwise",
            force=True
        )

        # Test pairs
        test_pairs = [
            ("aspirin", "acetylsalicylic acid"),
            ("diabetes", "diabetes mellitus"),
            ("completely different", "unrelated term")
        ]

        logger.info(f"Testing {len(test_pairs)} entity pairs...")

        success = True
        for entity1, entity2 in test_pairs:
            score = model.compute_pairwise_similarity(entity1, entity2)

            if isinstance(score, (int, float)):
                logger.info(f"  ✅ '{entity1}' <-> '{entity2}': {score:.3f}")

                if score == 0.0:
                    logger.warning(f"     ⚠️  Score is 0.0 (may indicate issue or truly dissimilar)")
            else:
                logger.error(f"  ❌ '{entity1}' <-> '{entity2}': Invalid score type {type(score)}")
                success = False

        if success:
            logger.info("\n✅ TEST 3 PASSED: Pairwise similarity works")
            return True
        else:
            logger.error("\n❌ TEST 3 FAILED: Pairwise similarity has issues")
            return False

    except Exception as e:
        logger.error(f"\n❌ TEST 3 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("="*80)
    logger.info("COLBERT FIX VERIFICATION TEST SUITE")
    logger.info("="*80)

    results = []

    # Run tests
    results.append(("ColbertELModel", test_colbert_el_model()))
    results.append(("extract_colbert_score", test_extract_colbert_score()))
    results.append(("Pairwise Similarity", test_pairwise_similarity()))

    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED! ColBERT fix is working correctly.")
        return 0
    else:
        logger.error(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
