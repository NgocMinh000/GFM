"""
Test ColBERT Pairwise Similarity
=================================

Kiểm tra khả năng tính độ tương đồng giữa 2 thực thể của ColBERT.

Usage:
    python test_colbert_similarity.py
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from gfmrag.kg_construction.entity_linking_model import ColbertELModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_colbert_pairwise_similarity():
    """Test ColBERT pairwise similarity computation"""

    print("="*80)
    print("TEST: ColBERT Pairwise Similarity")
    print("="*80)

    # Test cases: (entity1, entity2, expected_high_similarity)
    test_cases = [
        ("diabetes", "diabetes mellitus", True),
        ("MI", "myocardial infarction", True),
        ("aspirin", "acetylsalicylic acid", True),
        ("hypertension", "high blood pressure", True),
        ("cancer", "diabetes", False),
        ("headache", "heart attack", False),
    ]

    print("\n1. Initializing ColBERT model...")
    print("-" * 80)

    try:
        model = ColbertELModel(
            model_name_or_path="colbert-ir/colbertv2.0",
            root="tmp/colbert_test",
            force=False,
        )
        print("✅ Model initialized successfully")
        print(f"   Model: {model.model_name_or_path}")
        print(f"   Root: {model.root}")
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        return False

    print("\n2. Testing pairwise similarity computation...")
    print("-" * 80)

    results = []

    for i, (entity1, entity2, expected_high) in enumerate(test_cases, 1):
        print(f"\nTest {i}: '{entity1}' vs '{entity2}'")
        print(f"  Expected: {'HIGH' if expected_high else 'LOW'} similarity")

        try:
            # Compute similarity
            score = model.compute_pairwise_similarity(entity1, entity2)

            # Check result type
            print(f"  Result type: {type(score)}")
            print(f"  Result value: {score}")

            # Validate result
            if not isinstance(score, (int, float)):
                print(f"  ❌ INVALID TYPE: Expected float/int, got {type(score)}")
                results.append({
                    "test": i,
                    "entity1": entity1,
                    "entity2": entity2,
                    "status": "FAIL",
                    "reason": f"Invalid type: {type(score)}",
                    "score": None
                })
                continue

            if score < 0 or score > 1:
                print(f"  ⚠️  WARNING: Score out of range [0, 1]: {score}")

            # Check if result matches expectation
            threshold = 0.5
            is_high = score >= threshold
            matches = is_high == expected_high

            if matches:
                print(f"  ✅ PASS: Score = {score:.4f} (expected {'high' if expected_high else 'low'})")
                results.append({
                    "test": i,
                    "entity1": entity1,
                    "entity2": entity2,
                    "status": "PASS",
                    "score": score
                })
            else:
                print(f"  ⚠️  UNEXPECTED: Score = {score:.4f} (expected {'high' if expected_high else 'low'})")
                results.append({
                    "test": i,
                    "entity1": entity1,
                    "entity2": entity2,
                    "status": "UNEXPECTED",
                    "score": score
                })

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "test": i,
                "entity1": entity1,
                "entity2": entity2,
                "status": "ERROR",
                "reason": str(e),
                "score": None
            })

    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    unexpected = sum(1 for r in results if r["status"] == "UNEXPECTED")

    print(f"Total tests: {len(results)}")
    print(f"✅ Passed: {passed}")
    print(f"⚠️  Unexpected: {unexpected}")
    print(f"❌ Failed: {failed}")
    print(f"💥 Errors: {errors}")

    if failed > 0 or errors > 0:
        print("\n❌ ISSUES FOUND:")
        for r in results:
            if r["status"] in ["FAIL", "ERROR"]:
                print(f"  - Test {r['test']}: {r['entity1']} vs {r['entity2']}")
                print(f"    Status: {r['status']}")
                if "reason" in r:
                    print(f"    Reason: {r['reason']}")

    print("\n" + "="*80)

    return failed == 0 and errors == 0


def test_colbert_index_and_search():
    """Test ColBERT indexing and search"""

    print("\n" + "="*80)
    print("TEST: ColBERT Indexing and Search")
    print("="*80)

    print("\n1. Initializing ColBERT model...")

    try:
        model = ColbertELModel(
            model_name_or_path="colbert-ir/colbertv2.0",
            root="tmp/colbert_test",
            force=True,  # Force rebuild index
        )
        print("✅ Model initialized")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return False

    print("\n2. Indexing test entities...")

    test_entities = [
        "diabetes",
        "diabetes mellitus",
        "type 2 diabetes",
        "hypertension",
        "high blood pressure",
        "aspirin",
        "acetylsalicylic acid",
    ]

    try:
        model.index(test_entities)
        print(f"✅ Indexed {len(test_entities)} entities")
    except Exception as e:
        print(f"❌ Failed to index: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n3. Testing search functionality...")

    queries = ["diabetes", "blood pressure", "aspirin"]

    for query in queries:
        print(f"\nQuery: '{query}'")

        try:
            results = model([query], topk=3)

            print(f"  Results type: {type(results)}")
            print(f"  Results keys: {list(results.keys()) if isinstance(results, dict) else 'N/A'}")

            if query.lower() in results or query in results:
                query_key = query.lower() if query.lower() in results else query
                query_results = results[query_key]

                print(f"  Found {len(query_results)} results:")
                for i, res in enumerate(query_results, 1):
                    print(f"    {i}. Type: {type(res)}")
                    if isinstance(res, dict):
                        print(f"       Keys: {list(res.keys())}")
                        print(f"       Entity: {res.get('entity', 'N/A')}")
                        print(f"       Score: {res.get('score', 'N/A')}")
                    else:
                        print(f"       Value: {res}")
            else:
                print(f"  ⚠️  Query not found in results")
                print(f"  Available keys: {list(results.keys())}")

        except Exception as e:
            print(f"  ❌ Search failed: {e}")
            import traceback
            traceback.print_exc()

    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("ColBERT Similarity Testing Suite")
    print("="*80)

    success = True

    # Test 1: Pairwise similarity
    if not test_colbert_pairwise_similarity():
        success = False

    # Test 2: Indexing and search
    if not test_colbert_index_and_search():
        success = False

    print("\n" + "="*80)
    if success:
        print("✅ ALL TESTS COMPLETED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*80)
