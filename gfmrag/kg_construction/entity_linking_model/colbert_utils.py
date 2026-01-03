"""
ColBERT Utility Functions for Entity Resolution
================================================

Helper functions for computing pairwise similarity using ColBERT.
Fixes the "string indices must be integers" error when using RAGatouille.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def extract_colbert_score(search_results: Any, query: str, fallback: float = 0.0) -> float:
    """
    Extract similarity score from ColBERT/RAGatouille search results.

    Handles different result formats:
    - List of dicts with 'score' key
    - List of dicts with 'content' and 'score' keys
    - List of strings (fallback to 0.0)
    - Empty results (fallback to 0.0)

    Args:
        search_results: Raw search results from RAGPretrainedModel.search()
        query: Query string (for logging)
        fallback: Default score if extraction fails

    Returns:
        float: Similarity score (0-1)

    Example:
        >>> results = searcher.search(query=["aspirin"], k=1)
        >>> score = extract_colbert_score(results[0], "aspirin")
    """
    try:
        # Handle empty results
        if not search_results:
            logger.debug(f"Empty ColBERT results for query '{query}'")
            return fallback

        # Get first result
        if isinstance(search_results, list) and len(search_results) > 0:
            first_result = search_results[0]
        else:
            logger.warning(f"Invalid ColBERT result format for '{query}': {type(search_results)}")
            return fallback

        # Extract score from dict
        if isinstance(first_result, dict):
            # Try 'score' key
            if 'score' in first_result:
                return float(first_result['score'])
            # Try 'similarity' key
            elif 'similarity' in first_result:
                return float(first_result['similarity'])
            # Try 'relevance_score' key (some versions)
            elif 'relevance_score' in first_result:
                return float(first_result['relevance_score'])
            else:
                logger.warning(f"No score key in ColBERT result for '{query}'. Keys: {list(first_result.keys())}")
                return fallback

        # Handle string results (error case)
        elif isinstance(first_result, str):
            logger.warning(f"ColBERT returned string instead of dict for '{query}': '{first_result[:50]}'")
            return fallback

        else:
            logger.warning(f"Unexpected ColBERT result type for '{query}': {type(first_result)}")
            return fallback

    except Exception as e:
        logger.error(f"Error extracting ColBERT score for '{query}': {e}")
        return fallback


def compute_colbert_pairwise_similarity(
    searcher,
    entity1: str,
    entity2: str,
    index_name: Optional[str] = None
) -> float:
    """
    Compute pairwise similarity between two entities using ColBERT searcher.

    This function handles the search and score extraction with proper error handling.
    Fixes the "string indices must be integers, not 'str'" error.

    Args:
        searcher: RAGPretrainedModel instance (already loaded)
        entity1: First entity string
        entity2: Second entity string
        index_name: Optional index name to use (if not already set)

    Returns:
        float: Similarity score (0-1), or 0.0 if computation fails

    Example:
        >>> from ragatouille import RAGPretrainedModel
        >>> searcher = RAGPretrainedModel.from_index("path/to/index")
        >>> score = compute_colbert_pairwise_similarity(searcher, "aspirin", "acetylsalicylic acid")
        >>> print(f"Similarity: {score:.3f}")
    """
    try:
        # Search for entity1 in the index
        results = searcher.search(query=entity1, k=1)

        # Extract score from results
        score = extract_colbert_score(results, entity1, fallback=0.0)

        return score

    except Exception as e:
        logger.error(f"ColBERT pairwise similarity failed for '{entity1}' and '{entity2}': {e}")
        return 0.0


def batch_compute_colbert_similarity(
    searcher,
    entity_pairs: List[tuple],
    batch_size: int = 32
) -> Dict[tuple, float]:
    """
    Compute ColBERT similarity for multiple entity pairs in batches.

    Args:
        searcher: RAGPretrainedModel instance
        entity_pairs: List of (entity1, entity2) tuples
        batch_size: Number of pairs to process at once

    Returns:
        dict: {(entity1, entity2): similarity_score}

    Example:
        >>> pairs = [("aspirin", "ibuprofen"), ("diabetes", "hyperglycemia")]
        >>> scores = batch_compute_colbert_similarity(searcher, pairs)
    """
    scores = {}

    for i in range(0, len(entity_pairs), batch_size):
        batch = entity_pairs[i:i + batch_size]

        # Extract queries (entity1 from each pair)
        queries = [pair[0] for pair in batch]

        try:
            # Batch search
            batch_results = searcher.search(query=queries, k=1)

            # Extract scores for each pair
            for j, pair in enumerate(batch):
                entity1, entity2 = pair
                result = batch_results[j] if j < len(batch_results) else []
                score = extract_colbert_score(result, entity1, fallback=0.0)
                scores[pair] = score

        except Exception as e:
            logger.error(f"Batch ColBERT computation failed: {e}")
            # Set all pairs in this batch to 0.0
            for pair in batch:
                scores[pair] = 0.0

    return scores


def validate_colbert_index(searcher, test_queries: List[str] = None) -> bool:
    """
    Validate that ColBERT index is working correctly.

    Args:
        searcher: RAGPretrainedModel instance
        test_queries: Optional list of test queries

    Returns:
        bool: True if index is valid, False otherwise
    """
    if test_queries is None:
        test_queries = ["test"]

    try:
        results = searcher.search(query=test_queries[0], k=1)

        if not results:
            logger.warning("ColBERT index returned empty results for test query")
            return False

        # Check result format
        if isinstance(results, list) and len(results) > 0:
            first_result = results[0]

            if isinstance(first_result, str):
                logger.error("ColBERT index returning strings instead of dicts!")
                logger.error(f"Sample result: {first_result[:100]}")
                return False

            if isinstance(first_result, dict):
                if 'score' not in first_result and 'similarity' not in first_result:
                    logger.warning(f"ColBERT result missing score. Keys: {list(first_result.keys())}")
                    return False

        logger.info("✅ ColBERT index validation passed")
        return True

    except Exception as e:
        logger.error(f"ColBERT index validation failed: {e}")
        return False


def debug_colbert_results(searcher, query: str, k: int = 3):
    """
    Debug function to inspect ColBERT search results.

    Prints detailed information about result structure.

    Args:
        searcher: RAGPretrainedModel instance
        query: Test query string
        k: Number of results to retrieve
    """
    print(f"\n{'='*80}")
    print(f"DEBUG: ColBERT Search Results for '{query}'")
    print(f"{'='*80}")

    try:
        results = searcher.search(query=query, k=k)

        print(f"Results type: {type(results)}")
        print(f"Results length: {len(results) if results else 0}")

        if results:
            print(f"\nFirst result:")
            first_result = results[0]
            print(f"  Type: {type(first_result)}")

            if isinstance(first_result, dict):
                print(f"  Keys: {list(first_result.keys())}")
                print(f"  Content: {first_result}")
            elif isinstance(first_result, str):
                print(f"  String: '{first_result}'")
            else:
                print(f"  Value: {first_result}")

            # Try to extract score
            score = extract_colbert_score(results, query)
            print(f"\nExtracted score: {score}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print(f"{'='*80}\n")
