"""
Safe ColBERT Search Wrapper - Fixes "string indices must be integers" error

This module provides a safe wrapper around RAGatouille search that handles
all possible result formats and prevents crashes.

Usage:
    from gfmrag.kg_construction.entity_linking_model.safe_colbert import safe_colbert_search

    results = safe_colbert_search(searcher, query="diabetes", k=5)
    # Results are guaranteed to be list of dicts with 'content' and 'score' keys
"""

import logging
from typing import Any, List, Dict, Union

logger = logging.getLogger(__name__)


def safe_colbert_search(
    searcher: Any,
    query: Union[str, List[str]],
    k: int = 10,
    return_documents: bool = True
) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """
    Safe wrapper around RAGatouille search that handles all result formats.

    This function wraps searcher.search() and ensures the results are always
    in a consistent format: list of dicts with 'content' and 'score' keys.

    Args:
        searcher: RAGPretrainedModel instance
        query: Single query string or list of query strings
        k: Number of results to return
        return_documents: Whether to return document content (default: True)

    Returns:
        If query is str: List[Dict] with keys 'content' and 'score'
        If query is List[str]: List[List[Dict]] for each query

    Examples:
        >>> searcher = RAGPretrainedModel.from_index("path/to/index")
        >>>
        >>> # Single query
        >>> results = safe_colbert_search(searcher, "diabetes", k=3)
        >>> print(results[0]['score'])  # Safe!
        >>>
        >>> # Multiple queries
        >>> results = safe_colbert_search(searcher, ["diabetes", "aspirin"], k=3)
        >>> print(results[0][0]['score'])  # Safe!
    """
    # Normalize query to list
    single_query = isinstance(query, str)
    queries = [query] if single_query else query

    try:
        # Call RAGatouille search
        raw_results = searcher.search(queries, k=k)

    except Exception as e:
        logger.error(f"RAGatouille search failed: {e}")
        # Return empty results
        empty = []
        return empty if single_query else [empty for _ in queries]

    # Process and validate results
    processed_results = []

    for i, query_str in enumerate(queries):
        query_results = []

        try:
            raw_result = raw_results[i] if i < len(raw_results) else []

            if not raw_result:
                logger.debug(f"Empty results for query: '{query_str}'")
                processed_results.append(query_results)
                continue

            # Validate and convert each result
            for j, item in enumerate(raw_result):
                try:
                    validated_item = _validate_and_convert_result(item, query_str, j)
                    if validated_item:
                        query_results.append(validated_item)
                except Exception as e:
                    logger.warning(f"Failed to process result {j} for query '{query_str}': {e}")
                    continue

            if not query_results:
                logger.warning(f"No valid results after processing for query: '{query_str}'")

        except Exception as e:
            logger.error(f"Error processing results for query '{query_str}': {e}")

        processed_results.append(query_results)

    # Return single list if single query, otherwise list of lists
    return processed_results[0] if single_query else processed_results


def _validate_and_convert_result(
    item: Any,
    query: str,
    index: int
) -> Union[Dict[str, Any], None]:
    """
    Validate and convert a single search result item to standard format.

    Args:
        item: Raw result item from RAGatouille
        query: Query string (for logging)
        index: Result index (for logging)

    Returns:
        Dict with 'content' and 'score' keys, or None if invalid
    """
    # Case 1: Already a dict
    if isinstance(item, dict):
        # Sub-case 1a: Has 'content' and 'score'
        if 'content' in item and 'score' in item:
            return {
                'content': str(item['content']),
                'score': float(item['score'])
            }

        # Sub-case 1b: Has 'text' and 'score'
        elif 'text' in item and 'score' in item:
            return {
                'content': str(item['text']),
                'score': float(item['score'])
            }

        # Sub-case 1c: Has 'content' and 'similarity'
        elif 'content' in item and 'similarity' in item:
            return {
                'content': str(item['content']),
                'score': float(item['similarity'])
            }

        # Sub-case 1d: Has 'text' and 'similarity'
        elif 'text' in item and 'similarity' in item:
            return {
                'content': str(item['text']),
                'score': float(item['similarity'])
            }

        # Sub-case 1e: Dict but missing required keys
        else:
            logger.warning(
                f"Result dict for query '{query}' at index {index} "
                f"has unexpected keys: {list(item.keys())}"
            )
            # Try to extract content and score from any available keys
            content = item.get('content') or item.get('text') or item.get('document') or str(item)
            score = item.get('score') or item.get('similarity') or item.get('relevance_score') or 0.0

            return {
                'content': str(content),
                'score': float(score)
            }

    # Case 2: String result (RAGatouille sometimes returns strings)
    elif isinstance(item, str):
        logger.warning(
            f"Result for query '{query}' at index {index} is a string instead of dict. "
            "This is a known RAGatouille issue. Assigning default score of 0.0."
        )
        return {
            'content': item,
            'score': 0.0  # No score available for string results
        }

    # Case 3: Tuple result (document, score)
    elif isinstance(item, (tuple, list)) and len(item) >= 2:
        return {
            'content': str(item[0]),
            'score': float(item[1])
        }

    # Case 4: Unexpected type
    else:
        logger.warning(
            f"Result for query '{query}' at index {index} has unexpected type: {type(item)}. "
            f"Value: {str(item)[:100]}"
        )
        return None


def safe_colbert_pairwise_similarity(
    searcher: Any,
    entity1: str,
    entity2: str,
    reindex: bool = True
) -> float:
    """
    Safely compute pairwise similarity between two entities using ColBERT.

    This function indexes entity2 and searches for entity1, returning the similarity score.

    Args:
        searcher: RAGPretrainedModel instance
        entity1: Query entity
        entity2: Reference entity to index
        reindex: Whether to reindex entity2 (default: True)

    Returns:
        Similarity score (0.0-1.0)

    Example:
        >>> searcher = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        >>> score = safe_colbert_pairwise_similarity(
        ...     searcher, "aspirin", "acetylsalicylic acid"
        ... )
        >>> print(f"Similarity: {score:.3f}")
    """
    try:
        if reindex:
            # Index entity2
            searcher.index(
                collection=[entity2],
                index_name=f"temp_pairwise_{hash(entity2)}",
                max_document_length=256,
                split_documents=False
            )

        # Search for entity1
        results = safe_colbert_search(searcher, entity1, k=1)

        if results and len(results) > 0:
            return results[0]['score']
        else:
            logger.warning(f"No results for pairwise similarity: '{entity1}' vs '{entity2}'")
            return 0.0

    except Exception as e:
        logger.error(f"Pairwise similarity failed for '{entity1}' and '{entity2}': {e}")
        return 0.0


# Convenience exports
__all__ = [
    'safe_colbert_search',
    'safe_colbert_pairwise_similarity',
]
