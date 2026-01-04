"""
Stage 3.3: Synonym Cluster Aggregation
Aggregates candidates across synonym clusters using voting
"""

import numpy as np
from typing import Dict, List, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import logging

from .config import UMLSMappingConfig
from .candidate_generator import Candidate

logger = logging.getLogger(__name__)


@dataclass
class AggregatedCandidate:
    """Candidate with cluster-level aggregation"""
    cui: str
    name: str
    score: float
    cluster_support: int  # Number of entities in cluster voting for this CUI
    is_outlier: bool
    method: str = 'cluster_aggregated'


class ClusterAggregator:
    """
    Aggregates candidates across synonym clusters

    Uses voting mechanism:
    - Each entity in cluster votes for its top candidates
    - Aggregate scores using weighted average
    - Detect outliers (entities disagreeing with cluster)
    """

    def __init__(self, config: UMLSMappingConfig):
        self.config = config

    def aggregate_cluster(
        self,
        entity_candidates: Dict[str, List[Candidate]],
        cluster_members: Set[str]
    ) -> List[AggregatedCandidate]:
        """
        Aggregate candidates from all entities in a synonym cluster

        Args:
            entity_candidates: Dict mapping entity -> list of candidates
            cluster_members: Set of entities in the same cluster

        Returns:
            Aggregated candidates for the cluster
        """

        # Collect all CUI votes
        cui_votes: Dict[str, List[float]] = defaultdict(list)
        cui_names: Dict[str, str] = {}
        entity_support: Dict[str, Set[str]] = defaultdict(set)

        for entity in cluster_members:
            if entity not in entity_candidates:
                continue

            candidates = entity_candidates[entity]
            for candidate in candidates:
                cui_votes[candidate.cui].append(candidate.score)
                cui_names[candidate.cui] = candidate.name
                entity_support[candidate.cui].add(entity)

        # Compute aggregated scores
        aggregated = []
        for cui, scores in cui_votes.items():
            # Average score
            avg_score = np.mean(scores)

            # Cluster consensus (how many entities agree)
            support_count = len(entity_support[cui])
            consensus = support_count / len(cluster_members)

            # Method diversity (how consistently this CUI appears)
            diversity_bonus = min(len(scores) / len(cluster_members), 1.0)

            # Final score (weighted combination)
            final_score = (
                avg_score * 0.6 +
                consensus * 0.3 +
                diversity_bonus * 0.1
            )

            aggregated.append(AggregatedCandidate(
                cui=cui,
                name=cui_names[cui],
                score=final_score,
                cluster_support=support_count,
                is_outlier=False
            ))

        # Sort by score
        aggregated.sort(key=lambda x: x.score, reverse=True)

        # Outlier detection
        if len(aggregated) > 0:
            aggregated = self._detect_outliers(aggregated, cluster_members)

        # Return top-k
        k = self.config.cluster_output_k
        return aggregated[:k]

    def _detect_outliers(
        self,
        candidates: List[AggregatedCandidate],
        cluster_members: Set[str]
    ) -> List[AggregatedCandidate]:
        """
        Detect outlier candidates (low cluster agreement)

        Mark as outlier if:
        - Cluster support < threshold
        - Score gap to top-1 is large
        """

        if len(candidates) == 0:
            return candidates

        threshold = self.config.cluster_output_k  # Using output_k as threshold placeholder
        top_score = candidates[0].score

        for candidate in candidates:
            # Low support
            support_ratio = candidate.cluster_support / len(cluster_members)
            if support_ratio < 0.5:
                candidate.is_outlier = True

            # Large score gap
            score_gap = top_score - candidate.score
            if score_gap > 0.5:
                candidate.is_outlier = True

        return candidates

    def aggregate_multiple_clusters(
        self,
        entity_candidates: Dict[str, List[Candidate]],
        entity_to_cluster: Dict[str, Set[str]]
    ) -> Dict[str, List[AggregatedCandidate]]:
        """
        Aggregate candidates for multiple clusters

        Args:
            entity_candidates: Dict mapping entity -> list of candidates
            entity_to_cluster: Dict mapping entity -> cluster members

        Returns:
            Dict mapping representative entity -> aggregated candidates
        """

        results = {}
        processed_clusters = set()

        for entity, cluster_members in entity_to_cluster.items():
            # Get cluster ID (use frozenset as identifier)
            cluster_id = frozenset(cluster_members)

            if cluster_id in processed_clusters:
                continue

            # Aggregate this cluster
            aggregated = self.aggregate_cluster(entity_candidates, cluster_members)
            results[entity] = aggregated

            processed_clusters.add(cluster_id)

        logger.info(f"Aggregated candidates for {len(results)} clusters")

        return results
