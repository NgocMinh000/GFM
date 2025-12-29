"""
Stage 3.6: Confidence Scoring & Graph Propagation
Computes multi-factor confidence and propagates through synonym clusters
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

from .config import UMLSMappingConfig
from .cross_encoder_reranker import RerankedCandidate

logger = logging.getLogger(__name__)


@dataclass
class FinalMapping:
    """Final UMLS mapping with confidence"""
    entity: str
    cui: str
    name: str
    confidence: float
    tier: str  # 'high', 'medium', 'low'
    alternatives: List[Tuple[str, str, float]]  # [(cui, name, score)]
    cluster_members: Set[str]
    is_propagated: bool
    confidence_factors: Dict[str, float]


class ConfidencePropagator:
    """
    Computes multi-factor confidence and propagates through clusters

    Confidence factors:
    1. Score margin (gap between top-1 and top-2)
    2. Absolute score
    3. Cluster consensus
    4. Method agreement
    """

    def __init__(self, config: UMLSMappingConfig):
        self.config = config

    def compute_confidence(
        self,
        entity: str,
        candidates: List[RerankedCandidate],
        cluster_members: Set[str],
        cluster_mappings: Dict[str, str] = None
    ) -> FinalMapping:
        """
        Compute confidence for entity mapping

        Args:
            entity: Original entity
            candidates: Reranked candidates
            cluster_members: Entities in same cluster
            cluster_mappings: Existing mappings in cluster

        Returns:
            FinalMapping with confidence score
        """

        if len(candidates) == 0:
            return FinalMapping(
                entity=entity,
                cui="",
                name="",
                confidence=0.0,
                tier='low',
                alternatives=[],
                cluster_members=cluster_members,
                is_propagated=False,
                confidence_factors={}
            )

        # Top candidate
        top_candidate = candidates[0]

        # Compute confidence factors
        factors = {}

        # 1. Score margin (gap to top-2)
        if len(candidates) > 1:
            score_margin = candidates[0].score - candidates[1].score
        else:
            score_margin = candidates[0].score
        factors['score_margin'] = min(score_margin, 1.0)

        # 2. Absolute score
        factors['absolute_score'] = top_candidate.score

        # 3. Cluster consensus
        if cluster_mappings:
            consensus = self._compute_cluster_consensus(
                top_candidate.cui,
                cluster_mappings
            )
        else:
            consensus = 1.0  # Single entity cluster
        factors['cluster_consensus'] = consensus

        # 4. Method agreement (placeholder - would track which methods agreed)
        factors['method_agreement'] = 0.8  # Default

        # Weighted confidence
        confidence = (
            factors['score_margin'] * 0.35 +
            factors['absolute_score'] * 0.25 +
            factors['cluster_consensus'] * 0.25 +
            factors['method_agreement'] * 0.15
        )

        # Determine tier
        if confidence >= self.config.confidence_high_threshold:
            tier = 'high'
        elif confidence >= 0.50:
            tier = 'medium'
        else:
            tier = 'low'

        # Alternatives
        alternatives = [
            (c.cui, c.name, c.score)
            for c in candidates[1:4]  # Top 3 alternatives
        ]

        return FinalMapping(
            entity=entity,
            cui=top_candidate.cui,
            name=top_candidate.name,
            confidence=confidence,
            tier=tier,
            alternatives=alternatives,
            cluster_members=cluster_members,
            is_propagated=False,
            confidence_factors=factors
        )

    def propagate_through_cluster(
        self,
        cluster_members: Set[str],
        entity_mappings: Dict[str, FinalMapping]
    ) -> Dict[str, FinalMapping]:
        """
        Propagate high-confidence mappings through synonym cluster

        Args:
            cluster_members: Set of entities in cluster
            entity_mappings: Existing mappings

        Returns:
            Updated mappings with propagation
        """

        # Collect CUI votes from high-confidence mappings
        cui_votes = defaultdict(list)
        for entity in cluster_members:
            if entity in entity_mappings:
                mapping = entity_mappings[entity]
                if mapping.tier == 'high':
                    cui_votes[mapping.cui].append((entity, mapping.confidence))

        # Check if cluster agrees
        if len(cui_votes) == 0:
            return entity_mappings

        # Find consensus CUI
        best_cui = max(cui_votes.items(), key=lambda x: len(x[1]))[0]
        support_count = len(cui_votes[best_cui])
        agreement = support_count / len(cluster_members)

        # Propagate if strong agreement
        if agreement >= self.config.propagation_min_agreement:
            # Get best mapping for this CUI
            best_mapping = max(
                [entity_mappings[e] for e, _ in cui_votes[best_cui]],
                key=lambda m: m.confidence
            )

            # Propagate to low-confidence entities
            for entity in cluster_members:
                if entity not in entity_mappings or entity_mappings[entity].tier != 'high':
                    # Create propagated mapping
                    propagated_confidence = best_mapping.confidence * self.config.propagation_min_agreement

                    entity_mappings[entity] = FinalMapping(
                        entity=entity,
                        cui=best_cui,
                        name=best_mapping.name,
                        confidence=propagated_confidence,
                        tier='medium' if propagated_confidence >= 0.5 else 'low',
                        alternatives=best_mapping.alternatives,
                        cluster_members=cluster_members,
                        is_propagated=True,
                        confidence_factors={
                            'propagated_from': best_mapping.entity,
                            'cluster_agreement': agreement
                        }
                    )

        return entity_mappings

    def _compute_cluster_consensus(
        self,
        cui: str,
        cluster_mappings: Dict[str, str]
    ) -> float:
        """
        Compute consensus within cluster

        Args:
            cui: Candidate CUI
            cluster_mappings: Dict mapping entity -> CUI in cluster

        Returns:
            Consensus score (0-1)
        """

        if not cluster_mappings:
            return 1.0

        # Count how many entities in cluster agree on this CUI
        agreements = sum(1 for mapped_cui in cluster_mappings.values() if mapped_cui == cui)
        consensus = agreements / len(cluster_mappings)

        return consensus

    def finalize_all_mappings(
        self,
        entity_mappings: Dict[str, FinalMapping],
        clusters: Dict[int, Set[str]]
    ) -> Dict[str, FinalMapping]:
        """
        Finalize all mappings with cluster-wide propagation

        Args:
            entity_mappings: Initial mappings
            clusters: Dict mapping cluster_id -> set of entities

        Returns:
            Finalized mappings with propagation
        """

        logger.info("Propagating high-confidence mappings through clusters...")

        for cluster_id, cluster_members in clusters.items():
            # Skip small clusters
            if len(cluster_members) > 20:  # Max cluster size
                continue

            # Propagate
            entity_mappings = self.propagate_through_cluster(
                cluster_members,
                entity_mappings
            )

        # Statistics
        total = len(entity_mappings)
        high = sum(1 for m in entity_mappings.values() if m.tier == 'high')
        medium = sum(1 for m in entity_mappings.values() if m.tier == 'medium')
        low = sum(1 for m in entity_mappings.values() if m.tier == 'low')
        propagated = sum(1 for m in entity_mappings.values() if m.is_propagated)

        logger.info(f"Finalized {total} mappings:")
        logger.info(f"  High confidence: {high} ({high/total*100:.1f}%)")
        logger.info(f"  Medium confidence: {medium} ({medium/total*100:.1f}%)")
        logger.info(f"  Low confidence: {low} ({low/total*100:.1f}%)")
        logger.info(f"  Propagated: {propagated} ({propagated/total*100:.1f}%)")

        return entity_mappings
