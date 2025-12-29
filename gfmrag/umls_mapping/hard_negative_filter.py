"""
Stage 3.4: Hard Negative Filtering & Semantic Type Checking
Filters candidates using hard negative detection and semantic type constraints
"""

import numpy as np
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging
from difflib import SequenceMatcher

from .config import UMLSMappingConfig
from .cluster_aggregator import AggregatedCandidate
from .umls_loader import UMLSLoader

logger = logging.getLogger(__name__)


@dataclass
class FilteredCandidate:
    """Candidate with hard negative and type filtering"""
    cui: str
    name: str
    score: float
    semantic_types: List[str]
    type_match: bool
    hard_negative_penalty: float
    method: str = 'hard_neg_filtered'


class HardNegativeFilter:
    """
    Filters candidates using:
    1. Hard negative detection (similar names, different CUIs)
    2. Semantic type checking (infer from KG relations)
    """

    def __init__(self, config: UMLSMappingConfig, umls_loader: UMLSLoader):
        self.config = config
        self.umls_loader = umls_loader

        # Semantic type groups from config
        self.type_groups = {
            'disease': set([
                'Disease or Syndrome',
                'Neoplastic Process',
                'Mental or Behavioral Dysfunction',
                'Pathologic Function',
                'Sign or Symptom'
            ]),
            'drug': set([
                'Pharmacologic Substance',
                'Antibiotic',
                'Immunologic Factor',
                'Vitamin',
                'Hormone'
            ]),
            'procedure': set([
                'Therapeutic or Preventive Procedure',
                'Diagnostic Procedure',
                'Laboratory Procedure',
                'Health Care Activity'
            ]),
            'anatomy': set([
                'Body Part, Organ, or Organ Component',
                'Tissue',
                'Cell',
                'Body System'
            ]),
            'biological': set([
                'Amino Acid, Peptide, or Protein',
                'Enzyme',
                'Nucleic Acid, Nucleoside, or Nucleotide',
                'Biologically Active Substance'
            ])
        }

    def filter_candidates(
        self,
        entity: str,
        candidates: List[AggregatedCandidate],
        kg_context: Dict[str, List[Tuple[str, str]]] = None
    ) -> List[FilteredCandidate]:
        """
        Filter candidates using hard negatives and semantic types

        Args:
            entity: Original entity text
            candidates: List of aggregated candidates
            kg_context: Optional KG context (entity -> [(relation, target)])

        Returns:
            Filtered candidates
        """

        # Infer semantic type from KG context
        expected_type = self._infer_semantic_type(entity, kg_context)

        # Detect hard negatives
        hard_negative_scores = self._detect_hard_negatives(candidates)

        # Score candidates
        filtered = []
        for candidate in candidates:
            # Get UMLS concept
            umls_concept = self.umls_loader.concepts.get(candidate.cui)
            if not umls_concept:
                continue

            semantic_types = umls_concept.semantic_types

            # Check type match
            type_match = self._check_type_match(semantic_types, expected_type)

            # Hard negative penalty
            hard_neg_penalty = hard_negative_scores.get(candidate.cui, 0.0)

            # Compute final score
            final_score = (
                candidate.score * 0.7 +
                (1.0 if type_match else 0.0) * 0.2 -
                hard_neg_penalty * 0.1
            )

            filtered.append(FilteredCandidate(
                cui=candidate.cui,
                name=candidate.name,
                score=final_score,
                semantic_types=semantic_types,
                type_match=type_match,
                hard_negative_penalty=hard_neg_penalty
            ))

        # Sort by final score
        filtered.sort(key=lambda x: x.score, reverse=True)

        # Return top-k
        k = self.config.hard_neg_output_k
        return filtered[:k]

    def _infer_semantic_type(
        self,
        entity: str,
        kg_context: Dict[str, List[Tuple[str, str]]]
    ) -> str:
        """
        Infer semantic type from KG relations

        Examples:
        - treats -> drug
        - symptom_of -> disease
        - located_in -> anatomy
        """

        if not kg_context or entity not in kg_context:
            return None

        relations = kg_context[entity]

        # Count relation types
        relation_counts = defaultdict(int)
        for relation, target in relations:
            relation_counts[relation] += 1

        # Inference rules
        if 'treats' in relation_counts or 'treated_by' in relation_counts:
            return 'drug'
        elif 'symptom_of' in relation_counts or 'causes' in relation_counts:
            return 'disease'
        elif 'located_in' in relation_counts or 'part_of' in relation_counts:
            return 'anatomy'
        elif 'associated_with' in relation_counts:
            return 'biological'

        return None

    def _check_type_match(
        self,
        semantic_types: List[str],
        expected_type: str
    ) -> bool:
        """Check if semantic types match expected type"""

        if not expected_type:
            return True  # No constraint

        expected_types = self.type_groups.get(expected_type, set())

        for sty in semantic_types:
            if sty in expected_types:
                return True

        return False

    def _detect_hard_negatives(
        self,
        candidates: List[AggregatedCandidate]
    ) -> Dict[str, float]:
        """
        Detect hard negatives (similar names but different CUIs)

        Returns:
            Dict mapping CUI -> penalty score
        """

        penalties = defaultdict(float)

        # Compare all pairs
        for i, cand_i in enumerate(candidates):
            for j, cand_j in enumerate(candidates):
                if i >= j:
                    continue

                # Check string similarity
                similarity = self._string_similarity(cand_i.name, cand_j.name)

                # Hard negative: high string similarity but different CUIs
                if similarity > self.config.hard_neg_similarity_threshold and cand_i.cui != cand_j.cui:
                    # Penalize both
                    penalty = (similarity - self.config.hard_neg_similarity_threshold) * 0.5
                    penalties[cand_i.cui] += penalty
                    penalties[cand_j.cui] += penalty

        return penalties

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Compute string similarity using SequenceMatcher"""
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
