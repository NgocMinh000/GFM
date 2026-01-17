"""
Knowledge Graph Filtering Module
Filter entities based on confidence, semantic types, and other criteria
"""

import re
import logging
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """Configuration for KG filtering"""
    enabled: bool = True
    min_confidence: float = 0.6
    remove_all_triples: bool = True
    require_medical_semantic_type: bool = False
    allowed_semantic_type_groups: List[str] = None
    entity_blacklist: List[str] = None
    filter_patterns: List[str] = None


class KGFilter:
    """
    Filter knowledge graph entities based on mapping quality and medical relevance
    """

    def __init__(self, config: FilterConfig, umls_loader=None):
        self.config = config
        self.umls_loader = umls_loader

        # Compile regex patterns
        self.compiled_patterns = []
        if config.filter_patterns:
            for pattern in config.filter_patterns:
                try:
                    self.compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{pattern}': {e}")

        # Normalize blacklist
        self.blacklist = set()
        if config.entity_blacklist:
            self.blacklist = {e.lower().strip() for e in config.entity_blacklist}

        logger.info(f"KG Filter initialized:")
        logger.info(f"  Min confidence: {config.min_confidence}")
        logger.info(f"  Remove all triples: {config.remove_all_triples}")
        logger.info(f"  Blacklist size: {len(self.blacklist)}")
        logger.info(f"  Filter patterns: {len(self.compiled_patterns)}")

    def should_include_entity(self, entity: str, mapping: 'EntityMapping') -> Tuple[bool, str]:
        """
        Determine if entity should be included in final KG

        Returns:
            (should_include, reason_for_exclusion)
        """
        if not self.config.enabled:
            return True, ""

        # 1. Check blacklist
        if entity.lower().strip() in self.blacklist:
            return False, f"blacklisted entity: '{entity}'"

        # 2. Check filter patterns
        for pattern in self.compiled_patterns:
            if pattern.match(entity):
                return False, f"matches filter pattern: '{pattern.pattern}'"

        # 3. Check confidence threshold
        if mapping.confidence < self.config.min_confidence:
            return False, f"low confidence: {mapping.confidence:.3f} < {self.config.min_confidence}"

        # 4. Check semantic type (if required)
        if self.config.require_medical_semantic_type and self.umls_loader:
            if not self._has_medical_semantic_type(mapping.cui):
                return False, f"non-medical semantic type"

        return True, ""

    def _has_medical_semantic_type(self, cui: str) -> bool:
        """Check if CUI has allowed medical semantic types"""
        if not self.umls_loader or not self.config.allowed_semantic_type_groups:
            return True

        # Get concept
        concept = self.umls_loader.concepts.get(cui)
        if not concept or not concept.semantic_types:
            return False

        # Check if any semantic type matches allowed groups
        # NOTE: This requires semantic_type_groups config to be loaded
        # For now, we just check if concept has any semantic types
        return len(concept.semantic_types) > 0

    def filter_mappings(
        self,
        mappings: Dict[str, 'EntityMapping']
    ) -> Tuple[Dict[str, 'EntityMapping'], Dict[str, str]]:
        """
        Filter entity mappings based on criteria

        Args:
            mappings: Dictionary of entity -> EntityMapping

        Returns:
            (included_mappings, excluded_entities_with_reasons)
        """
        included = {}
        excluded = {}

        for entity, mapping in mappings.items():
            should_include, reason = self.should_include_entity(entity, mapping)

            if should_include:
                included[entity] = mapping
            else:
                excluded[entity] = reason

        logger.info(f"KG Filtering Results:")
        logger.info(f"  Total entities: {len(mappings)}")
        logger.info(f"  Included: {len(included)} ({len(included)/len(mappings)*100:.1f}%)")
        logger.info(f"  Excluded: {len(excluded)} ({len(excluded)/len(mappings)*100:.1f}%)")

        # Log exclusion reasons breakdown
        if excluded:
            reason_counts = {}
            for reason in excluded.values():
                # Extract reason category (before colon)
                category = reason.split(':')[0] if ':' in reason else reason
                reason_counts[category] = reason_counts.get(category, 0) + 1

            logger.info("  Exclusion reasons:")
            for category, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
                logger.info(f"    - {category}: {count}")

        return included, excluded

    def filter_kg_triples(
        self,
        triples: List[Tuple[str, str, str]],
        excluded_entities: Set[str]
    ) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        """
        Filter KG triples to remove those involving excluded entities

        Args:
            triples: List of (head, relation, tail) triples
            excluded_entities: Set of entities to exclude

        Returns:
            (included_triples, excluded_triples)
        """
        if not self.config.remove_all_triples:
            return triples, []

        included = []
        excluded_triples = []

        for head, relation, tail in triples:
            # Check if either head or tail is excluded
            if head in excluded_entities or tail in excluded_entities:
                excluded_triples.append((head, relation, tail))
            else:
                included.append((head, relation, tail))

        logger.info(f"KG Triple Filtering:")
        logger.info(f"  Total triples: {len(triples)}")
        logger.info(f"  Included: {len(included)} ({len(included)/len(triples)*100:.1f}%)")
        logger.info(f"  Excluded: {len(excluded_triples)} ({len(excluded_triples)/len(triples)*100:.1f}%)")

        return included, excluded_triples
