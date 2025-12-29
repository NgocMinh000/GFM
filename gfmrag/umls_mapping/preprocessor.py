"""
Stage 3.1: Preprocessing & Entity Preparation
Extracts entities from kg_clean.txt and builds synonym clusters
"""

import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm
import logging

from .config import UMLSMappingConfig
from .utils import normalize_text, expand_abbreviations

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Entity with normalization and metadata"""
    original: str
    normalized: str
    cluster_id: int
    synonym_group: Set[str]


class Preprocessor:
    """
    Extracts entities from kg_clean.txt and builds synonym clusters

    Input: kg_clean.txt with triples (head | relation | tail)
    Output: Entities with synonym clusters
    """

    def __init__(self, config: UMLSMappingConfig):
        self.config = config
        self.entities: Dict[str, Entity] = {}
        self.synonym_clusters: Dict[int, Set[str]] = defaultdict(set)
        self.entity_to_cluster: Dict[str, int] = {}

    def process(self, kg_clean_path: str) -> Dict[str, Entity]:
        """
        Process kg_clean.txt and extract entities with synonym information

        Returns:
            Dict mapping entity -> Entity object
        """
        logger.info(f"Processing KG file: {kg_clean_path}")

        # Step 1: Extract all entities and synonyms_of relationships
        all_entities, synonym_pairs = self._parse_kg_file(kg_clean_path)

        # Step 2: Build synonym clusters using union-find
        self._build_synonym_clusters(all_entities, synonym_pairs)

        # Step 3: Normalize entities
        self._normalize_entities()

        logger.info(f"Extracted {len(self.entities)} entities in {len(self.synonym_clusters)} clusters")

        return self.entities

    def _parse_kg_file(self, kg_path: str) -> Tuple[Set[str], List[Tuple[str, str]]]:
        """Parse kg_clean.txt to extract entities and synonym relationships"""

        all_entities = set()
        synonym_pairs = []

        kg_path = Path(kg_path)
        if not kg_path.exists():
            raise FileNotFoundError(f"KG file not found: {kg_path}")

        with open(kg_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Parsing KG"):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split('|')
                if len(parts) != 3:
                    continue

                head = parts[0].strip()
                relation = parts[1].strip()
                tail = parts[2].strip()

                # Collect all entities
                all_entities.add(head)
                all_entities.add(tail)

                # Collect synonyms_of relationships
                if relation == 'synonyms_of':
                    synonym_pairs.append((head, tail))

        logger.info(f"Found {len(all_entities)} unique entities")
        logger.info(f"Found {len(synonym_pairs)} synonym pairs")

        return all_entities, synonym_pairs

    def _build_synonym_clusters(self, all_entities: Set[str], synonym_pairs: List[Tuple[str, str]]):
        """Build synonym clusters using union-find algorithm"""

        # Union-Find data structure
        parent = {entity: entity for entity in all_entities}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]

        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                parent[root_y] = root_x

        # Union all synonym pairs
        for head, tail in synonym_pairs:
            union(head, tail)

        # Build clusters
        cluster_members = defaultdict(set)
        for entity in all_entities:
            root = find(entity)
            cluster_members[root].add(entity)

        # Assign cluster IDs
        cluster_id = 0
        for root, members in cluster_members.items():
            self.synonym_clusters[cluster_id] = members
            for member in members:
                self.entity_to_cluster[member] = cluster_id
            cluster_id += 1

        logger.info(f"Built {len(self.synonym_clusters)} synonym clusters")

        # Log cluster size distribution
        cluster_sizes = [len(members) for members in self.synonym_clusters.values()]
        logger.info(f"Cluster size - Min: {min(cluster_sizes)}, Max: {max(cluster_sizes)}, Avg: {sum(cluster_sizes)/len(cluster_sizes):.2f}")

    def _normalize_entities(self):
        """Normalize all entities"""

        for entity in tqdm(self.entity_to_cluster.keys(), desc="Normalizing entities"):
            cluster_id = self.entity_to_cluster[entity]
            synonym_group = self.synonym_clusters[cluster_id]

            # Normalize text
            normalized = normalize_text(entity)
            normalized = expand_abbreviations(normalized)

            # Create Entity object
            self.entities[entity] = Entity(
                original=entity,
                normalized=normalized,
                cluster_id=cluster_id,
                synonym_group=synonym_group
            )

    def get_entities_for_mapping(self) -> List[str]:
        """
        Get list of entities that need UMLS mapping

        Returns one representative per cluster (the normalized form)
        """
        representatives = []
        for cluster_id, members in self.synonym_clusters.items():
            # Pick the first entity in cluster as representative
            representative = next(iter(members))
            representatives.append(representative)

        logger.info(f"Selected {len(representatives)} representative entities for mapping")
        return representatives

    def get_cluster_members(self, entity: str) -> Set[str]:
        """Get all members in the same synonym cluster"""
        if entity not in self.entity_to_cluster:
            return {entity}
        cluster_id = self.entity_to_cluster[entity]
        return self.synonym_clusters[cluster_id]
