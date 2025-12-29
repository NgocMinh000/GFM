"""
Stage 3.0: UMLS Data Loader
Loads and indexes UMLS concept data from RRF files
"""

import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm
import logging

from .config import UMLSMappingConfig

logger = logging.getLogger(__name__)


@dataclass
class UMLSConcept:
    """UMLS concept with all metadata"""
    cui: str
    preferred_name: str
    aliases: List[str]
    semantic_types: List[str]
    definitions: List[str]


class UMLSLoader:
    """
    Loads UMLS data from RRF files and builds indices

    Files:
    - MRCONSO.RRF: Concept names and synonyms
    - MRSTY.RRF: Semantic types
    - MRDEF.RRF: Definitions (optional)
    """

    def __init__(self, config: UMLSMappingConfig):
        self.config = config
        self.concepts: Dict[str, UMLSConcept] = {}
        self.name_to_cuis: Dict[str, Set[str]] = defaultdict(set)

        # Cache paths
        cache_dir = Path(config.umls_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_dir / "umls_concepts.pkl"

    def load(self) -> Dict[str, UMLSConcept]:
        """Load UMLS data (from cache or parse RRF files)"""

        # Try cache first
        if not self.config.force_recompute and self.cache_path.exists():
            logger.info(f"Loading UMLS data from cache: {self.cache_path}")
            with open(self.cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                self.concepts = cached_data['concepts']
                self.name_to_cuis = cached_data['name_to_cuis']
            logger.info(f"Loaded {len(self.concepts)} concepts from cache")
            return self.concepts

        # Parse RRF files
        logger.info("Parsing UMLS RRF files...")
        self._parse_mrconso()
        self._parse_mrsty()
        if self.config.mrdef_path and Path(self.config.mrdef_path).exists():
            self._parse_mrdef()

        # Save to cache
        logger.info(f"Saving UMLS data to cache: {self.cache_path}")
        with open(self.cache_path, 'wb') as f:
            pickle.dump({
                'concepts': self.concepts,
                'name_to_cuis': self.name_to_cuis
            }, f)

        return self.concepts

    def _parse_mrconso(self):
        """Parse MRCONSO.RRF for concept names and synonyms"""
        logger.info(f"Parsing MRCONSO.RRF: {self.config.mrconso_path}")

        mrconso_path = Path(self.config.mrconso_path)
        if not mrconso_path.exists():
            raise FileNotFoundError(f"MRCONSO.RRF not found: {mrconso_path}")

        # Count lines for progress bar
        with open(mrconso_path, 'r', encoding='utf-8', errors='ignore') as f:
            total_lines = sum(1 for _ in f)

        with open(mrconso_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, total=total_lines, desc="Parsing MRCONSO"):
                fields = line.strip().split('|')

                # Fields: CUI, LAT, TS, LUI, STT, SUI, ISPREF, AUI, SAUI, SCUI, SDUI, SAB, TTY, CODE, STR, SRL, SUPPRESS, CVF
                cui = fields[0]
                language = fields[1]
                is_preferred = fields[6] == 'Y'
                name = fields[14].lower()

                # Only English
                if language != self.config.umls_language:
                    continue

                # Initialize concept if not exists
                if cui not in self.concepts:
                    self.concepts[cui] = UMLSConcept(
                        cui=cui,
                        preferred_name="",
                        aliases=[],
                        semantic_types=[],
                        definitions=[]
                    )

                # Set preferred name
                if is_preferred and not self.concepts[cui].preferred_name:
                    self.concepts[cui].preferred_name = name

                # Add to aliases
                if name not in self.concepts[cui].aliases:
                    self.concepts[cui].aliases.append(name)

                # Build name -> CUI index
                self.name_to_cuis[name].add(cui)

        logger.info(f"Parsed {len(self.concepts)} concepts from MRCONSO")

    def _parse_mrsty(self):
        """Parse MRSTY.RRF for semantic types"""
        logger.info(f"Parsing MRSTY.RRF: {self.config.mrsty_path}")

        mrsty_path = Path(self.config.mrsty_path)
        if not mrsty_path.exists():
            raise FileNotFoundError(f"MRSTY.RRF not found: {mrsty_path}")

        with open(mrsty_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, desc="Parsing MRSTY"):
                fields = line.strip().split('|')

                # Fields: CUI, TUI, STN, STY, ATUI, CVF
                cui = fields[0]
                semantic_type = fields[3]

                if cui in self.concepts:
                    if semantic_type not in self.concepts[cui].semantic_types:
                        self.concepts[cui].semantic_types.append(semantic_type)

        logger.info(f"Added semantic types to {len(self.concepts)} concepts")

    def _parse_mrdef(self):
        """Parse MRDEF.RRF for definitions (optional)"""
        logger.info(f"Parsing MRDEF.RRF: {self.config.mrdef_path}")

        mrdef_path = Path(self.config.mrdef_path)
        if not mrdef_path.exists():
            logger.warning(f"MRDEF.RRF not found: {mrdef_path}")
            return

        with open(mrdef_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, desc="Parsing MRDEF"):
                fields = line.strip().split('|')

                # Fields: CUI, AUI, ATUI, SATUI, SAB, DEF, SUPPRESS, CVF
                cui = fields[0]
                definition = fields[5]

                if cui in self.concepts:
                    if definition not in self.concepts[cui].definitions:
                        self.concepts[cui].definitions.append(definition)

        logger.info(f"Added definitions to concepts")

    def get_all_names(self) -> List[str]:
        """Get all unique concept names for indexing"""
        all_names = set()
        for concept in self.concepts.values():
            all_names.update(concept.aliases)
        return list(all_names)

    def lookup_by_name(self, name: str) -> List[str]:
        """Lookup CUIs by concept name"""
        return list(self.name_to_cuis.get(name.lower(), set()))
