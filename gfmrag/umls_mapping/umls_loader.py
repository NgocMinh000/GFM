"""
Stage 3.0: UMLS Data Loader
Loads and indexes UMLS concept data from RRF files
"""

import pickle
import json
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
        self.umls_aliases: Dict[str, List[str]] = defaultdict(list)  # alias -> [CUI1, CUI2, ...]

        # Cache paths
        cache_dir = Path(config.umls_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path_concepts = cache_dir / "umls_concepts.pkl"
        self.cache_path_aliases = cache_dir / "umls_aliases.pkl"
        self.cache_path_stats = cache_dir / "umls_stats.json"

        # Statistics tracking
        self.stats = {
            'lines_processed': 0,
            'lines_english': 0,
            'total_concepts': 0,
            'total_aliases': 0,
            'concepts_with_semantic_types': 0,
            'concepts_with_definitions': 0,
            'avg_aliases_per_concept': 0.0,
            'avg_semantic_types_per_concept': 0.0,
        }

    def load(self) -> Dict[str, UMLSConcept]:
        """Load UMLS data (from cache or parse RRF files)"""

        # Try cache first
        if not self.config.force_recompute and self.cache_path_concepts.exists():
            logger.info(f"Loading UMLS data from cache: {self.cache_path_concepts}")
            with open(self.cache_path_concepts, 'rb') as f:
                self.concepts = pickle.load(f)
            with open(self.cache_path_aliases, 'rb') as f:
                self.umls_aliases = pickle.load(f)
            with open(self.cache_path_stats, 'r') as f:
                self.stats = json.load(f)
            logger.info(f"Loaded {len(self.concepts)} concepts from cache")
            return self.concepts

        # Parse RRF files
        logger.info("Parsing UMLS RRF files...")
        logger.info("This may take 30-60 minutes for full UMLS...")

        self._parse_mrconso()
        self._parse_mrsty()
        if self.config.mrdef_path and Path(self.config.mrdef_path).exists():
            self._parse_mrdef()

        # Post-processing
        self._post_process()

        # Compute statistics
        self._compute_statistics()

        # Save to cache
        self._save_cache()

        return self.concepts

    def _parse_mrconso(self):
        """Parse MRCONSO.RRF for concept names and synonyms"""
        logger.info(f"Parsing MRCONSO.RRF: {self.config.mrconso_path}")
        logger.info("Format: CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|...")

        mrconso_path = Path(self.config.mrconso_path)
        if not mrconso_path.exists():
            raise FileNotFoundError(f"MRCONSO.RRF not found: {mrconso_path}")

        # Count lines for progress bar
        logger.info("Counting lines...")
        with open(mrconso_path, 'r', encoding='utf-8', errors='ignore') as f:
            total_lines = sum(1 for _ in f)
        logger.info(f"Total lines: {total_lines:,}")

        lines_processed = 0
        lines_english = 0

        logger.info("Processing...")
        with open(mrconso_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, total=total_lines, desc="MRCONSO"):
                lines_processed += 1

                fields = line.strip().split('|')

                if len(fields) < 15:
                    continue

                # Fields: CUI, LAT, TS, LUI, STT, SUI, ISPREF, AUI, SAUI, SCUI, SDUI, SAB, TTY, CODE, STR, ...
                cui = fields[0]
                language = fields[1]
                term_type = fields[12]  # PT = Preferred Term (as per requirements)
                text = fields[14]

                # Only English
                if language != self.config.umls_language:
                    continue

                lines_english += 1

                # Skip empty
                if not text:
                    continue

                # Normalize text (apply same normalization as Task 1.3)
                normalized = normalize_text(text)
                expanded = expand_abbreviations(normalized)

                # Initialize concept if not exists
                if cui not in self.concepts:
                    self.concepts[cui] = UMLSConcept(
                        cui=cui,
                        preferred_name="",
                        aliases=[],
                        semantic_types=[],
                        definitions=[]
                    )

                # Set preferred name (use TTY='PT' as per requirements)
                if term_type == 'PT' and not self.concepts[cui].preferred_name:
                    self.concepts[cui].preferred_name = text  # Store original preferred name

                # Add to aliases (normalized)
                if expanded not in self.concepts[cui].aliases:
                    self.concepts[cui].aliases.append(expanded)

                # Build reverse index: alias -> CUIs
                if cui not in self.umls_aliases[expanded]:
                    self.umls_aliases[expanded].append(cui)

        logger.info(f"✓ MRCONSO processed:")
        logger.info(f"  Total lines:    {lines_processed:,}")
        logger.info(f"  English lines:  {lines_english:,}")
        logger.info(f"  Concepts:       {len(self.concepts):,}")
        logger.info(f"  Unique aliases: {len(self.umls_aliases):,}")

        self.stats['lines_processed'] = lines_processed
        self.stats['lines_english'] = lines_english

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

    def _post_process(self):
        """Post-processing: Remove concepts without preferred names"""
        logger.info("Post-processing...")

        concepts_removed = 0
        cuis_to_remove = []

        for cui, concept in self.concepts.items():
            if not concept.preferred_name:
                cuis_to_remove.append(cui)
                concepts_removed += 1

        for cui in cuis_to_remove:
            del self.concepts[cui]

        logger.info(f"✓ Removed {concepts_removed:,} concepts without preferred names")
        logger.info(f"✓ Remaining concepts: {len(self.concepts):,}")

    def _compute_statistics(self):
        """Compute statistics"""
        logger.info("Computing statistics...")

        self.stats['total_concepts'] = len(self.concepts)
        self.stats['total_aliases'] = len(self.umls_aliases)

        self.stats['concepts_with_semantic_types'] = sum(
            1 for c in self.concepts.values() if c.semantic_types
        )
        self.stats['concepts_with_definitions'] = sum(
            1 for c in self.concepts.values() if c.definitions
        )

        if len(self.concepts) > 0:
            self.stats['avg_aliases_per_concept'] = sum(
                len(c.aliases) for c in self.concepts.values()
            ) / len(self.concepts)

            self.stats['avg_semantic_types_per_concept'] = sum(
                len(c.semantic_types) for c in self.concepts.values()
            ) / len(self.concepts)

        logger.info(f"✓ Statistics computed")

    def _save_cache(self):
        """Save parsed data to cache files"""
        logger.info("Saving to cache...")

        # Save concepts
        logger.info(f"Saving umls_concepts.pkl...")
        with open(self.cache_path_concepts, 'wb') as f:
            pickle.dump(self.concepts, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save aliases
        logger.info(f"Saving umls_aliases.pkl...")
        with open(self.cache_path_aliases, 'wb') as f:
            pickle.dump(dict(self.umls_aliases), f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save stats
        logger.info(f"Saving umls_stats.json...")
        with open(self.cache_path_stats, 'w') as f:
            json.dump(self.stats, f, indent=2)

        # Log file sizes
        import os
        concepts_size = os.path.getsize(self.cache_path_concepts) / (1024**3)
        aliases_size = os.path.getsize(self.cache_path_aliases) / (1024**3)
        stats_size = os.path.getsize(self.cache_path_stats) / 1024

        logger.info(f"✓ Files saved:")
        logger.info(f"  umls_concepts.pkl:  {concepts_size:.2f} GB")
        logger.info(f"  umls_aliases.pkl:   {aliases_size:.2f} GB")
        logger.info(f"  umls_stats.json:    {stats_size:.1f} KB")

        # Log final statistics
        logger.info(f"\n{'='*70}")
        logger.info(f"FINAL STATISTICS")
        logger.info(f"{'='*70}")
        logger.info(f"Concepts:")
        logger.info(f"  Total:                    {self.stats['total_concepts']:,}")
        logger.info(f"  With semantic types:      {self.stats['concepts_with_semantic_types']:,} ({self.stats['concepts_with_semantic_types']/self.stats['total_concepts']*100:.1f}%)")
        logger.info(f"  With definitions:         {self.stats['concepts_with_definitions']:,} ({self.stats['concepts_with_definitions']/self.stats['total_concepts']*100:.1f}%)")
        logger.info(f"Aliases:")
        logger.info(f"  Total unique aliases:     {self.stats['total_aliases']:,}")
        logger.info(f"  Avg per concept:          {self.stats['avg_aliases_per_concept']:.1f}")
        logger.info(f"Semantic Types:")
        logger.info(f"  Avg per concept:          {self.stats['avg_semantic_types_per_concept']:.1f}")

        # Sample concepts
        logger.info(f"\nSample concepts (first 3):")
        for cui in list(self.concepts.keys())[:3]:
            data = self.concepts[cui]
            logger.info(f"  {cui}: {data.preferred_name}")
            logger.info(f"    Aliases: {len(data.aliases)}")
            logger.info(f"    Types: {data.semantic_types[:2] if data.semantic_types else 'None'}")

    def get_all_names(self) -> List[str]:
        """Get all unique concept names for indexing"""
        return list(self.umls_aliases.keys())

    def lookup_by_name(self, name: str) -> List[str]:
        """Lookup CUIs by concept name (normalized)"""
        # Normalize input first
        normalized = normalize_text(name)
        expanded = expand_abbreviations(normalized)
        return self.umls_aliases.get(expanded, [])
