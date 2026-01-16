#!/usr/bin/env python3
"""
PrimeKG to UMLS CUI-based Triples Converter (Multi-Source v2)

Based on exhaustive compatibility check results:
- 22 compatible mapping files
- 6/9 kg.csv sources have mappings (96.4% entities)
- Handles normalization and conflicts

Usage:
    python primekg_to_umls_multi_source_v2.py
    python primekg_to_umls_multi_source_v2.py --keep-unmapped
"""

import argparse
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiSourceUMLSConverter:
    """Convert PrimeKG to UMLS CUI-based triples using 22 compatible mapping sources"""

    # kg.csv source → list of compatible mapping files (based on exhaustive check)
    SOURCE_TO_MAPPINGS = {
        'MONDO': ['mondo', 'gard', 'hp', 'icd11.foundation'],
        'GO': ['doid', 'orphanet', 'medgen'],
        'DrugBank': ['icd9', 'decipher'],  # Requires normalization
        'NCBI': ['efo', 'omim', 'omimps', 'mfomd', 'ogms', 'ido', 'mpath', 'mth', 'scdo'],
        'UBERON': ['nord', 'hgnc', 'nando'],
        'HPO': ['birnlex'],
    }

    # Sources that require normalization (remove prefix, strip leading zeros)
    NORMALIZE_SOURCES = {
        'MONDO',      # Remove 'MONDO:', strip zeros
        'DrugBank',   # Remove 'DB', strip zeros
        'GO',         # May have 'GO:' prefix
        'HPO',        # May have 'HP:' prefix
    }

    def __init__(
        self,
        kg_path: str = 'primekg_data/kg.csv',
        mappings_dir: str = 'primekg_data/mappings',
        output_path: str = 'primekg_data/umls_triples_multi_v2.txt',
        keep_unmapped: bool = False,
        use_display_relation: bool = True
    ):
        """
        Initialize converter

        Args:
            kg_path: Path to PrimeKG kg.csv
            mappings_dir: Directory containing *_to_umls.csv mapping files
            output_path: Output file for UMLS triples
            keep_unmapped: Keep entities without UMLS CUI (use original ID)
            use_display_relation: Use display_relation instead of relation
        """
        self.kg_path = Path(kg_path)
        self.mappings_dir = Path(mappings_dir)
        self.output_path = Path(output_path)
        self.keep_unmapped = keep_unmapped
        self.use_display_relation = use_display_relation

        # Multi-source mappings: source_name → {normalized_id → umls_cui}
        self.source_mappings: Dict[str, Dict[str, str]] = {}

        # Statistics
        self.stats = defaultdict(int)
        self.conflicts = defaultdict(int)  # Track conflicts (same ID → different CUIs)

    def normalize_id(self, entity_id: str, source: str) -> str:
        """
        Normalize entity ID for matching

        Args:
            entity_id: Original entity ID
            source: Data source

        Returns:
            Normalized ID
        """
        entity_id_str = str(entity_id).strip()

        if source not in self.NORMALIZE_SOURCES:
            return entity_id_str

        # Remove common prefixes
        for prefix in ['MONDO:', 'HP:', 'GO:', 'DB', 'DOID:', 'OMIM:', 'Orphanet:']:
            entity_id_str = entity_id_str.replace(prefix, '')

        # Strip leading zeros (but keep at least one digit)
        normalized = entity_id_str.lstrip('0')
        if not normalized:
            normalized = '0'

        return normalized

    def load_all_mappings(self):
        """Load only the 22 compatible mapping files"""
        logger.info("="*60)
        logger.info("LOADING COMPATIBLE MAPPING FILES")
        logger.info("="*60)

        if not self.mappings_dir.exists():
            logger.error(f"Mappings directory not found: {self.mappings_dir}")
            logger.error("Run: python extract_all_mondo_xrefs.py first")
            return

        # Collect all compatible mapping file names
        compatible_files = set()
        for source, mapping_names in self.SOURCE_TO_MAPPINGS.items():
            compatible_files.update(mapping_names)

        logger.info(f"Loading {len(compatible_files)} compatible mapping files:")

        total_mappings = 0
        loaded_count = 0

        for mapping_name in sorted(compatible_files):
            mapping_file = self.mappings_dir / f"{mapping_name}_to_umls.csv"

            if not mapping_file.exists():
                logger.warning(f"  ⚠️  Not found: {mapping_file.name}")
                continue

            try:
                df = pd.read_csv(mapping_file)

                if len(df) == 0:
                    logger.warning(f"  ⚠️  Empty: {mapping_file.name}")
                    continue

                # Get columns
                source_col = df.columns[0]
                umls_col = df.columns[1]

                # Clean data
                df_clean = df[[source_col, umls_col]].dropna()
                df_clean[source_col] = df_clean[source_col].astype(str).str.strip()
                df_clean[umls_col] = df_clean[umls_col].astype(str).str.strip()

                # Apply normalization to mapping IDs
                # Determine which sources use this mapping
                using_sources = [s for s, maps in self.SOURCE_TO_MAPPINGS.items() if mapping_name in maps]

                # Normalize IDs if any using source requires it
                needs_normalization = any(s in self.NORMALIZE_SOURCES for s in using_sources)

                if needs_normalization:
                    df_clean[source_col] = df_clean[source_col].apply(
                        lambda x: self.normalize_id(x, using_sources[0] if using_sources else '')
                    )

                # Create mapping dict
                mapping_dict = dict(zip(df_clean[source_col], df_clean[umls_col]))

                # Store mapping
                self.source_mappings[mapping_name] = mapping_dict

                logger.info(f"  ✅ {mapping_file.name:<35} {len(mapping_dict):>6,} mappings")
                total_mappings += len(mapping_dict)
                loaded_count += 1

            except Exception as e:
                logger.error(f"  ❌ Error loading {mapping_file.name}: {e}")
                continue

        logger.info(f"\nLoaded: {loaded_count}/{len(compatible_files)} files, {total_mappings:,} total mappings")

        # Show mapping availability by kg.csv source
        logger.info("\nMapping availability by kg.csv source:")
        for kg_source, mapping_names in self.SOURCE_TO_MAPPINGS.items():
            available = [m for m in mapping_names if m in self.source_mappings]
            total_maps = sum(len(self.source_mappings[m]) for m in available)
            logger.info(f"  {kg_source:<20} {len(available)}/{len(mapping_names)} files, {total_maps:,} mappings")

    def map_to_cui(self, entity_id: str, source: str) -> Optional[str]:
        """
        Map entity ID to UMLS CUI using all available mappings for this source

        Args:
            entity_id: Entity identifier
            source: Data source (e.g., MONDO, GO, DrugBank)

        Returns:
            UMLS CUI if found, else None (or original ID if keep_unmapped=True)
        """
        # Direct UMLS CUI
        if source == "UMLS":
            return entity_id

        # Check if this source has mappings
        if source not in self.SOURCE_TO_MAPPINGS:
            if self.keep_unmapped:
                return entity_id
            else:
                return None

        # Normalize entity ID
        normalized_id = self.normalize_id(entity_id, source)

        # Try all mappings for this source (in order of priority)
        mapping_names = self.SOURCE_TO_MAPPINGS[source]
        found_cuis = []

        for mapping_name in mapping_names:
            if mapping_name not in self.source_mappings:
                continue

            mapping_dict = self.source_mappings[mapping_name]

            if normalized_id in mapping_dict:
                cui = mapping_dict[normalized_id]
                found_cuis.append((mapping_name, cui))

        # Handle results
        if not found_cuis:
            # No mapping found
            self.stats[f'{source}_unmapped'] += 1
            if self.keep_unmapped:
                return entity_id
            else:
                return None

        elif len(found_cuis) == 1:
            # Single mapping found
            self.stats[f'{source}_mapped'] += 1
            self.stats[f'{source}_from_{found_cuis[0][0]}'] += 1
            return found_cuis[0][1]

        else:
            # Multiple mappings found - check for conflicts
            cuis = [cui for _, cui in found_cuis]
            if len(set(cuis)) == 1:
                # All mappings agree on same CUI
                self.stats[f'{source}_mapped'] += 1
                self.stats[f'{source}_multi_agree'] += 1
                return cuis[0]
            else:
                # Conflict: different CUIs
                self.stats[f'{source}_mapped'] += 1
                self.stats[f'{source}_conflict'] += 1
                self.conflicts[f'{source}:{normalized_id}'] = found_cuis
                # Use first mapping (priority order)
                return found_cuis[0][1]

    def convert(self, chunk_size: int = 500000) -> int:
        """
        Convert kg.csv to UMLS triples

        Args:
            chunk_size: Number of rows to process at once

        Returns:
            Number of triples generated
        """
        logger.info("="*60)
        logger.info("CONVERTING kg.csv TO UMLS TRIPLES")
        logger.info("="*60)

        if not self.kg_path.exists():
            raise FileNotFoundError(f"kg.csv not found: {self.kg_path}")

        # Determine relation column
        relation_col = 'display_relation' if self.use_display_relation else 'relation'

        # Open output file
        with open(self.output_path, 'w') as f:
            f.write("head,relation,tail\n")

            total_rows = 0
            total_triples = 0

            # Process in chunks
            for chunk in tqdm(
                pd.read_csv(self.kg_path, chunksize=chunk_size, low_memory=False),
                desc="Processing chunks"
            ):
                total_rows += len(chunk)

                # Check if relation column exists
                if relation_col not in chunk.columns:
                    logger.warning(f"Column '{relation_col}' not found, using 'relation'")
                    relation_col = 'relation'

                # Map entities to CUIs
                chunk['x_cui'] = chunk.apply(
                    lambda row: self.map_to_cui(row['x_id'], row['x_source']),
                    axis=1
                )

                chunk['y_cui'] = chunk.apply(
                    lambda row: self.map_to_cui(row['y_id'], row['y_source']),
                    axis=1
                )

                # Filter: only keep rows with both CUIs mapped
                if not self.keep_unmapped:
                    mapped_chunk = chunk.dropna(subset=['x_cui', 'y_cui'])
                else:
                    mapped_chunk = chunk

                # Write triples
                for _, row in mapped_chunk.iterrows():
                    head = row['x_cui']
                    relation = str(row[relation_col])
                    tail = row['y_cui']

                    # Skip if any None
                    if pd.isna(head) or pd.isna(tail) or pd.isna(relation):
                        continue

                    f.write(f"{head},{relation},{tail}\n")
                    total_triples += 1

        logger.info(f"\n{'='*60}")
        logger.info(f"CONVERSION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Input rows: {total_rows:,}")
        logger.info(f"Output triples: {total_triples:,} ({total_triples/total_rows*100:.2f}%)")

        # Show mapping statistics by source
        logger.info("\nMapping statistics by source:")
        source_stats = defaultdict(lambda: {'mapped': 0, 'unmapped': 0, 'conflicts': 0})

        for key, count in self.stats.items():
            if key.endswith('_mapped'):
                source = key.replace('_mapped', '')
                source_stats[source]['mapped'] = count
            elif key.endswith('_unmapped'):
                source = key.replace('_unmapped', '')
                source_stats[source]['unmapped'] = count
            elif key.endswith('_conflict'):
                source = key.replace('_conflict', '')
                source_stats[source]['conflicts'] = count

        # Sort by total
        sorted_sources = sorted(
            source_stats.items(),
            key=lambda x: x[1]['mapped'] + x[1]['unmapped'],
            reverse=True
        )

        logger.info(f"\n{'Source':<20} {'Mapped':>12} {'Unmapped':>12} {'Conflicts':>12} {'Rate':>8}")
        logger.info("-"*76)

        for source, stats in sorted_sources:
            mapped = stats['mapped']
            unmapped = stats['unmapped']
            conflicts = stats['conflicts']
            total = mapped + unmapped
            rate = mapped / total * 100 if total > 0 else 0

            logger.info(
                f"{source:<20} {mapped:>12,} {unmapped:>12,} {conflicts:>12,} {rate:>7.1f}%"
            )

        # Show breakdown by mapping file
        logger.info("\nMappings used (by file):")
        mapping_usage = [(k, v) for k, v in self.stats.items() if '_from_' in k]
        for key, count in sorted(mapping_usage, key=lambda x: -x[1])[:20]:
            logger.info(f"  {key:<50} {count:>8,}")

        # Show conflicts if any
        if self.conflicts:
            logger.info(f"\nConflicts detected: {len(self.conflicts)}")
            logger.info("Sample conflicts (first 10):")
            for i, (entity, mappings) in enumerate(list(self.conflicts.items())[:10]):
                logger.info(f"  {entity}:")
                for mapping_name, cui in mappings:
                    logger.info(f"    {mapping_name:<20} → {cui}")

        return total_triples


def main():
    parser = argparse.ArgumentParser(
        description='Convert PrimeKG to UMLS CUI-based triples (22 compatible mappings)'
    )
    parser.add_argument(
        '--kg-path',
        default='primekg_data/kg.csv',
        help='Path to kg.csv (default: primekg_data/kg.csv)'
    )
    parser.add_argument(
        '--mappings-dir',
        default='primekg_data/mappings',
        help='Directory with mapping files (default: primekg_data/mappings)'
    )
    parser.add_argument(
        '--output',
        default='primekg_data/umls_triples_multi_v2.txt',
        help='Output file (default: primekg_data/umls_triples_multi_v2.txt)'
    )
    parser.add_argument(
        '--keep-unmapped',
        action='store_true',
        help='Keep entities without UMLS mapping (use original ID)'
    )
    parser.add_argument(
        '--use-relation',
        action='store_true',
        help='Use "relation" column instead of "display_relation"'
    )

    args = parser.parse_args()

    # Initialize converter
    converter = MultiSourceUMLSConverter(
        kg_path=args.kg_path,
        mappings_dir=args.mappings_dir,
        output_path=args.output,
        keep_unmapped=args.keep_unmapped,
        use_display_relation=not args.use_relation
    )

    # Load mappings
    converter.load_all_mappings()

    if not converter.source_mappings:
        logger.error("No mapping files loaded. Exiting.")
        sys.exit(1)

    # Convert
    num_triples = converter.convert()

    logger.info(f"\n✅ Conversion complete!")
    logger.info(f"   Output: {converter.output_path}")
    logger.info(f"   Triples: {num_triples:,}")


if __name__ == "__main__":
    main()
