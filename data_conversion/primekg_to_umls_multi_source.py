#!/usr/bin/env python3
"""
PrimeKG to UMLS CUI-based Triples Converter (Multi-Source Version)

Converts PrimeKG knowledge graph to triples using UMLS CUI identifiers.
Supports multiple mapping sources (MONDO, HPO, DOID, MESH, NCIT, etc.)

Strategy: Load ALL mapping files and apply them based on source

Usage:
    python primekg_to_umls_multi_source.py
    python primekg_to_umls_multi_source.py --keep-unmapped
"""

import argparse
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiSourceUMLSConverter:
    """Convert PrimeKG to UMLS CUI-based triples using multiple mapping sources"""

    # Source name normalization: kg.csv source → mapping file prefix
    SOURCE_NORMALIZATION = {
        'MONDO': 'mondo',
        'HPO': 'hp',
        'DOID': 'doid',
        'MESH': 'mesh',
        'NCIT': 'ncit',
        'OMIM': 'omim',
        'Orphanet': 'orphanet',
        'SCTID': 'sctid',
        'ICD10': 'icd10',
        'ICD9': 'icd9',
        'GARD': 'gard',
        'EFO': 'efo',
        'REACTOME': 'reactome',
        'MedDRA': 'meddra',
        'NORD': 'nord',
        'Oncotree': 'oncotree',
        'DECIPHER': 'decipher',
        'HGNC': 'hgnc',
        'GTR': 'gtr',
        'MedGen': 'medgen',
    }

    def __init__(
        self,
        kg_path: str = 'primekg_data/kg.csv',
        mappings_dir: str = 'primekg_data/mappings',
        output_path: str = 'primekg_data/umls_triples.txt',
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

        # Multi-source mappings: source_name → {id → umls_cui}
        self.source_mappings: Dict[str, Dict[str, str]] = {}

        # Statistics
        self.stats = defaultdict(int)

    def load_all_mappings(self):
        """Load all mapping files from mappings directory"""
        logger.info("="*60)
        logger.info("LOADING ALL MAPPING FILES")
        logger.info("="*60)

        if not self.mappings_dir.exists():
            logger.warning(f"Mappings directory not found: {self.mappings_dir}")
            logger.warning("Run: python extract_all_mondo_xrefs.py first")
            return

        mapping_files = list(self.mappings_dir.glob('*_to_umls.csv'))

        if not mapping_files:
            logger.warning(f"No mapping files found in {self.mappings_dir}")
            return

        logger.info(f"Found {len(mapping_files)} mapping files")

        total_mappings = 0

        for mapping_file in sorted(mapping_files):
            source_name = mapping_file.stem.replace('_to_umls', '')

            try:
                df = pd.read_csv(mapping_file)

                if len(df) == 0:
                    logger.warning(f"  Skipping empty file: {mapping_file.name}")
                    continue

                # Get columns (assume first column is source ID, second is UMLS CUI)
                source_col = df.columns[0]
                umls_col = df.columns[1]

                # Clean and convert to dict
                df_clean = df[[source_col, umls_col]].dropna()
                df_clean[source_col] = df_clean[source_col].astype(str).str.strip()
                df_clean[umls_col] = df_clean[umls_col].astype(str).str.strip()

                # Apply MONDO normalization if this is mondo mapping
                if source_name == 'mondo':
                    # Normalize MONDO IDs to match kg.csv format
                    # kg.csv uses: "8019", "11043" (no prefix, no leading zeros)
                    # mondo_to_umls.csv uses: "MONDO:0008019" or may already be normalized
                    df_clean[source_col] = df_clean[source_col].str.replace('MONDO:', '', regex=False)
                    df_clean[source_col] = df_clean[source_col].str.lstrip('0')
                    df_clean[source_col] = df_clean[source_col].replace('', '0')

                # Create mapping dict
                mapping_dict = dict(zip(df_clean[source_col], df_clean[umls_col]))
                self.source_mappings[source_name] = mapping_dict

                logger.info(f"  ✅ {mapping_file.name:<35} {len(mapping_dict):>6,} mappings")
                total_mappings += len(mapping_dict)

            except Exception as e:
                logger.error(f"  ❌ Error loading {mapping_file.name}: {e}")
                continue

        logger.info(f"\nTotal: {len(self.source_mappings)} sources, {total_mappings:,} mappings")

        # Show source normalization map
        logger.info("\nSource normalization map:")
        for kg_source, map_source in self.SOURCE_NORMALIZATION.items():
            if map_source in self.source_mappings:
                count = len(self.source_mappings[map_source])
                logger.info(f"  {kg_source:<20} → {map_source:<15} ({count:,} mappings)")

    def map_to_cui(self, entity_id: str, source: str) -> Optional[str]:
        """
        Map entity ID to UMLS CUI

        Args:
            entity_id: Entity identifier
            source: Data source (e.g., MONDO, HPO, DOID)

        Returns:
            UMLS CUI if found, else None (or original ID if keep_unmapped=True)
        """
        # Direct UMLS CUI
        if source == "UMLS":
            return entity_id

        # Normalize source name
        normalized_source = self.SOURCE_NORMALIZATION.get(source)

        if not normalized_source:
            # Source not in normalization map
            if self.keep_unmapped:
                return entity_id
            else:
                return None

        # Check if we have mapping for this source
        if normalized_source not in self.source_mappings:
            # No mapping file for this source
            if self.keep_unmapped:
                return entity_id
            else:
                return None

        # Look up in mapping
        mapping_dict = self.source_mappings[normalized_source]
        entity_id_str = str(entity_id).strip()

        if entity_id_str in mapping_dict:
            self.stats[f'{source}_mapped'] += 1
            return mapping_dict[entity_id_str]

        # Not found in mapping
        if self.keep_unmapped:
            self.stats[f'{source}_unmapped'] += 1
            return entity_id
        else:
            self.stats[f'{source}_unmapped'] += 1
            return None

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

                    # Skip if any None (shouldn't happen with keep_unmapped=False)
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
        source_stats = defaultdict(lambda: {'mapped': 0, 'unmapped': 0})

        for key, count in self.stats.items():
            if key.endswith('_mapped'):
                source = key.replace('_mapped', '')
                source_stats[source]['mapped'] = count
            elif key.endswith('_unmapped'):
                source = key.replace('_unmapped', '')
                source_stats[source]['unmapped'] = count

        # Sort by total (mapped + unmapped)
        sorted_sources = sorted(
            source_stats.items(),
            key=lambda x: x[1]['mapped'] + x[1]['unmapped'],
            reverse=True
        )

        logger.info(f"\n{'Source':<20} {'Mapped':>12} {'Unmapped':>12} {'Total':>12} {'Rate':>8}")
        logger.info("-"*68)

        for source, stats in sorted_sources:
            mapped = stats['mapped']
            unmapped = stats['unmapped']
            total = mapped + unmapped
            rate = mapped / total * 100 if total > 0 else 0

            logger.info(
                f"{source:<20} {mapped:>12,} {unmapped:>12,} {total:>12,} {rate:>7.1f}%"
            )

        return total_triples


def main():
    parser = argparse.ArgumentParser(
        description='Convert PrimeKG to UMLS CUI-based triples (multi-source)'
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
        default='primekg_data/umls_triples_multi.txt',
        help='Output file (default: primekg_data/umls_triples_multi.txt)'
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
