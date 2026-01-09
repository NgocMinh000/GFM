#!/usr/bin/env python3
"""
PrimeKG to UMLS CUI-based Triples Converter

Converts PrimeKG knowledge graph to triples using UMLS CUI identifiers.

Strategy: Map MONDO disease IDs to UMLS CUIs using umls_mondo.csv mapping file.

Usage:
    python primekg_to_umls_triples.py kg.csv umls_mondo.csv output.txt
    python primekg_to_umls_triples.py kg.csv umls_mondo.csv output.txt --strategy filter
    python primekg_to_umls_triples.py kg.csv umls_mondo.csv output.txt --keep-unmapped
"""

import argparse
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PrimeKGToUMLSConverter:
    """Convert PrimeKG to UMLS CUI-based triples"""

    def __init__(
        self,
        kg_path: str,
        umls_mondo_path: Optional[str] = None,
        strategy: str = "map",
        keep_unmapped: bool = False,
        use_display_relation: bool = True
    ):
        """
        Initialize converter

        Args:
            kg_path: Path to PrimeKG kg.csv
            umls_mondo_path: Path to umls_mondo.csv mapping file
            strategy: "map" (map MONDO→UMLS) or "filter" (filter UMLS only)
            keep_unmapped: Keep entities without UMLS CUI (use original ID)
            use_display_relation: Use display_relation instead of relation
        """
        self.kg_path = Path(kg_path)
        self.umls_mondo_path = Path(umls_mondo_path) if umls_mondo_path else None
        self.strategy = strategy
        self.keep_unmapped = keep_unmapped
        self.use_display_relation = use_display_relation

        # Mapping dicts
        self.mondo_to_umls: Dict[str, str] = {}
        self.umls_to_mondo: Dict[str, str] = {}

        # Statistics
        self.stats = {
            'total_rows': 0,
            'strategy': strategy,
            'mapped_triples': 0,
            'unmapped_triples': 0,
            'skipped_rows': 0,
            'x_mapped': 0,
            'y_mapped': 0,
            'x_umls_direct': 0,
            'y_umls_direct': 0,
            'x_mondo_mapped': 0,
            'y_mondo_mapped': 0,
            'x_unmapped': 0,
            'y_unmapped': 0,
            'source_distribution': Counter(),
            'type_distribution': Counter(),
            'relation_distribution': Counter()
        }

    def load_umls_mondo_mapping(self):
        """Load UMLS-MONDO mapping file"""
        if not self.umls_mondo_path:
            logger.warning("No umls_mondo.csv provided, mapping will be limited")
            return

        if not self.umls_mondo_path.exists():
            raise FileNotFoundError(f"Mapping file not found: {self.umls_mondo_path}")

        logger.info(f"Loading UMLS-MONDO mapping from: {self.umls_mondo_path}")

        try:
            df = pd.read_csv(self.umls_mondo_path)

            # Check columns
            if 'umls_id' not in df.columns or 'mondo_id' not in df.columns:
                raise ValueError("Mapping file must have 'umls_id' and 'mondo_id' columns")

            # Create bidirectional mappings
            self.mondo_to_umls = dict(zip(df['mondo_id'], df['umls_id']))
            self.umls_to_mondo = dict(zip(df['umls_id'], df['mondo_id']))

            logger.info(f"Loaded {len(self.mondo_to_umls)} MONDO→UMLS mappings")
            logger.info(f"Loaded {len(self.umls_to_mondo)} UMLS→MONDO mappings")

        except Exception as e:
            raise ValueError(f"Failed to load mapping file: {e}")

    def map_to_cui(self, entity_id: str, source: str) -> Optional[str]:
        """
        Map entity ID to UMLS CUI

        Args:
            entity_id: Entity identifier (e.g., MONDO:0005148, DB00001)
            source: Data source (e.g., MONDO, UMLS, DrugBank)

        Returns:
            UMLS CUI if found, else None (or original ID if keep_unmapped=True)
        """
        # Direct UMLS CUI
        if source == "UMLS":
            return entity_id

        # MONDO → UMLS mapping
        if source == "MONDO" and entity_id in self.mondo_to_umls:
            return self.mondo_to_umls[entity_id]

        # Not mapped
        if self.keep_unmapped:
            return entity_id
        else:
            return None

    def convert_strategy_filter(self, kg_df: pd.DataFrame) -> pd.DataFrame:
        """
        Strategy 1: Filter only UMLS entities

        Only keep rows where x_source=UMLS or y_source=UMLS
        """
        logger.info("Using FILTER strategy (UMLS only)")

        # Filter
        filtered = kg_df[
            (kg_df['x_source'] == 'UMLS') | (kg_df['y_source'] == 'UMLS')
        ].copy()

        logger.info(f"Filtered: {len(kg_df)} → {len(filtered)} rows ({len(filtered)/len(kg_df)*100:.1f}%)")

        # Use x_id and y_id directly as CUIs
        filtered['x_cui'] = filtered['x_id']
        filtered['y_cui'] = filtered['y_id']

        return filtered

    def convert_strategy_map(self, kg_df: pd.DataFrame) -> pd.DataFrame:
        """
        Strategy 2: Map MONDO → UMLS

        Map all MONDO IDs to UMLS CUIs using mapping file
        """
        logger.info("Using MAP strategy (MONDO→UMLS)")

        if not self.mondo_to_umls:
            logger.warning("No MONDO mapping loaded, falling back to filter strategy")
            return self.convert_strategy_filter(kg_df)

        logger.info("Mapping entities to UMLS CUIs...")

        # Map x_id and y_id to CUIs
        tqdm.pandas(desc="Mapping x_id")
        kg_df['x_cui'] = kg_df.progress_apply(
            lambda row: self.map_to_cui(row['x_id'], row['x_source']),
            axis=1
        )

        tqdm.pandas(desc="Mapping y_id")
        kg_df['y_cui'] = kg_df.progress_apply(
            lambda row: self.map_to_cui(row['y_id'], row['y_source']),
            axis=1
        )

        # Count mappings
        self.stats['x_mapped'] = kg_df['x_cui'].notna().sum()
        self.stats['y_mapped'] = kg_df['y_cui'].notna().sum()
        self.stats['x_unmapped'] = kg_df['x_cui'].isna().sum()
        self.stats['y_unmapped'] = kg_df['y_cui'].isna().sum()

        # Detailed mapping stats
        self.stats['x_umls_direct'] = (kg_df['x_source'] == 'UMLS').sum()
        self.stats['y_umls_direct'] = (kg_df['y_source'] == 'UMLS').sum()
        self.stats['x_mondo_mapped'] = (
            (kg_df['x_source'] == 'MONDO') & kg_df['x_cui'].notna()
        ).sum()
        self.stats['y_mondo_mapped'] = (
            (kg_df['y_source'] == 'MONDO') & kg_df['y_cui'].notna()
        ).sum()

        # Filter: only keep rows with both CUIs mapped
        if not self.keep_unmapped:
            mapped = kg_df.dropna(subset=['x_cui', 'y_cui']).copy()
            logger.info(
                f"Mapped: {len(kg_df)} → {len(mapped)} rows "
                f"({len(mapped)/len(kg_df)*100:.1f}%)"
            )
            return mapped
        else:
            # Keep all rows (unmapped IDs retained)
            kg_df['x_cui'] = kg_df['x_cui'].fillna(kg_df['x_id'])
            kg_df['y_cui'] = kg_df['y_cui'].fillna(kg_df['y_id'])
            logger.info(f"Kept all {len(kg_df)} rows (unmapped IDs retained)")
            return kg_df

    def convert(self, output_path: str) -> Dict:
        """
        Convert PrimeKG to UMLS CUI-based triples

        Args:
            output_path: Path to output triples file

        Returns:
            Statistics dictionary
        """
        # Load data
        logger.info(f"Loading PrimeKG from: {self.kg_path}")

        if not self.kg_path.exists():
            raise FileNotFoundError(f"KG file not found: {self.kg_path}")

        try:
            kg_df = pd.read_csv(self.kg_path, low_memory=False)
        except Exception as e:
            raise ValueError(f"Failed to read KG file: {e}")

        self.stats['total_rows'] = len(kg_df)
        logger.info(f"Loaded {len(kg_df):,} rows")

        # Validate required columns
        required_cols = ['x_id', 'y_id', 'x_source', 'y_source', 'relation']
        if self.use_display_relation:
            required_cols.append('display_relation')

        missing_cols = [col for col in required_cols if col not in kg_df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            logger.info(f"Available columns: {list(kg_df.columns)}")
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Load mapping if needed
        if self.strategy == "map":
            self.load_umls_mondo_mapping()

        # Statistics: source distribution
        self.stats['source_distribution'].update(kg_df['x_source'].tolist())
        self.stats['source_distribution'].update(kg_df['y_source'].tolist())

        # Statistics: type distribution
        if 'x_type' in kg_df.columns:
            self.stats['type_distribution'].update(kg_df['x_type'].tolist())
        if 'y_type' in kg_df.columns:
            self.stats['type_distribution'].update(kg_df['y_type'].tolist())

        # Apply conversion strategy
        if self.strategy == "filter":
            converted_df = self.convert_strategy_filter(kg_df)
        elif self.strategy == "map":
            converted_df = self.convert_strategy_map(kg_df)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Choose relation column
        relation_col = 'display_relation' if self.use_display_relation else 'relation'

        # Statistics: relation distribution
        self.stats['relation_distribution'].update(converted_df[relation_col].tolist())

        # Write output triples
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Writing {len(converted_df):,} triples to: {output_path}")

        triples_written = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for _, row in tqdm(converted_df.iterrows(), total=len(converted_df), desc="Writing"):
                x_cui = row['x_cui']
                relation = row[relation_col]
                y_cui = row['y_cui']

                # Validate
                if not x_cui or not relation or not y_cui:
                    self.stats['skipped_rows'] += 1
                    continue

                f.write(f"{x_cui},{relation},{y_cui}\n")
                triples_written += 1

        self.stats['mapped_triples'] = triples_written
        self.stats['unmapped_triples'] = len(converted_df) - triples_written

        # Print summary
        self.print_summary()

        return self.stats

    def print_summary(self):
        """Print conversion summary"""
        logger.info("\n" + "="*60)
        logger.info("CONVERSION SUMMARY")
        logger.info("="*60)
        logger.info(f"Strategy:              {self.stats['strategy']}")
        logger.info(f"Total rows processed:  {self.stats['total_rows']:,}")
        logger.info(f"Triples written:       {self.stats['mapped_triples']:,}")
        logger.info(f"Skipped rows:          {self.stats['skipped_rows']:,}")

        if self.strategy == "map":
            logger.info("")
            logger.info("Mapping statistics:")
            logger.info(f"  X entities mapped:   {self.stats['x_mapped']:,}")
            logger.info(f"    - Direct UMLS:     {self.stats['x_umls_direct']:,}")
            logger.info(f"    - MONDO mapped:    {self.stats['x_mondo_mapped']:,}")
            logger.info(f"  X entities unmapped: {self.stats['x_unmapped']:,}")
            logger.info(f"  Y entities mapped:   {self.stats['y_mapped']:,}")
            logger.info(f"    - Direct UMLS:     {self.stats['y_umls_direct']:,}")
            logger.info(f"    - MONDO mapped:    {self.stats['y_mondo_mapped']:,}")
            logger.info(f"  Y entities unmapped: {self.stats['y_unmapped']:,}")

        logger.info("")
        logger.info("Top 10 data sources:")
        for source, count in self.stats['source_distribution'].most_common(10):
            logger.info(f"  {source:20s} {count:,}")

        if self.stats['type_distribution']:
            logger.info("")
            logger.info("Top 10 entity types:")
            for etype, count in self.stats['type_distribution'].most_common(10):
                logger.info(f"  {etype:20s} {count:,}")

        logger.info("")
        logger.info("Top 10 relations:")
        for relation, count in self.stats['relation_distribution'].most_common(10):
            logger.info(f"  {relation:30s} {count:,}")

        logger.info("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PrimeKG to UMLS CUI-based triples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Strategy 1: Filter UMLS only (simple)
  python primekg_to_umls_triples.py kg.csv output.txt --strategy filter

  # Strategy 2: Map MONDO→UMLS (recommended)
  python primekg_to_umls_triples.py kg.csv output.txt --mapping umls_mondo.csv

  # Keep unmapped entities (use original IDs)
  python primekg_to_umls_triples.py kg.csv output.txt --mapping umls_mondo.csv --keep-unmapped

  # Use 'relation' instead of 'display_relation'
  python primekg_to_umls_triples.py kg.csv output.txt --no-display-relation
        """
    )

    parser.add_argument('kg_csv', help='Path to PrimeKG kg.csv file')
    parser.add_argument('output', help='Path to output triples file')
    parser.add_argument(
        '--mapping',
        help='Path to umls_mondo.csv mapping file (for map strategy)'
    )
    parser.add_argument(
        '--strategy',
        choices=['filter', 'map'],
        default='map',
        help='Conversion strategy (default: map)'
    )
    parser.add_argument(
        '--keep-unmapped',
        action='store_true',
        help='Keep entities without UMLS CUI (use original ID)'
    )
    parser.add_argument(
        '--no-display-relation',
        action='store_true',
        help='Use "relation" column instead of "display_relation"'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate inputs
    if args.strategy == 'map' and not args.mapping:
        logger.warning(
            "Map strategy requires --mapping file. "
            "Falling back to filter strategy or provide mapping file."
        )

    # Create converter
    try:
        converter = PrimeKGToUMLSConverter(
            kg_path=args.kg_csv,
            umls_mondo_path=args.mapping,
            strategy=args.strategy,
            keep_unmapped=args.keep_unmapped,
            use_display_relation=not args.no_display_relation
        )

        # Run conversion
        stats = converter.convert(args.output)

        logger.info("✅ Conversion completed successfully!")
        sys.exit(0)

    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
