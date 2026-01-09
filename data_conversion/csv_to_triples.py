#!/usr/bin/env python3
"""
CSV to Triples Converter
========================

Converts complex CSV files with entity relationship data into simple triple format (e-r-e).

Input CSV Format:
    relation,display_relation,x_index,x_id,x_type,x_name,x_source,y_index,y_id,y_type,y_name,y_source

Output Triple Format:
    head,relation,tail

Example:
    Input:  protein_protein,ppi,0,9796,gene/protein,PHYHIP,NCBI,8889,56992,gene/protein,KIF15,NCBI
    Output: PHYHIP,ppi,KIF15

Usage:
    python csv_to_triples.py input.csv output.txt
    python csv_to_triples.py input.csv output.txt --relation-column display_relation
    python csv_to_triples.py input.csv output.txt --add-metadata
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CSVToTriplesConverter:
    """Converter from complex CSV to simple triples format"""

    def __init__(
        self,
        input_path: str,
        output_path: str,
        head_column: str = "x_name",
        relation_column: str = "display_relation",
        tail_column: str = "y_name",
        add_metadata: bool = False,
        deduplicate: bool = True,
        normalize_entities: bool = True,
        fallback_relation_column: str = "relation"
    ):
        """
        Initialize converter

        Args:
            input_path: Path to input CSV file
            output_path: Path to output triples file
            head_column: Column name for head entity (default: x_name)
            relation_column: Column name for relation (default: display_relation)
            tail_column: Column name for tail entity (default: y_name)
            add_metadata: Include metadata triples (types, sources, IDs)
            deduplicate: Remove duplicate triples
            normalize_entities: Normalize entity names (strip, lowercase)
            fallback_relation_column: Fallback relation column if primary is empty
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.head_column = head_column
        self.relation_column = relation_column
        self.tail_column = tail_column
        self.add_metadata = add_metadata
        self.deduplicate = deduplicate
        self.normalize_entities = normalize_entities
        self.fallback_relation_column = fallback_relation_column

        # Statistics
        self.stats = {
            'total_rows': 0,
            'valid_triples': 0,
            'invalid_rows': 0,
            'duplicate_triples': 0,
            'unique_entities': set(),
            'unique_relations': set(),
            'relation_counts': Counter(),
            'entity_types': Counter()
        }

    def normalize_text(self, text: str) -> str:
        """Normalize entity or relation text"""
        if pd.isna(text):
            return ""

        text = str(text).strip()

        if self.normalize_entities:
            # Optional: lowercase (uncomment if needed)
            # text = text.lower()

            # Remove extra whitespace
            text = ' '.join(text.split())

        return text

    def validate_triple(self, head: str, relation: str, tail: str) -> bool:
        """Validate that triple has all required fields"""
        return bool(head and relation and tail)

    def create_metadata_triples(self, row: Dict) -> List[Tuple[str, str, str]]:
        """
        Create additional metadata triples from row data

        Example metadata:
            PHYHIP,has_type,gene/protein
            PHYHIP,has_id,9796
            PHYHIP,has_source,NCBI
        """
        metadata_triples = []

        # Head entity metadata
        head = self.normalize_text(row.get(self.head_column, ''))
        if head:
            if 'x_type' in row and pd.notna(row['x_type']):
                x_type = self.normalize_text(row['x_type'])
                metadata_triples.append((head, 'has_type', x_type))
                self.stats['entity_types'][x_type] += 1

            if 'x_id' in row and pd.notna(row['x_id']):
                x_id = self.normalize_text(row['x_id'])
                metadata_triples.append((head, 'has_id', x_id))

            if 'x_source' in row and pd.notna(row['x_source']):
                x_source = self.normalize_text(row['x_source'])
                metadata_triples.append((head, 'has_source', x_source))

        # Tail entity metadata
        tail = self.normalize_text(row.get(self.tail_column, ''))
        if tail:
            if 'y_type' in row and pd.notna(row['y_type']):
                y_type = self.normalize_text(row['y_type'])
                metadata_triples.append((tail, 'has_type', y_type))
                self.stats['entity_types'][y_type] += 1

            if 'y_id' in row and pd.notna(row['y_id']):
                y_id = self.normalize_text(row['y_id'])
                metadata_triples.append((tail, 'has_id', y_id))

            if 'y_source' in row and pd.notna(row['y_source']):
                y_source = self.normalize_text(row['y_source'])
                metadata_triples.append((tail, 'has_source', y_source))

        return metadata_triples

    def convert(self) -> Dict:
        """
        Convert CSV to triples

        Returns:
            Statistics dictionary
        """
        logger.info(f"Reading CSV file: {self.input_path}")

        # Check if file exists
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        # Read CSV
        try:
            df = pd.read_csv(self.input_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV: {e}")

        self.stats['total_rows'] = len(df)
        logger.info(f"Total rows in CSV: {self.stats['total_rows']}")

        # Validate required columns
        required_cols = [self.head_column, self.tail_column]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            logger.info(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check relation column
        if self.relation_column not in df.columns:
            if self.fallback_relation_column in df.columns:
                logger.warning(
                    f"Column '{self.relation_column}' not found, "
                    f"using '{self.fallback_relation_column}' instead"
                )
                self.relation_column = self.fallback_relation_column
            else:
                raise ValueError(f"Relation column not found: {self.relation_column}")

        # Collect triples
        triples = []
        seen_triples = set() if self.deduplicate else None

        logger.info("Converting rows to triples...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            # Extract main triple
            head = self.normalize_text(row[self.head_column])
            relation = self.normalize_text(row[self.relation_column])
            tail = self.normalize_text(row[self.tail_column])

            # Validate triple
            if not self.validate_triple(head, relation, tail):
                self.stats['invalid_rows'] += 1
                logger.debug(f"Row {idx}: Invalid triple - head={head}, rel={relation}, tail={tail}")
                continue

            # Check for duplicates
            triple = (head, relation, tail)
            if self.deduplicate:
                if triple in seen_triples:
                    self.stats['duplicate_triples'] += 1
                    continue
                seen_triples.add(triple)

            # Add main triple
            triples.append(triple)
            self.stats['valid_triples'] += 1
            self.stats['unique_entities'].add(head)
            self.stats['unique_entities'].add(tail)
            self.stats['unique_relations'].add(relation)
            self.stats['relation_counts'][relation] += 1

            # Add metadata triples if requested
            if self.add_metadata:
                metadata = self.create_metadata_triples(row)
                for meta_triple in metadata:
                    if self.deduplicate:
                        if meta_triple not in seen_triples:
                            triples.append(meta_triple)
                            seen_triples.add(meta_triple)
                            self.stats['valid_triples'] += 1
                    else:
                        triples.append(meta_triple)
                        self.stats['valid_triples'] += 1

        # Write output
        logger.info(f"Writing {len(triples)} triples to: {self.output_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, 'w', encoding='utf-8') as f:
            for head, relation, tail in triples:
                f.write(f"{head},{relation},{tail}\n")

        # Update statistics
        self.stats['unique_entities'] = len(self.stats['unique_entities'])
        self.stats['unique_relations'] = len(self.stats['unique_relations'])

        # Print summary
        self.print_summary()

        return self.stats

    def print_summary(self):
        """Print conversion summary"""
        logger.info("\n" + "="*60)
        logger.info("CONVERSION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total rows processed:    {self.stats['total_rows']}")
        logger.info(f"Valid triples:           {self.stats['valid_triples']}")
        logger.info(f"Invalid rows:            {self.stats['invalid_rows']}")
        logger.info(f"Duplicate triples:       {self.stats['duplicate_triples']}")
        logger.info(f"Unique entities:         {self.stats['unique_entities']}")
        logger.info(f"Unique relations:        {self.stats['unique_relations']}")
        logger.info("")
        logger.info("Top 10 relations:")
        for relation, count in self.stats['relation_counts'].most_common(10):
            logger.info(f"  {relation:30s} {count:6d} triples")

        if self.add_metadata and self.stats['entity_types']:
            logger.info("")
            logger.info("Entity types:")
            for entity_type, count in self.stats['entity_types'].most_common():
                logger.info(f"  {entity_type:30s} {count:6d} entities")

        logger.info("="*60)
        logger.info(f"Output saved to: {self.output_path}")
        logger.info("="*60 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Convert CSV with entity relationships to triple format (e-r-e)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python csv_to_triples.py input.csv output.txt

  # Use 'relation' column instead of 'display_relation'
  python csv_to_triples.py input.csv output.txt --relation-column relation

  # Add metadata triples (types, IDs, sources)
  python csv_to_triples.py input.csv output.txt --add-metadata

  # Custom column names
  python csv_to_triples.py input.csv output.txt \\
      --head-column source_entity \\
      --relation-column edge_type \\
      --tail-column target_entity

  # Keep duplicates and disable normalization
  python csv_to_triples.py input.csv output.txt \\
      --no-deduplicate \\
      --no-normalize
        """
    )

    parser.add_argument('input_csv', help='Input CSV file path')
    parser.add_argument('output_triples', help='Output triples file path')

    parser.add_argument(
        '--head-column',
        default='x_name',
        help='Column name for head entity (default: x_name)'
    )
    parser.add_argument(
        '--relation-column',
        default='display_relation',
        help='Column name for relation (default: display_relation)'
    )
    parser.add_argument(
        '--tail-column',
        default='y_name',
        help='Column name for tail entity (default: y_name)'
    )
    parser.add_argument(
        '--fallback-relation',
        default='relation',
        help='Fallback relation column if primary not found (default: relation)'
    )
    parser.add_argument(
        '--add-metadata',
        action='store_true',
        help='Include metadata triples (entity types, IDs, sources)'
    )
    parser.add_argument(
        '--no-deduplicate',
        action='store_true',
        help='Keep duplicate triples'
    )
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Disable entity name normalization'
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

    # Create converter
    try:
        converter = CSVToTriplesConverter(
            input_path=args.input_csv,
            output_path=args.output_triples,
            head_column=args.head_column,
            relation_column=args.relation_column,
            tail_column=args.tail_column,
            add_metadata=args.add_metadata,
            deduplicate=not args.no_deduplicate,
            normalize_entities=not args.no_normalize,
            fallback_relation_column=args.fallback_relation
        )

        # Run conversion
        stats = converter.convert()

        logger.info("✅ Conversion completed successfully!")
        sys.exit(0)

    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
