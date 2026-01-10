#!/usr/bin/env python3
"""
PrimeKG to Triples - Using MONDO Disease IDs

Since kg.csv does NOT have UMLS source but HAS MONDO IDs,
this script creates triples using MONDO identifiers directly.

Output format: head,relation,tail
"""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PrimeKGToTriples:
    """Convert PrimeKG to triples using MONDO/other identifiers"""

    def __init__(
        self,
        kg_path: str,
        output_path: str,
        include_sources: str = "MONDO,DrugBank,NCBI",
        use_display_relation: bool = True
    ):
        """
        Initialize converter

        Args:
            kg_path: Path to kg.csv
            output_path: Path to output triples
            include_sources: Comma-separated list of sources to include (default: MONDO,DrugBank,NCBI)
            use_display_relation: Use display_relation instead of relation
        """
        self.kg_path = Path(kg_path)
        self.output_path = Path(output_path)
        self.include_sources = set(s.strip() for s in include_sources.split(','))
        self.use_display_relation = use_display_relation

        self.stats = {
            'total_rows': 0,
            'triples_written': 0,
            'source_distribution': Counter(),
            'type_distribution': Counter(),
            'relation_distribution': Counter()
        }

    def load_kg(self) -> pd.DataFrame:
        """Load PrimeKG CSV"""
        logger.info(f"Loading PrimeKG from: {self.kg_path}")

        if not self.kg_path.exists():
            raise FileNotFoundError(f"kg.csv not found: {self.kg_path}")

        df = pd.read_csv(self.kg_path, low_memory=False)
        self.stats['total_rows'] = len(df)

        logger.info(f"Loaded {len(df):,} rows")

        # Show available sources
        x_sources = df['x_source'].value_counts()
        y_sources = df['y_source'].value_counts()
        all_sources = pd.concat([x_sources, y_sources]).groupby(level=0).sum().sort_values(ascending=False)

        logger.info(f"\nAvailable sources:")
        for source, count in all_sources.head(15).items():
            logger.info(f"  {source:20} {count:>10,}")

        return df

    def convert(self, df: pd.DataFrame):
        """Convert to triples"""
        logger.info(f"\nConverting to triples...")
        logger.info(f"Including sources: {', '.join(sorted(self.include_sources))}")

        relation_col = 'display_relation' if self.use_display_relation else 'relation'

        # Filter rows where either x_source or y_source is in include_sources
        if self.include_sources:
            filtered = df[
                df['x_source'].isin(self.include_sources) |
                df['y_source'].isin(self.include_sources)
            ].copy()
            logger.info(f"Filtered: {len(df):,} → {len(filtered):,} rows ({len(filtered)/len(df)*100:.1f}%)")
        else:
            filtered = df.copy()

        # Generate triples
        triples = []
        for _, row in tqdm(filtered.iterrows(), total=len(filtered), desc="Converting"):
            head = str(row['x_id'])
            relation = str(row[relation_col])
            tail = str(row['y_id'])

            # Add source prefixes for clarity
            x_source = row['x_source']
            y_source = row['y_source']

            # Format IDs with source prefix if not already present
            if x_source == 'MONDO' and not head.startswith('MONDO:'):
                head = f"MONDO:{head}"
            if y_source == 'MONDO' and not tail.startswith('MONDO:'):
                tail = f"MONDO:{tail}"

            triples.append((head, relation, tail))

            # Stats
            self.stats['source_distribution'][x_source] += 1
            self.stats['source_distribution'][y_source] += 1
            self.stats['type_distribution'][row['x_type']] += 1
            self.stats['type_distribution'][row['y_type']] += 1
            self.stats['relation_distribution'][relation] += 1

        # Deduplicate
        logger.info(f"\nDeduplicating triples...")
        triples = list(set(triples))
        logger.info(f"Unique triples: {len(triples):,}")

        # Write to file
        logger.info(f"\nWriting triples to: {self.output_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, 'w') as f:
            for head, relation, tail in tqdm(triples, desc="Writing"):
                f.write(f"{head},{relation},{tail}\n")

        self.stats['triples_written'] = len(triples)

        logger.info(f"✅ Wrote {len(triples):,} triples")

    def print_stats(self):
        """Print conversion statistics"""
        logger.info("")
        logger.info("="*60)
        logger.info("CONVERSION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total rows processed:  {self.stats['total_rows']:,}")
        logger.info(f"Triples written:       {self.stats['triples_written']:,}")
        logger.info("")

        logger.info("Top 10 data sources:")
        for source, count in self.stats['source_distribution'].most_common(10):
            logger.info(f"  {source:20} {count:>10,}")

        logger.info("")
        logger.info("Top 10 entity types:")
        for etype, count in self.stats['type_distribution'].most_common(10):
            logger.info(f"  {etype:20} {count:>10,}")

        logger.info("")
        logger.info("Top 10 relations:")
        for rel, count in self.stats['relation_distribution'].most_common(10):
            logger.info(f"  {rel:30} {count:>10,}")

        logger.info("="*60)

    def run(self):
        """Run conversion"""
        df = self.load_kg()
        self.convert(df)
        self.print_stats()

        logger.info("")
        logger.info("✅ Conversion completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PrimeKG to triples using MONDO/other IDs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # MONDO + DrugBank + NCBI (default)
  python primekg_to_triples_mondo.py kg.csv output.txt

  # Only MONDO diseases
  python primekg_to_triples_mondo.py kg.csv output.txt --sources MONDO

  # All sources
  python primekg_to_triples_mondo.py kg.csv output.txt --sources all

  # Custom sources
  python primekg_to_triples_mondo.py kg.csv output.txt --sources "MONDO,DrugBank,GO,HPO"
        """
    )

    parser.add_argument('kg_path', help='Path to kg.csv')
    parser.add_argument('output_path', help='Path to output triples file')
    parser.add_argument(
        '--sources',
        default='MONDO,DrugBank,NCBI',
        help='Comma-separated sources to include (or "all" for all sources)'
    )
    parser.add_argument(
        '--use-relation',
        action='store_true',
        help='Use relation instead of display_relation'
    )

    args = parser.parse_args()

    # Handle "all" sources
    if args.sources.lower() == 'all':
        # Will include all sources (no filtering)
        sources = ""
    else:
        sources = args.sources

    try:
        converter = PrimeKGToTriples(
            kg_path=args.kg_path,
            output_path=args.output_path,
            include_sources=sources if sources else "",
            use_display_relation=not args.use_relation
        )

        converter.run()
        sys.exit(0)

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
