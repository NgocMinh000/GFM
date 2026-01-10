#!/usr/bin/env python3
"""
Create UMLS-MONDO mapping from MONDO .obo file

Downloads MONDO ontology and extracts UMLS cross-references.
Based on PrimeKG's processing scripts.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MONDOParser:
    """Parse MONDO .obo file and extract UMLS mappings"""

    # Try multiple sources for MONDO .obo
    MONDO_OBO_URLS = [
        "https://github.com/monarch-initiative/mondo/releases/latest/download/mondo.obo",
        "http://purl.obolibrary.org/obo/mondo.obo",
        "https://raw.githubusercontent.com/monarch-initiative/mondo/master/mondo.obo"
    ]

    def __init__(self, output_dir: str = "./primekg_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.obo_path = self.output_dir / "mondo.obo"
        self.mapping_path = self.output_dir / "umls_mondo.csv"

    def download_mondo_obo(self) -> bool:
        """Download MONDO .obo file"""
        if self.obo_path.exists():
            logger.info(f"MONDO .obo already exists: {self.obo_path}")
            size_mb = self.obo_path.stat().st_size / (1024 * 1024)
            logger.info(f"File size: {size_mb:.1f} MB")
            return True

        logger.info("Trying multiple sources for MONDO .obo...")
        logger.info("This may take a few minutes (~100 MB)...")

        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        }

        for i, url in enumerate(self.MONDO_OBO_URLS, 1):
            logger.info(f"\nAttempt {i}/{len(self.MONDO_OBO_URLS)}: {url}")

            try:
                response = requests.get(url, stream=True, headers=headers, timeout=30)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))

                with open(self.obo_path, 'wb') as f, tqdm(
                    desc="Downloading",
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

                logger.info(f"✓ Downloaded to: {self.obo_path}")
                size_mb = self.obo_path.stat().st_size / (1024 * 1024)
                logger.info(f"File size: {size_mb:.1f} MB")
                return True

            except Exception as e:
                logger.warning(f"Failed: {e}")
                if self.obo_path.exists():
                    self.obo_path.unlink()
                continue

        logger.error("All download attempts failed!")
        logger.error("\nManual download option:")
        logger.error("  wget https://github.com/monarch-initiative/mondo/releases/latest/download/mondo.obo")
        logger.error(f"  mv mondo.obo {self.obo_path}")
        return False

    def parse_mondo_obo(self) -> pd.DataFrame:
        """
        Parse MONDO .obo file and extract UMLS cross-references

        Returns DataFrame with columns: mondo_id, umls_id
        """
        logger.info(f"Parsing MONDO .obo file: {self.obo_path}")

        if not self.obo_path.exists():
            raise FileNotFoundError(f"MONDO .obo not found: {self.obo_path}")

        # Parse .obo file
        mondo_xrefs = []
        current_id = None
        in_term = False

        with open(self.obo_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Parsing"):
                line = line.strip()

                # New term block
                if line == "[Term]":
                    in_term = True
                    current_id = None
                    continue

                # End of term block
                if not line and in_term:
                    in_term = False
                    current_id = None
                    continue

                # Not in term block
                if not in_term:
                    continue

                # Extract MONDO ID
                if line.startswith("id: MONDO:"):
                    current_id = line.split(": ")[1].replace("MONDO:", "")
                    continue

                # Skip obsolete terms
                if line.startswith("is_obsolete: true"):
                    in_term = False
                    current_id = None
                    continue

                # Extract cross-references
                if current_id and line.startswith("xref: "):
                    xref = line.split(": ", 1)[1].split()[0]  # Get first part before comments

                    # Parse xref (format: ONTOLOGY:ID)
                    if ":" in xref:
                        ont, ontology_id = xref.split(":", 1)

                        # Save UMLS references
                        if ont == "UMLS":
                            mondo_xrefs.append({
                                'mondo_id': f"MONDO:{current_id}",
                                'umls_id': ontology_id,  # Just the CUI, e.g., C0011849
                                'ontology': ont
                            })

                # Extract from property_value skos:exactMatch (common format)
                # Format: property_value: skos:exactMatch UMLS:C0012634
                if current_id and 'property_value: skos:exactMatch UMLS:' in line:
                    # Extract UMLS CUI after "UMLS:"
                    umls_cui = line.split('UMLS:')[1].strip()
                    # Remove any trailing quotes or whitespace
                    umls_cui = umls_cui.split()[0].strip('"')
                    if umls_cui.startswith('C'):  # Valid UMLS CUI
                        mondo_xrefs.append({
                            'mondo_id': f"MONDO:{current_id}",
                            'umls_id': umls_cui,
                            'ontology': 'UMLS'
                        })

                # Extract from property_value closeMatch (alternative format)
                if current_id and 'property_value: http://www.geneontology.org/formats/oboInOwl#hasDbXref' in line:
                    # Extract URL from property value
                    match = re.search(r'http://[^\s"]+', line)
                    if match:
                        url = match.group(0)

                        # UMLS closeMatch
                        if 'identifiers.org/umls' in url.lower():
                            umls_id = url.split('/')[-1]
                            if umls_id.startswith('C'):  # Valid UMLS CUI
                                mondo_xrefs.append({
                                    'mondo_id': f"MONDO:{current_id}",
                                    'umls_id': umls_id,
                                    'ontology': 'UMLS'
                                })

        logger.info(f"Extracted {len(mondo_xrefs)} MONDO→UMLS references")

        # Convert to DataFrame and deduplicate
        df = pd.DataFrame(mondo_xrefs)

        if len(df) == 0:
            logger.warning("No UMLS references found in MONDO .obo file!")
            return pd.DataFrame(columns=['mondo_id', 'umls_id'])

        # Keep only UMLS references
        df = df[df['ontology'] == 'UMLS'].copy()
        df = df[['mondo_id', 'umls_id']].drop_duplicates()

        logger.info(f"Final mapping: {len(df)} unique MONDO→UMLS mappings")

        # Show statistics
        mondo_count = df['mondo_id'].nunique()
        umls_count = df['umls_id'].nunique()
        logger.info(f"  - Unique MONDO IDs: {mondo_count:,}")
        logger.info(f"  - Unique UMLS CUIs: {umls_count:,}")

        # Show sample
        logger.info("\nSample mappings:")
        for _, row in df.head(5).iterrows():
            logger.info(f"  {row['mondo_id']} → {row['umls_id']}")

        return df

    def save_mapping(self, df: pd.DataFrame) -> bool:
        """Save mapping to CSV"""
        logger.info(f"\nSaving mapping to: {self.mapping_path}")

        try:
            df.to_csv(self.mapping_path, index=False)
            logger.info(f"✓ Saved {len(df)} mappings")

            # Verify
            df_check = pd.read_csv(self.mapping_path)
            logger.info(f"✓ Verified: {len(df_check)} rows")

            return True

        except Exception as e:
            logger.error(f"Failed to save mapping: {e}")
            return False

    def run(self) -> bool:
        """Complete workflow"""
        logger.info("="*60)
        logger.info("Creating UMLS-MONDO Mapping from MONDO Ontology")
        logger.info("="*60)

        # Step 1: Download
        if not self.download_mondo_obo():
            return False

        # Step 2: Parse
        try:
            df = self.parse_mondo_obo()
            if len(df) == 0:
                logger.error("No mappings extracted!")
                return False
        except Exception as e:
            logger.error(f"Failed to parse MONDO .obo: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Step 3: Save
        if not self.save_mapping(df):
            return False

        logger.info("")
        logger.info("="*60)
        logger.info("✅ UMLS-MONDO Mapping Created Successfully!")
        logger.info("="*60)
        logger.info(f"\nOutput: {self.mapping_path}")
        logger.info(f"Mappings: {len(df):,}")
        logger.info("")
        logger.info("Next step:")
        logger.info("  python primekg_pipeline.py --skip-download --strategy map")
        logger.info("="*60)

        return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Create UMLS-MONDO mapping from MONDO ontology"
    )
    parser.add_argument(
        '--output-dir',
        default='./primekg_data',
        help='Output directory (default: ./primekg_data)'
    )

    args = parser.parse_args()

    try:
        parser = MONDOParser(output_dir=args.output_dir)
        success = parser.run()
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("\n\nCancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
