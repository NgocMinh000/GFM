#!/usr/bin/env python3
"""
Quick workaround: Run PrimeKG pipeline without umls_mondo.csv

Since umls_mondo.csv download failed, this script runs the pipeline
using "filter" strategy which doesn't require the mapping file.

Usage:
    python primekg_workaround.py

This will:
1. Skip mapping download
2. Use existing kg.csv
3. Run converter with --strategy filter
4. Generate triples (UMLS only, 200K-500K triples)
"""

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("="*60)
    logger.info("PrimeKG Workaround - Filter Strategy (No Mapping Needed)")
    logger.info("="*60)
    logger.info("")

    # Check if kg.csv exists
    data_dir = Path("./primekg_data")
    kg_path = data_dir / "kg.csv"

    if not kg_path.exists():
        logger.error(f"kg.csv not found at {kg_path}")
        logger.error("Please download kg.csv first:")
        logger.error("  wget -O primekg_data/kg.csv https://dataverse.harvard.edu/api/access/datafile/6180620")
        return 1

    logger.info(f"✓ Found kg.csv ({kg_path.stat().st_size / (1024**3):.1f} GB)")
    logger.info("")

    # Run converter with filter strategy
    output_dir = Path("./primekg_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "primekg_umls_triples.txt"

    logger.info("Running converter with 'filter' strategy (UMLS only)...")
    logger.info("This will take ~3-5 minutes...")
    logger.info("")

    cmd = [
        sys.executable,
        "primekg_to_umls_triples.py",
        str(kg_path),
        str(output_path),
        "--strategy", "filter"
    ]

    try:
        result = subprocess.run(cmd, check=True)

        logger.info("")
        logger.info("="*60)
        logger.info("✅ SUCCESS!")
        logger.info("="*60)
        logger.info(f"Output: {output_path}")

        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024**2)
            with open(output_path) as f:
                lines = sum(1 for _ in f)
            logger.info(f"Size: {size_mb:.1f} MB")
            logger.info(f"Triples: {lines:,}")

        logger.info("")
        logger.info("Next steps:")
        logger.info(f"  cp {output_path} /home/user/GFM/data/kg.txt")
        logger.info("  cd /home/user/GFM")
        logger.info("  python -m gfmrag.workflow.stage1_index_dataset")
        logger.info("")

        return 0

    except subprocess.CalledProcessError as e:
        logger.error(f"Conversion failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
