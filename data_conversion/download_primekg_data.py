#!/usr/bin/env python3
"""
PrimeKG Auto Downloader

Automatically download PrimeKG data and mapping files from Harvard Dataverse and GitHub.

Downloads:
1. kg.csv - Main PrimeKG knowledge graph (4M triples, ~1.5GB)
2. umls_mondo.csv - UMLS-MONDO mapping file (~15K mappings, ~500KB)

Usage:
    python download_primekg_data.py
    python download_primekg_data.py --output-dir ./primekg_data
    python download_primekg_data.py --skip-kg  # Only download mapping
"""

import argparse
import hashlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PrimeKGDownloader:
    """Download PrimeKG data from Harvard Dataverse"""

    # Harvard Dataverse API endpoints
    KG_URL = "https://dataverse.harvard.edu/api/access/datafile/6180620"
    KG_FILENAME = "kg.csv"
    KG_SIZE_MB = 1500  # Approximate size

    # GitHub raw URLs for mapping files
    MAPPING_BASE_URL = "https://raw.githubusercontent.com/mims-harvard/PrimeKG/main/datasets/data/umls"
    UMLS_MONDO_URL = f"{MAPPING_BASE_URL}/umls_mondo.csv"
    MAPPING_FILENAME = "umls_mondo.csv"

    def __init__(self, output_dir: str = "./primekg_data", chunk_size: int = 8192):
        """
        Initialize downloader

        Args:
            output_dir: Directory to save downloaded files
            chunk_size: Download chunk size in bytes
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size

        self.kg_path = self.output_dir / self.KG_FILENAME
        self.mapping_path = self.output_dir / self.MAPPING_FILENAME

    def download_file(
        self,
        url: str,
        output_path: Path,
        description: str,
        expected_size_mb: Optional[int] = None
    ) -> bool:
        """
        Download file with progress bar and retry logic

        Args:
            url: Download URL
            output_path: Path to save file
            description: Description for progress bar
            expected_size_mb: Expected file size in MB

        Returns:
            True if successful, False otherwise
        """
        # Check if file already exists
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"File already exists: {output_path} ({file_size_mb:.1f} MB)")

            # Verify size if expected size provided
            if expected_size_mb and abs(file_size_mb - expected_size_mb) > expected_size_mb * 0.1:
                logger.warning(f"File size mismatch (expected ~{expected_size_mb}MB, got {file_size_mb:.1f}MB)")
                response = input("Re-download? (y/N): ")
                if response.lower() != 'y':
                    return True
            else:
                response = input("Re-download? (y/N): ")
                if response.lower() != 'y':
                    return True

        logger.info(f"Downloading {description} from: {url}")

        # Retry logic
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(1, max_retries + 1):
            try:
                # Send GET request with stream=True
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()

                # Get file size from headers
                total_size = int(response.headers.get('content-length', 0))

                # Create progress bar
                progress_bar = tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc=description,
                    ncols=80
                )

                # Download in chunks
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                            progress_bar.update(len(chunk))

                progress_bar.close()

                # Verify download
                if total_size > 0:
                    actual_size = output_path.stat().st_size
                    if actual_size != total_size:
                        logger.error(f"Download incomplete: {actual_size}/{total_size} bytes")
                        output_path.unlink()
                        raise ValueError("Incomplete download")

                logger.info(f"✓ Downloaded: {output_path} ({total_size / (1024*1024):.1f} MB)")
                return True

            except requests.exceptions.RequestException as e:
                logger.error(f"Download failed (attempt {attempt}/{max_retries}): {e}")

                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed to download after {max_retries} attempts")
                    return False

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                if output_path.exists():
                    output_path.unlink()
                return False

        return False

    def download_kg(self) -> bool:
        """Download main PrimeKG kg.csv file"""
        return self.download_file(
            url=self.KG_URL,
            output_path=self.kg_path,
            description="PrimeKG kg.csv",
            expected_size_mb=self.KG_SIZE_MB
        )

    def download_mapping(self) -> bool:
        """Download UMLS-MONDO mapping file"""
        return self.download_file(
            url=self.UMLS_MONDO_URL,
            output_path=self.mapping_path,
            description="umls_mondo.csv"
        )

    def verify_files(self) -> bool:
        """Verify downloaded files"""
        logger.info("\n" + "="*60)
        logger.info("VERIFICATION")
        logger.info("="*60)

        success = True

        # Check kg.csv
        if self.kg_path.exists():
            size_mb = self.kg_path.stat().st_size / (1024 * 1024)
            logger.info(f"✓ kg.csv: {size_mb:.1f} MB")

            # Quick validation: check first line
            try:
                with open(self.kg_path, 'r') as f:
                    header = f.readline().strip()
                    if 'x_id' in header and 'y_id' in header:
                        logger.info(f"  Header: {header[:80]}...")
                    else:
                        logger.warning(f"  Unexpected header: {header[:80]}...")
                        success = False
            except Exception as e:
                logger.error(f"  Error reading file: {e}")
                success = False
        else:
            logger.error("✗ kg.csv: Not found")
            success = False

        # Check umls_mondo.csv
        if self.mapping_path.exists():
            size_kb = self.mapping_path.stat().st_size / 1024
            logger.info(f"✓ umls_mondo.csv: {size_kb:.1f} KB")

            # Quick validation: check format
            try:
                with open(self.mapping_path, 'r') as f:
                    header = f.readline().strip()
                    if 'umls_id' in header and 'mondo_id' in header:
                        logger.info(f"  Header: {header}")

                        # Count lines
                        line_count = sum(1 for _ in f)
                        logger.info(f"  Mappings: {line_count:,}")
                    else:
                        logger.warning(f"  Unexpected header: {header}")
                        success = False
            except Exception as e:
                logger.error(f"  Error reading file: {e}")
                success = False
        else:
            logger.error("✗ umls_mondo.csv: Not found")
            success = False

        logger.info("="*60)
        return success

    def run(self, skip_kg: bool = False, skip_mapping: bool = False) -> bool:
        """
        Run full download process

        Args:
            skip_kg: Skip downloading kg.csv
            skip_mapping: Skip downloading mapping file

        Returns:
            True if successful, False otherwise
        """
        logger.info("="*60)
        logger.info("PrimeKG Data Downloader")
        logger.info("="*60)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("")

        success = True

        # Download kg.csv
        if not skip_kg:
            logger.info("Step 1: Downloading kg.csv (~1.5GB, may take 5-15 minutes)...")
            if not self.download_kg():
                logger.error("Failed to download kg.csv")
                success = False
        else:
            logger.info("Step 1: Skipping kg.csv download")

        logger.info("")

        # Download mapping
        if not skip_mapping:
            logger.info("Step 2: Downloading umls_mondo.csv (~500KB)...")
            if not self.download_mapping():
                logger.error("Failed to download umls_mondo.csv")
                success = False
        else:
            logger.info("Step 2: Skipping umls_mondo.csv download")

        logger.info("")

        # Verify
        logger.info("Step 3: Verifying downloads...")
        if not self.verify_files():
            logger.error("Verification failed")
            success = False

        logger.info("")

        if success:
            logger.info("="*60)
            logger.info("✅ Download complete!")
            logger.info("="*60)
            logger.info("")
            logger.info("Downloaded files:")
            logger.info(f"  - {self.kg_path}")
            logger.info(f"  - {self.mapping_path}")
            logger.info("")
            logger.info("Next steps:")
            logger.info("  1. Convert to UMLS CUI triples:")
            logger.info(f"     python primekg_to_umls_triples.py {self.kg_path} output.txt \\")
            logger.info(f"         --mapping {self.mapping_path} --strategy map")
            logger.info("")
            logger.info("  2. Or run full pipeline:")
            logger.info("     python primekg_pipeline.py")
            logger.info("="*60)
        else:
            logger.error("="*60)
            logger.error("❌ Download failed")
            logger.error("="*60)
            logger.error("")
            logger.error("Please check:")
            logger.error("  - Network connection")
            logger.error("  - Disk space (need ~2GB)")
            logger.error("  - Firewall/proxy settings")
            logger.error("")
            logger.error("Manual download:")
            logger.error(f"  wget -O {self.kg_path} {self.KG_URL}")
            logger.error(f"  wget -O {self.mapping_path} {self.UMLS_MONDO_URL}")
            logger.error("="*60)

        return success


def main():
    parser = argparse.ArgumentParser(
        description="Download PrimeKG data from Harvard Dataverse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all files
  python download_primekg_data.py

  # Custom output directory
  python download_primekg_data.py --output-dir ./my_data

  # Only download mapping (if you have kg.csv)
  python download_primekg_data.py --skip-kg

  # Only download kg.csv (if you have mapping)
  python download_primekg_data.py --skip-mapping
        """
    )

    parser.add_argument(
        '--output-dir',
        default='./primekg_data',
        help='Output directory (default: ./primekg_data)'
    )
    parser.add_argument(
        '--skip-kg',
        action='store_true',
        help='Skip downloading kg.csv'
    )
    parser.add_argument(
        '--skip-mapping',
        action='store_true',
        help='Skip downloading umls_mondo.csv'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=8192,
        help='Download chunk size in bytes (default: 8192)'
    )

    args = parser.parse_args()

    # Check dependencies
    try:
        import requests
        import tqdm
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install requests tqdm")
        sys.exit(1)

    # Run downloader
    try:
        downloader = PrimeKGDownloader(
            output_dir=args.output_dir,
            chunk_size=args.chunk_size
        )

        success = downloader.run(
            skip_kg=args.skip_kg,
            skip_mapping=args.skip_mapping
        )

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("\n\nDownload cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
