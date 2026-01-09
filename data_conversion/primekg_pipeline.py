#!/usr/bin/env python3
"""
PrimeKG to UMLS CUI Triples - Complete Pipeline

End-to-end pipeline:
1. Download PrimeKG kg.csv from Harvard Dataverse
2. Download umls_mondo.csv mapping from GitHub
3. Convert MONDO IDs to UMLS CUIs (reverse mapping)
4. Generate UMLS CUI-based triples
5. Validate output

Usage:
    # Full auto mode
    python primekg_pipeline.py

    # Custom options
    python primekg_pipeline.py --output-dir ./my_output --strategy map

    # Skip download if files exist
    python primekg_pipeline.py --skip-download

    # Run with specific strategy
    python primekg_pipeline.py --strategy filter  # UMLS only
    python primekg_pipeline.py --strategy map     # Map MONDO→UMLS (default)
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PrimeKGPipeline:
    """Complete PrimeKG to UMLS CUI triples pipeline"""

    def __init__(
        self,
        output_dir: str = "./primekg_output",
        data_dir: str = "./primekg_data",
        strategy: str = "map",
        keep_unmapped: bool = False,
        skip_download: bool = False
    ):
        """
        Initialize pipeline

        Args:
            output_dir: Directory for output triples
            data_dir: Directory for downloaded data
            strategy: Conversion strategy (filter/map)
            keep_unmapped: Keep entities without UMLS CUI
            skip_download: Skip download step if files exist
        """
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.strategy = strategy
        self.keep_unmapped = keep_unmapped
        self.skip_download = skip_download

        # File paths
        self.kg_path = self.data_dir / "kg.csv"
        self.mapping_path = self.data_dir / "umls_mondo.csv"
        self.output_path = self.output_dir / "primekg_umls_triples.txt"

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Script paths (in same directory)
        script_dir = Path(__file__).parent
        self.download_script = script_dir / "download_primekg_data.py"
        self.converter_script = script_dir / "primekg_to_umls_triples.py"
        self.validator_script = script_dir / "validate_triples.py"

    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed"""
        logger.info("Checking dependencies...")

        required_packages = ['requests', 'tqdm', 'pandas']
        missing = []

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)

        if missing:
            logger.error(f"Missing packages: {', '.join(missing)}")
            logger.error("Install with: pip install " + " ".join(missing))
            return False

        logger.info("✓ All dependencies installed")
        return True

    def check_scripts(self) -> bool:
        """Check if required scripts exist"""
        logger.info("Checking scripts...")

        scripts = {
            'download_primekg_data.py': self.download_script,
            'primekg_to_umls_triples.py': self.converter_script,
            'validate_triples.py': self.validator_script
        }

        missing = []
        for name, path in scripts.items():
            if not path.exists():
                missing.append(name)
                logger.error(f"✗ Missing: {name}")
            else:
                logger.info(f"✓ Found: {name}")

        if missing:
            logger.error(f"Missing required scripts: {', '.join(missing)}")
            return False

        return True

    def run_command(self, cmd: list, description: str) -> bool:
        """
        Run shell command with logging

        Args:
            cmd: Command as list
            description: Description for logging

        Returns:
            True if successful, False otherwise
        """
        logger.info("")
        logger.info("="*60)
        logger.info(description)
        logger.info("="*60)
        logger.info(f"Command: {' '.join(cmd)}")
        logger.info("")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True
            )
            logger.info("")
            logger.info(f"✓ {description} - SUCCESS")
            return True

        except subprocess.CalledProcessError as e:
            logger.error("")
            logger.error(f"✗ {description} - FAILED")
            logger.error(f"Exit code: {e.returncode}")
            return False

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return False

    def step1_download(self) -> bool:
        """Step 1: Download PrimeKG data"""
        if self.skip_download:
            logger.info("")
            logger.info("="*60)
            logger.info("STEP 1: Download PrimeKG Data - SKIPPED")
            logger.info("="*60)

            # Check if files exist
            if not self.kg_path.exists():
                logger.error(f"kg.csv not found at {self.kg_path}")
                logger.error("Run without --skip-download or provide file manually")
                return False

            if not self.mapping_path.exists():
                logger.error(f"umls_mondo.csv not found at {self.mapping_path}")
                logger.error("Run without --skip-download or provide file manually")
                return False

            logger.info(f"✓ Using existing files:")
            logger.info(f"  - {self.kg_path}")
            logger.info(f"  - {self.mapping_path}")
            return True

        cmd = [
            sys.executable,
            str(self.download_script),
            '--output-dir', str(self.data_dir)
        ]

        return self.run_command(cmd, "STEP 1: Download PrimeKG Data")

    def step2_convert(self) -> bool:
        """Step 2: Convert to UMLS CUI triples"""
        cmd = [
            sys.executable,
            str(self.converter_script),
            str(self.kg_path),
            str(self.output_path),
            '--strategy', self.strategy
        ]

        # Add mapping file if using map strategy
        if self.strategy == 'map':
            cmd.extend(['--mapping', str(self.mapping_path)])

        # Add keep-unmapped if requested
        if self.keep_unmapped:
            cmd.append('--keep-unmapped')

        return self.run_command(cmd, "STEP 2: Convert to UMLS CUI Triples")

    def step3_validate(self) -> bool:
        """Step 3: Validate output"""
        if not self.validator_script.exists():
            logger.warning("Validator script not found, skipping validation")
            return True

        cmd = [
            sys.executable,
            str(self.validator_script),
            str(self.output_path)
        ]

        return self.run_command(cmd, "STEP 3: Validate Output")

    def run(self) -> bool:
        """Run complete pipeline"""
        logger.info("")
        logger.info("="*60)
        logger.info("PRIMEKG TO UMLS CUI TRIPLES - COMPLETE PIPELINE")
        logger.info("="*60)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Data directory:   {self.data_dir}")
        logger.info(f"Strategy:         {self.strategy}")
        logger.info(f"Keep unmapped:    {self.keep_unmapped}")
        logger.info(f"Skip download:    {self.skip_download}")
        logger.info("="*60)

        # Check dependencies
        if not self.check_dependencies():
            return False

        # Check scripts
        if not self.check_scripts():
            return False

        # Step 1: Download
        if not self.step1_download():
            logger.error("\n❌ Pipeline failed at Step 1 (Download)")
            return False

        # Step 2: Convert
        if not self.step2_convert():
            logger.error("\n❌ Pipeline failed at Step 2 (Convert)")
            return False

        # Step 3: Validate
        if not self.step3_validate():
            logger.warning("\n⚠️  Validation failed, but conversion completed")
            # Don't fail pipeline, just warning

        # Success!
        logger.info("")
        logger.info("="*60)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info("")
        logger.info("Output file:")
        logger.info(f"  {self.output_path}")
        logger.info("")

        # Show file stats
        if self.output_path.exists():
            size_mb = self.output_path.stat().st_size / (1024 * 1024)
            with open(self.output_path) as f:
                line_count = sum(1 for _ in f)

            logger.info("Statistics:")
            logger.info(f"  Size:    {size_mb:.1f} MB")
            logger.info(f"  Triples: {line_count:,}")
            logger.info("")

        logger.info("Next steps:")
        logger.info(f"  1. Review output: head -20 {self.output_path}")
        logger.info(f"  2. Copy to GFM:   cp {self.output_path} /home/user/GFM/data/kg.txt")
        logger.info("  3. Run Stage 1:   cd /home/user/GFM && python -m gfmrag.workflow.stage1_index_dataset")
        logger.info("="*60)

        return True


def main():
    parser = argparse.ArgumentParser(
        description="PrimeKG to UMLS CUI triples - Complete pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Steps:
  1. Download kg.csv from Harvard Dataverse (~1.5GB)
  2. Download umls_mondo.csv from GitHub (~500KB)
  3. Convert MONDO IDs to UMLS CUIs (reverse mapping)
  4. Generate UMLS CUI-based triples
  5. Validate output

Examples:
  # Full auto mode (download + convert + validate)
  python primekg_pipeline.py

  # Custom output directory
  python primekg_pipeline.py --output-dir ./my_output

  # Skip download if files exist
  python primekg_pipeline.py --skip-download

  # Filter strategy (UMLS only, no mapping needed)
  python primekg_pipeline.py --strategy filter

  # Map strategy with unmapped entities kept
  python primekg_pipeline.py --strategy map --keep-unmapped

  # Custom data directory
  python primekg_pipeline.py --data-dir ./primekg_cache
        """
    )

    parser.add_argument(
        '--output-dir',
        default='./primekg_output',
        help='Output directory for triples (default: ./primekg_output)'
    )
    parser.add_argument(
        '--data-dir',
        default='./primekg_data',
        help='Data directory for downloads (default: ./primekg_data)'
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
        '--skip-download',
        action='store_true',
        help='Skip download if files exist in data-dir'
    )

    args = parser.parse_args()

    # Run pipeline
    try:
        pipeline = PrimeKGPipeline(
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            strategy=args.strategy,
            keep_unmapped=args.keep_unmapped,
            skip_download=args.skip_download
        )

        success = pipeline.run()
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("\n\nPipeline cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
