#!/usr/bin/env python3
"""
Quick test script to verify UMLS setup for Stage 3 training.

This script checks:
1. UMLS data files exist
2. UMLS can be loaded successfully
3. Prints basic statistics

Usage:
    python test_umls_setup.py
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def check_files():
    """Check if required UMLS files exist"""
    logger.info("="*80)
    logger.info("STEP 1: Checking UMLS Data Files")
    logger.info("="*80)

    required_files = {
        'MRCONSO.RRF': 'data/umls/META/MRCONSO.RRF',
        'MRSTY.RRF': 'data/umls/META/MRSTY.RRF',
    }

    all_exist = True
    for name, path in required_files.items():
        path_obj = Path(path)
        if path_obj.exists():
            size_mb = path_obj.stat().st_size / (1024 * 1024)
            logger.info(f"✓ {name}: {path} ({size_mb:.1f} MB)")
        else:
            logger.error(f"✗ {name}: {path} NOT FOUND")
            all_exist = False

    if not all_exist:
        logger.error("\nSome required files are missing!")
        logger.error("Please ensure UMLS data is downloaded to data/umls/META/")
        return False

    logger.info("\n✓ All required files found!")
    return True


def test_umls_loader():
    """Test UMLS loader"""
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Testing UMLS Loader")
    logger.info("="*80)

    try:
        from gfmrag.umls_mapping.umls_loader import UMLSLoader
        from gfmrag.umls_mapping.config import UMLSMappingConfig

        # Create config
        config = UMLSMappingConfig(
            kg_clean_path='dummy',  # Not needed for loading only
            umls_data_dir='data/umls',
            output_root='tmp/umls_test',
            mrconso_path='data/umls/META/MRCONSO.RRF',
            mrsty_path='data/umls/META/MRSTY.RRF',
            umls_cache_dir='data/umls/processed',
        )

        logger.info("Loading UMLS data...")
        logger.info("(First time may take 30-60 minutes, subsequent loads use cache)")

        # Load UMLS
        umls = UMLSLoader(config)
        umls.load()

        # Print statistics
        logger.info("\n" + "="*80)
        logger.info("UMLS Loading Statistics")
        logger.info("="*80)
        logger.info(f"Total concepts: {len(umls.concepts):,}")

        if umls.stats:
            logger.info(f"Lines processed: {umls.stats.get('lines_processed', 0):,}")
            logger.info(f"English terms: {umls.stats.get('lines_english', 0):,}")
            logger.info(f"Total aliases: {umls.stats.get('total_aliases', 0):,}")
            logger.info(f"Avg aliases per concept: {umls.stats.get('avg_aliases_per_concept', 0):.2f}")
            logger.info(f"Concepts with semantic types: {umls.stats.get('concepts_with_semantic_types', 0):,}")

        # Sample a few concepts
        logger.info("\n" + "="*80)
        logger.info("Sample Concepts (first 5)")
        logger.info("="*80)

        for i, (cui, concept) in enumerate(list(umls.concepts.items())[:5]):
            logger.info(f"\n{i+1}. CUI: {cui}")
            logger.info(f"   Preferred Name: {concept.preferred_name}")
            logger.info(f"   Aliases: {', '.join(concept.aliases[:3])}{'...' if len(concept.aliases) > 3 else ''}")
            logger.info(f"   Semantic Types: {', '.join(concept.semantic_types)}")

        logger.info("\n" + "="*80)
        logger.info("✓ UMLS Loading Test PASSED")
        logger.info("="*80)

        return True

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please ensure gfmrag package is installed")
        return False
    except Exception as e:
        logger.error(f"Error loading UMLS: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    logger.info("UMLS Setup Verification Script")
    logger.info("="*80)

    # Step 1: Check files
    if not check_files():
        logger.error("\n❌ UMLS setup verification FAILED (files missing)")
        sys.exit(1)

    # Step 2: Test loader
    if not test_umls_loader():
        logger.error("\n❌ UMLS setup verification FAILED (loading error)")
        sys.exit(1)

    # Success
    logger.info("\n" + "="*80)
    logger.info("✅ UMLS Setup Verification PASSED")
    logger.info("="*80)
    logger.info("\nYou can now proceed with Stage 3 training!")
    logger.info("Next step: python -m gfmrag.umls_mapping.training.cross_encoder_trainer")


if __name__ == "__main__":
    main()
