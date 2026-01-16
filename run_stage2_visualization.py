#!/usr/bin/env python3
"""
Standalone script to generate Stage 2 Entity Resolution visualizations
from existing output files.

Usage:
    python run_stage2_visualization.py [output_dir]

Example:
    python run_stage2_visualization.py tmp/entity_resolution
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

def main():
    # Get output directory from command line or use default
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path("tmp/entity_resolution")

    logger.info(f"Stage 2 Visualization Tool")
    logger.info(f"Output directory: {output_dir}")

    # Check if directory exists
    if not output_dir.exists():
        logger.error(f"Directory not found: {output_dir}")
        logger.error("Please provide a valid output directory containing Stage 2 results.")
        sys.exit(1)

    # Check required files
    required_files = [
        "stage0_entity_types.json",
        "stage1_embeddings.npy",
        "stage1_entity_ids.json"
    ]

    missing_files = []
    for file in required_files:
        if not (output_dir / file).exists():
            missing_files.append(file)

    if missing_files:
        logger.warning(f"Some files are missing: {missing_files}")
        logger.warning("Some plots may be skipped.")

    # Import and run visualization
    logger.info("\nGenerating visualization plots...")

    try:
        from gfmrag.workflow.stage2_visualization import visualize_stage2_metrics

        visualize_stage2_metrics(output_dir)

        logger.info("\n" + "="*80)
        logger.info("✅ Visualization completed!")
        logger.info(f"   Plots saved to: {output_dir / 'visualizations'}")
        logger.info("="*80)

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("\nPlease install required packages:")
        logger.error("  pip install matplotlib seaborn")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
