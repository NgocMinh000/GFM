#!/usr/bin/env python3
"""
UMLS Mapping Pipeline - Automated Workflow

Complete end-to-end UMLS entity mapping pipeline with:
- Automatic stage orchestration
- Resume capability
- Progress tracking
- Error handling

Usage:
    # Run complete pipeline
    python scripts/run_umls_mapping.py --config config/umls_mapping.yaml

    # Run specific stages
    python scripts/run_umls_mapping.py --config config/umls_mapping.yaml --stages stage2_candidate_generation stage3_cluster_aggregation

    # Resume from last successful stage
    python scripts/run_umls_mapping.py --config config/umls_mapping.yaml --resume

    # Force rerun all stages
    python scripts/run_umls_mapping.py --config config/umls_mapping.yaml --force

    # Check pipeline status
    python scripts/run_umls_mapping.py --config config/umls_mapping.yaml --status

    # Reset pipeline
    python scripts/run_umls_mapping.py --config config/umls_mapping.yaml --reset
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gfmrag.umls_mapping.config_loader import load_config
from gfmrag.umls_mapping.pipeline import UMLSMappingPipeline


def print_header():
    """Print pipeline header"""
    print("=" * 70)
    print("UMLS MAPPING PIPELINE - AUTOMATED WORKFLOW")
    print("=" * 70)
    print()


def print_available_stages():
    """Print available stages"""
    print("Available stages:")
    stages = [
        ("stage0_umls_loading", "Load UMLS database (one-time)"),
        ("stage1_preprocessing", "Entity extraction and preprocessing"),
        ("stage2_setup_sapbert", "SapBERT embeddings + FAISS (one-time, 2-3 hours)"),
        ("stage2_setup_tfidf", "TF-IDF vectorizer (one-time, 10-15 min)"),
        ("stage2_candidate_generation", "Generate 128 candidates per entity"),
        ("stage3_cluster_aggregation", "Cluster aggregation (64 candidates)"),
        ("stage4_hard_negative_filtering", "Hard negative filtering (32 candidates)"),
        ("stage5_cross_encoder_reranking", "Cross-encoder reranking"),
        ("stage6_final_output", "Final output and validation"),
    ]

    for stage, desc in stages:
        print(f"  • {stage:35} - {desc}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="UMLS Mapping Pipeline - Automated Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  %(prog)s --config config/umls_mapping.yaml

  # Run specific stages
  %(prog)s --config config/umls_mapping.yaml --stages stage2_candidate_generation stage3_cluster_aggregation

  # Resume from last successful stage
  %(prog)s --config config/umls_mapping.yaml --resume

  # Force rerun all stages
  %(prog)s --config config/umls_mapping.yaml --force

  # Check pipeline status
  %(prog)s --config config/umls_mapping.yaml --status

  # Reset pipeline
  %(prog)s --config config/umls_mapping.yaml --reset

For more information, see docs/UMLS_MAPPING_PIPELINE.md
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--stages', '-s',
        nargs='+',
        help='Specific stages to run (default: all)'
    )

    parser.add_argument(
        '--resume', '-r',
        action='store_true',
        help='Resume from last successful stage'
    )

    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force rerun completed stages'
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='Show pipeline status and exit'
    )

    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset pipeline status and exit'
    )

    parser.add_argument(
        '--list-stages',
        action='store_true',
        help='List available stages and exit'
    )

    args = parser.parse_args()

    # Print header
    print_header()

    # List stages
    if args.list_stages:
        print_available_stages()
        return 0

    # Load configuration
    try:
        config, runtime_options = load_config(args.config)
        print(f"✓ Loaded config from: {args.config}")
        print(f"  Output directory: {config.output_root}")
        print()
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return 1

    # Create pipeline
    try:
        pipeline = UMLSMappingPipeline(config)
    except Exception as e:
        print(f"✗ Failed to create pipeline: {e}")
        return 1

    # Handle status command
    if args.status:
        status = pipeline.get_status()
        print("Pipeline Status:")
        print(f"  Last run: {status.get('last_run', 'Never')}")
        print(f"  Last successful stage: {status.get('last_successful_stage', 'None')}")
        print(f"  Completed stages: {len(status.get('completed_stages', []))}")
        print(f"  Failed stages: {len(status.get('failed_stages', []))}")
        print()

        if status.get('completed_stages'):
            print("Completed stages:")
            for stage in status['completed_stages']:
                print(f"  ✓ {stage}")
            print()

        if status.get('failed_stages'):
            print("Failed stages:")
            for stage in status['failed_stages']:
                print(f"  ✗ {stage}")
            print()

        return 0

    # Handle reset command
    if args.reset:
        pipeline.reset()
        print("✓ Pipeline status reset")
        return 0

    # Run pipeline
    try:
        # Merge command-line args with config
        stages = args.stages or runtime_options.get('stages')
        resume = args.resume or runtime_options.get('resume', False)
        force = args.force or runtime_options.get('force', False)

        # Run
        success = pipeline.run(
            stages=stages,
            resume=resume,
            force=force
        )

        if success:
            print()
            print("=" * 70)
            print("✅ PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 70)
            print()
            print(f"Results saved to: {config.output_root}")
            print()
            return 0
        else:
            print()
            print("=" * 70)
            print("✗ PIPELINE FAILED")
            print("=" * 70)
            print()
            print("Check logs for details:")
            print(f"  {Path(config.output_root) / 'pipeline.log'}")
            print()
            print("Use --resume to continue from last successful stage")
            print()
            return 1

    except KeyboardInterrupt:
        print()
        print("Pipeline interrupted by user")
        print("Use --resume to continue from last successful stage")
        return 130

    except Exception as e:
        print(f"✗ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
