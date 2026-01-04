#!/usr/bin/env python3
"""
Simplified UMLS Mapping Pipeline Runner

Usage:
    # Run complete pipeline with defaults
    python run_umls_pipeline.py

    # Run specific stages
    python run_umls_pipeline.py --stages stage1_preprocessing stage2_setup_sapbert

    # Resume from last checkpoint
    python run_umls_pipeline.py --resume

    # Check status
    python run_umls_pipeline.py --status

    # Use custom paths
    python run_umls_pipeline.py --umls-dir /path/to/umls --kg-file /path/to/kg_clean.txt
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from gfmrag.umls_mapping import UMLSMappingPipeline

# ============================================================================
# DEFAULT CONFIGURATION (Embedded)
# ============================================================================

DEFAULT_CONFIG = {
    # Paths
    'paths': {
        'umls_dir': './data/umls',
        'kg_file': './data/kg_clean.txt',
        'output_dir': './tmp/umls_mapping',
        'cache_dir': './tmp/umls_mapping/cache'
    },

    # Stage 0: UMLS Loading
    'stage0': {
        'mrconso_file': 'MRCONSO.RRF',
        'mrsty_file': 'MRSTY.RRF',
        'mrdef_file': 'MRDEF.RRF',
        'semantic_types_filter': [
            'T047',  # Disease or Syndrome
            'T048',  # Mental or Behavioral Dysfunction
            'T191',  # Neoplastic Process
            'T046',  # Pathologic Function
            'T184',  # Sign or Symptom
            'T033',  # Finding
            'T037',  # Injury or Poisoning
            'T049',  # Cell or Molecular Dysfunction
            'T190',  # Anatomical Abnormality
            'T019',  # Congenital Abnormality
        ],
        'cache_file': 'umls_cache.pkl'
    },

    # Stage 1: Preprocessing
    'stage1': {
        'min_entity_length': 3,
        'max_entity_length': 100,
        'use_normalization': True,
        'expand_abbreviations': True
    },

    # Stage 2: Setup
    'stage2_setup': {
        'sapbert': {
            'model_name': 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
            'batch_size': 2048,  # Optimized: 8x larger
            'use_amp': True,     # FP16 mixed precision
            'use_multi_gpu': True,
            'device': 'cuda',
            'max_length': 64,
            'embeddings_file': 'umls_embeddings.pkl',
            'faiss_index_file': 'umls_faiss.index',
            'cui_order_file': 'umls_cui_order.pkl'
        },
        'tfidf': {
            'ngram_range': [3, 3],  # Character trigrams
            'max_features': 100000,
            'min_df': 2,
            'vectorizer_file': 'tfidf_vectorizer.pkl',
            'tfidf_matrix_file': 'tfidf_matrix.npz'
        }
    },

    # Stage 2: Candidate Generation
    'stage2': {
        'top_k_sapbert': 64,
        'top_k_tfidf': 64,
        'rrf_k': 60,
        'final_top_k': 128
    },

    # Stage 3: Cluster Aggregation
    'stage3': {
        'top_k': 64,
        'min_cluster_support': 0.3,
        'score_weight': 0.6,
        'consensus_weight': 0.4
    },

    # Stage 4: Hard Negative Filtering
    'stage4': {
        'top_k': 32,
        'confusion_threshold': 0.7,
        'penalty_factor': 0.5
    },

    # Stage 5: Reranking
    'stage5': {
        'cross_encoder_model': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        'batch_size': 32,
        'use_cross_encoder': False,  # Set to True if you have cross-encoder
        'sapbert_weight': 0.6,
        'tfidf_weight': 0.4
    },

    # Stage 6: Final Output
    'stage6': {
        'confidence_threshold': 0.5,
        'margin_weight': 0.35,
        'score_weight': 0.25,
        'consensus_weight': 0.25,
        'diversity_weight': 0.15
    },

    # Runtime settings
    'runtime': {
        'num_workers': 4,
        'verbose': True,
        'save_intermediate': True,
        'use_cache': True
    }
}


# ============================================================================
# Helper Functions
# ============================================================================

def ensure_directories(config):
    """Create necessary directories if they don't exist"""
    paths_to_create = [
        config['paths']['output_dir'],
        config['paths']['cache_dir'],
    ]

    for path in paths_to_create:
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory ready: {path}")


def validate_prerequisites(config):
    """Validate that required files and directories exist"""
    errors = []

    # Check UMLS directory
    umls_dir = Path(config['paths']['umls_dir'])
    if not umls_dir.exists():
        errors.append(f"UMLS directory not found: {umls_dir}")
    else:
        # Check required UMLS files
        required_files = [
            config['stage0']['mrconso_file'],
            config['stage0']['mrsty_file'],
            config['stage0']['mrdef_file']
        ]
        for fname in required_files:
            fpath = umls_dir / fname
            if not fpath.exists():
                errors.append(f"UMLS file not found: {fpath}")

    # Check KG file
    kg_file = Path(config['paths']['kg_file'])
    if not kg_file.exists():
        errors.append(f"Knowledge Graph file not found: {kg_file}")

    if errors:
        print("❌ Prerequisites validation failed:\n")
        for error in errors:
            print(f"  • {error}")
        print("\nPlease ensure:")
        print("  1. UMLS files are downloaded to data/umls/")
        print("     - MRCONSO.RRF")
        print("     - MRSTY.RRF")
        print("     - MRDEF.RRF")
        print("  2. Knowledge Graph file exists at data/kg_clean.txt")
        print("\nSee docs/DEPLOYMENT_GUIDE.md for details.")
        return False

    print("✓ Prerequisites validation passed")
    return True


def print_pipeline_info(config):
    """Print pipeline configuration summary"""
    print("\n" + "="*70)
    print("UMLS MAPPING PIPELINE - Configuration")
    print("="*70)
    print(f"\n📂 Paths:")
    print(f"  UMLS Directory:  {config['paths']['umls_dir']}")
    print(f"  KG File:         {config['paths']['kg_file']}")
    print(f"  Output Directory: {config['paths']['output_dir']}")

    print(f"\n⚙️  Stage 2 Setup (Optimized):")
    print(f"  SapBERT Model:   {config['stage2_setup']['sapbert']['model_name']}")
    print(f"  Batch Size:      {config['stage2_setup']['sapbert']['batch_size']} (8x optimized)")
    print(f"  Mixed Precision: {config['stage2_setup']['sapbert']['use_amp']} (FP16)")
    print(f"  Multi-GPU:       {config['stage2_setup']['sapbert']['use_multi_gpu']}")

    print(f"\n🎯 Pipeline Stages:")
    stages = [
        "Stage 0: UMLS Database Loading",
        "Stage 1: Entity Extraction & Preprocessing",
        "Stage 2: SapBERT Setup (Embeddings + FAISS)",
        "Stage 2: TF-IDF Setup (Vectorizer)",
        "Stage 2: Candidate Generation (128 candidates)",
        "Stage 3: Cluster Aggregation (64 candidates)",
        "Stage 4: Hard Negative Filtering (32 candidates)",
        "Stage 5: Cross-Encoder Reranking",
        "Stage 6: Final Output with Confidence Scores"
    ]
    for stage in stages:
        print(f"  • {stage}")

    print(f"\n⏱️  Expected Runtime (Optimized):")
    print(f"  Stage 0-1:       ~10-15 minutes")
    print(f"  Stage 2 Setup:   ~30-45 minutes (3-6x faster!)")
    print(f"  Stage 2-6:       ~15-20 minutes")
    print(f"  TOTAL:           ~60-90 minutes")
    print("\n" + "="*70 + "\n")


def run_pipeline(config, args):
    """Run the UMLS mapping pipeline"""

    # Validate prerequisites
    if not args.skip_validation:
        if not validate_prerequisites(config):
            return False

    # Ensure directories exist
    ensure_directories(config)

    # Print configuration
    if not args.quiet:
        print_pipeline_info(config)

    # Initialize pipeline
    print("🚀 Initializing UMLS Mapping Pipeline...\n")
    pipeline = UMLSMappingPipeline(config)

    # Determine stages to run
    stages = args.stages if args.stages else None

    # Run pipeline
    try:
        if args.status:
            # Just show status
            status = pipeline.status.get_status()
            print("📊 Pipeline Status:\n")
            for stage, info in status.items():
                status_icon = "✅" if info['completed'] else "⏳"
                print(f"  {status_icon} {stage}: {info['status']}")
            return True

        elif args.reset:
            # Reset pipeline
            print("🔄 Resetting pipeline status...")
            pipeline.status.reset()
            print("✅ Pipeline reset complete")
            return True

        else:
            # Run pipeline
            success = pipeline.run(
                stages=stages,
                resume=args.resume,
                force=args.force
            )

            if success:
                print("\n" + "="*70)
                print("✅ Pipeline completed successfully!")
                print("="*70)
                print(f"\n📁 Output files in: {config['paths']['output_dir']}")
                print(f"\n📊 Final output: {config['paths']['output_dir']}/final_umls_mappings.json")
                print("\nNext steps:")
                print("  1. Review mapping results")
                print("  2. Run validation: python scripts/final_validation.py")
                print("  3. Check metrics and confidence scores")
                return True
            else:
                print("\n❌ Pipeline failed. Check logs for details.")
                return False

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        print("💡 Run with --resume to continue from last checkpoint")
        return False
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='UMLS Mapping Pipeline - Simplified Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Path overrides
    parser.add_argument('--umls-dir', type=str,
                       help='Path to UMLS directory (default: ./data/umls)')
    parser.add_argument('--kg-file', type=str,
                       help='Path to KG file (default: ./data/kg_clean.txt)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory (default: ./tmp/umls_mapping)')

    # Execution modes
    parser.add_argument('--stages', nargs='+',
                       help='Specific stages to run (e.g., stage1_preprocessing stage2_setup_sapbert)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')
    parser.add_argument('--force', action='store_true',
                       help='Force rerun all stages')
    parser.add_argument('--status', action='store_true',
                       help='Show pipeline status')
    parser.add_argument('--reset', action='store_true',
                       help='Reset pipeline status')

    # Options
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip prerequisites validation')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')

    # Performance options
    parser.add_argument('--batch-size', type=int,
                       help='SapBERT batch size (default: 2048 optimized)')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable mixed precision (FP16)')
    parser.add_argument('--no-multi-gpu', action='store_true',
                       help='Disable multi-GPU')

    args = parser.parse_args()

    # Apply configuration overrides
    config = DEFAULT_CONFIG.copy()

    if args.umls_dir:
        config['paths']['umls_dir'] = args.umls_dir
    if args.kg_file:
        config['paths']['kg_file'] = args.kg_file
    if args.output_dir:
        config['paths']['output_dir'] = args.output_dir
    if args.batch_size:
        config['stage2_setup']['sapbert']['batch_size'] = args.batch_size
    if args.no_amp:
        config['stage2_setup']['sapbert']['use_amp'] = False
    if args.no_multi_gpu:
        config['stage2_setup']['sapbert']['use_multi_gpu'] = False

    # Run pipeline
    success = run_pipeline(config, args)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
