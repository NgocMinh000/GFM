#!/usr/bin/env python3
"""
UMLS Mapping Pipeline - Workflow Entry Point

Simplified entry point for Stage 3 UMLS Mapping Pipeline.

Usage:
    python workflow/stage3umlsmapping.py
    python -m workflow.stage3umlsmapping
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Import and run the main pipeline runner
from run_umls_pipeline import main

if __name__ == '__main__':
    # Print banner
    print("="*70)
    print("STAGE 3: UMLS MAPPING PIPELINE")
    print("="*70)
    print()

    # Run main pipeline
    main()
